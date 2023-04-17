import tensorflow as tf
import numpy as np
from keras.utils.layer_utils import count_params
tf.keras.backend.set_floatx('float64')

from sequentialcopy.utils.utils import LayerRegularizer, params_to_vec
from sequentialcopy.model.base_model import CopyModel

class FeedForwardModel(CopyModel):
    """Class to create, compile and train a Keras model."""
    
    def __init__(self,
                 input_dim,
                 hidden_layers,
                 output_dim,
                 activation='relu',
                 seed=42):
        """
        Args:
            input_dim: an integer representing the number of input features.
            hidden_layers: a list of integers representing the number of neurons in each hidden layer.
            output_dim: an integer representing the number of output classes.
            activation: a string representing the activation function to use (default is 'relu').
            seed: an integer representing the random seed (default is 42).
        """
        super(FeedForwardModel, self).__init__()
        tf.random.set_seed(seed)
        
        # Initialize several instance variables
        self.acc_test = []
        self.acc_train = []
        self.rho = []
        self.n = []
        self.lmda_vector = []
        self.dense = []
        
        # Create a list of dense layers, with the first one taking the input dimension as input shape
        if len(hidden_layers) > 0:
            self.dense.append(tf.keras.layers.Dense(hidden_layers[0], 
                                                  input_shape=(input_dim,), 
                                                  activation=activation,
                                                  kernel_initializer='he_normal', 
                                                  kernel_regularizer=LayerRegularizer(layer_num=0, model=self),  
                                                  name='layer0', autocast=False))
            if len(hidden_layers) > 1:
                for layer, n_neurons in enumerate(hidden_layers[1:]):
                    self.dense.append(tf.keras.layers.Dense(n_neurons, 
                                                            activation=activation, 
                                                            kernel_initializer='he_normal',
                                                            kernel_regularizer=LayerRegularizer(layer_num=layer+1, model=self),
                                                            name='layer{}'.format(layer+1)))
        self.dense.append(tf.keras.layers.Dense(output_dim, 
                                                activation='softmax',
                                                kernel_regularizer=LayerRegularizer(layer_num=len(hidden_layers), model=self),
                                                name='layer{}'.format(len(hidden_layers))))
        self.theta0 = None
    
    def call(self, inputs):
        x = self.layers[0](inputs)
        if len(self.layers) > 1:
            for layer in range(1,len(self.layers)):
                x = self.layers[layer](x)
        return x
    
    def fit(self, x, y, lmda, rho_max, epochs, batch_size, verbose):
        self.lmda = lmda
        self.rho_max = rho_max
        return super(FeedForwardModel,self).fit(x,y, epochs=epochs, batch_size=batch_size, verbose=0)
    
    def compile(self, optimizer, loss):
        """Set the optimizer and the loss function."""
        super(FeedForwardModel, self).compile()
        self.optimizer = optimizer
        self.loss = loss
        
    def train_step(self, data):
        """
            Define the train_step method for the model, which performs a forward pass, computes the loss and gradients,
            updates the weights, and returns a dictionary of metric values.
        """
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            reg = tf.tanh(tf.math.reduce_sum(self.losses)) # reg <= 1
            
            # Compute the loss value (the loss function is configured in `compile()`)
            rho = self.loss(y, y_pred) / self.rho_max
            loss = rho + self.lmda * reg
            
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        
        # Return a dict mapping metric names to current value
        return {'loss' :loss,
                'reg' : reg,
                'rho' : rho}
    
    def update_theta0(self):
        self.theta0 = params_to_vec(model=self)
        self.weights_dims = count_params(self.trainable_weights)
        