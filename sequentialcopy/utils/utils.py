import os
import csv
import pandas as pd
import pickle
from collections import Counter
import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.losses import Loss
from tensorflow import keras

from keras.losses import LossFunctionWrapper
from keras.utils import losses_utils

import warnings
warnings.filterwarnings("ignore")

tf.keras.backend.set_floatx('float64')
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def define_loss(d, loss_name = 'UncertaintyError'):
    if loss_name == 'UncertaintyError': 
        return UncertaintyError()
    elif d <= 2:
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)
    else:
        return tf.keras.losses.CategoricalCrossentropy(from_logits=True)

def dump_pickle(data,name,path='results'):
    """save data to pickle file. Used in tutorials"""    
    with open(os.path.join(path,name+'.pickle'), 'wb+') as f:
        pickle.dump(data, f)
    
def mean_sdt(vec):
    """return mean and sdt of a list of lists. Used in tutorials"""
    return np.mean(vec, axis=0), np.std(vec, axis=0)
    
def norm_theta(vec1,vec2):    
    """Given two parameter's vectors, return the Euclidean norm."""
    t = tf.constant(0.0, dtype = tf.float64)
    for i in range(np.shape(vec1)[0]):
        t = tf.add(t,tf.reduce_sum(tf.square(vec1[i] - vec2[i])))
    return tf.cast(t, tf.float64)

class LambdaParameter():
    """A class for managing the lambda parameter used for regularization."""
    
    def __init__(self, lmda=0.0, automatic_lmda=False, divider=2, multiplier=1.5):
        """Initializes a new instance of the LambdaParameter class.

            Args:
            - lmda (float): the value of lambda to use.
            - automatic_lmda (bool): whether or not to update lambda automatically.
            - divider (int): the factor by which to divide lambda when updating it.
            - multiplier (float): the factor by which to multiply lambda when updating it.
        """
        super(LambdaParameter, self).__init__()
        self.lmda = lmda
        self.automatic = automatic_lmda
        self.divider = divider
        self.multiplier = multiplier

    def update(self, nN_prev, nN, n_sampling):
        """Updates lambda automatically based on the number of samples before and after the selection process.

        Args:
        - nN_prev (int): the number of samples before the selection.
        - nN (int): the number of samples after the selection.
        - n_sampling (int): the number of samples generate at each step.
        """
        if self.automatic:
            if (nN_prev - nN)<n_sampling:
                self.lmda = self.lmda/self.divider
            elif (nN_prev - nN)>=n_sampling:
                self.lmda = self.lmda*self.multiplier

class UncertaintyError(LossFunctionWrapper):
    """Computes the norm between hard labels and soft predictions."""
    def __init__(
        self, reduction=losses_utils.ReductionV2.AUTO, name="uncertainty_error"
    ):
        """Initializes `UncertaintyError` instance.
        Args:
          reduction: Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `tf.keras` `compile` and `fit`, using `AUTO` or
            `SUM_OVER_BATCH_SIZE` will raise an error. Please see this custom
            training [tutorial](
            https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
          name: Optional name for the instance. Defaults to
            'uncertainty_error'.
        """
        super().__init__(uncertainty_error, name=name, reduction=reduction)
    

def uncertainty_error(y_true, y_pred):
    """Calculate the uncertainty error given the hard-labeled points and the predicted points."""
    
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    num_classes = tf.cast(tf.shape(y_true)[-1], y_pred.dtype)

    return tf.divide(tf.reduce_sum(tf.math.squared_difference(y_pred, y_true),1, keepdims=True), num_classes)

class LayerRegularizer(regularizers.Regularizer):
    """Custom regularization class inheriting from keras Regularizer."""
    
    def __init__(self, layer_num, model):
        """
        Initializes LayerRegularizer instance.

        Parameters:
        layer_num (int): the layer number to which the regularization will be applied.
        model (object): the neural network model to which this layer belongs to.
        """
        self.layer_num = layer_num
        self.model = model

    def __call__(self, x):
        """Calculates the regularization term value."""
        
        self.new_weigths = self.get_weigths(self.model.layers[self.layer_num])
        self.theta0 = self.model.theta0[self.layer_num]
        return tf.divide(tf.reduce_sum(tf.square(self.theta0-self.new_weigths)),
                         tf.constant(self.model.weights_dims, dtype = tf.float64))
    
    def get_weigths(self, layer):
        """Retrieves the weights and bias of the given layer and returns them concatenated.
        Parameters:
        layer (object): the layer for which weights and bias will be retrieved.

        Returns:
        tensor: concatenated weights and bias of the layer.
        """
        #weights of the layer
        t0 = tf.reshape(layer.trainable_variables[0],
                        [np.shape(layer.trainable_variables[0])[0]*np.shape(layer.trainable_variables[0])[1]])
        #bias of the layer
        t1 = tf.reshape(layer.trainable_variables[1],
                        [np.shape(layer.trainable_variables[1])[0]])
        return tf.concat([t0,t1],0)