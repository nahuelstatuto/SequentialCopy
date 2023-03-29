import os
import logging
import numpy as np
import tensorflow as tf
from keras.utils import losses_utils

from sequentialcopy.utils.utils import LambdaParameter
from sequentialcopy.model.feedforward_model import params_to_vec
import sequentialcopy.utils.plots as pt

import warnings
warnings.filterwarnings("ignore")

tf.keras.backend.set_floatx('float64')
logging.getLogger('tensorflow').setLevel(logging.FATAL)
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def sequential_train(model, 
                     sampler, 
                     original, 
                     n_samples_iter=100, 
                     max_iter=1, 
                     epochs=1000, 
                     batch_size=32, 
                     max_subtrain=2,
                     sample_selection=False,
                     thres=1e-9,
                     lmda_par = LambdaParameter(), 
                     verbose=False,
                     X_test=None,
                     y_test=None,
                     plot_every=False):
    """
    Trains a given parametric model using the sequential training method.

    Args:
        model: A Keras parametric model to train.
        sampler: A sampler object to generate/read new samples.
        original: The original model to be used as an Oracle.
        n_samples_iter: Number of samples to feed the model at each iteration.
        max_iter: Maximum number of iterations to run.
        epochs: Number of epochs to train the model on each subtraining set.
        batch_size: Batch size used in training.
        max_subtrain: Maximum number of subtrainings to perform.
        sample_selection: Whether to perform sample selection before training.
        thres: Threshold value used in sample selection.
        lmda_par: A LambdaParameter object used in adjusting regularization strength.
        verbose: Whether to print progress messages during training.
        X_test: Test input data.
        y_test: Test target data.
        plot_every: Whether to plot model state at every iteration.

    Returns:
        The trained Keras model.
    """
    t = 0  
    lr = model.optimizer.lr.numpy()
    rho_max = tf.constant(1.0, dtype = tf.float64)
    n_subtrain = 0
    
    X_train, y_train = np.empty((0, sampler.d)), np.empty((0), dtype=int)
    
    while t < max_iter: 
        
        # Generate new 'n_samples_iter' samples
        X, y = sampler.get_samples(original, n_samples_iter)
        
        # New data is marge with previous data
        X_train, y_train = np.vstack((X, X_train)), np.append(y, y_train)
        nN_prev=len(X_train)
        
        # Selection of data before training 
        if sample_selection:
            X_train, y_train = sample_selection_policy(model, X_train, y_train, sampler.d, sampler.n_classes, thres)            
        
        model.n.append(len(X_train))
        
        # Depending on the number of deleted datapoints, lambda is updated
        lmda_par.update(nN_prev, len(X_train), n_samples_iter)
        lmda = lmda_par.lmda
        
        y_errors = y_train
        rho_mean = model.loss(tf.one_hot(y_train, sampler.n_classes), model.predict(X_train, verbose=0)).numpy()
        
        model.theta0, model.weights_dims = params_to_vec(model, return_dims=True)
        
        while len(y_errors)!=0 and n_subtrain<=max_subtrain:
            if n_subtrain > 0:
                #  During subtrainig, lr is increased by a factor of 2.0, 1.33, 1.2, 1.14, ....
                model.optimizer.lr=model.optimizer.lr*(n_subtrain-1+1)/(n_subtrain-1+0.5)
            
            # Training the model
            y_ohe = tf.one_hot(y_train, sampler.n_classes) 
            model.fit(X_train, 
                      y_ohe, 
                      lmda, 
                      rho_max, 
                      epochs=epochs, 
                      batch_size=batch_size, 
                      verbose=0)
            
            rho_max = model.loss(tf.one_hot(y_train, sampler.n_classes), model.predict(X_train, verbose=0)).numpy()
            
            #y_pred_ohe = model.predict(X_train, verbose=0)
            y_pred_ohe = model.predict(X_train.reshape(np.shape(X_train)[0],np.shape(X_train)[1],1), verbose=0)
            y_pred = np.argmax(y_pred_ohe, axis=1)
            X_errors = X_train[y_pred!=y_train,:]
            y_errors = y_train[y_pred!=y_train]
        
            n_subtrain +=1
       
        model.optimizer.lr = lr
        n_subtrain = 0
        t += 1
        
        if X_test is not None:
            acc_test = evaluation(model, X_test, y_test, sampler.d, sampler.n_classes)
            model.acc_test.append(acc_test)
        acc_train = evaluation(model, X_train, y_train, sampler.d, sampler.n_classes)
        model.acc_train.append(acc_train)
        model.rho.append(rho_max)
        model.lmda_vector.append(lmda)
        
        # If input dimension is 2, then the decision boundary can be plot
        if plot_every and sampler.d == 2:
            pt.plot_copy_model(model,X_train,y_train,X_errors,y_errors)
        
    return model

def evaluation(model, X, y,d,n_classes):
    """evaluate the model accuracy."""

    y_pred = model.predict(X,verbose=0)
    y_pred = np.argmax(y_pred, axis=1)
    
    return sum(y_pred == y)/len(X)

def sample_selection_policy(model, X_train, y_train, d, n_classes, thresh):
    """Funtion used to select datapoint with an Uncertainty (loss value) above a pre-defined threshold."""
    X = np.empty((0, d))
    y = np.empty((0), dtype=int)
    y_pred = model.predict(X_train, verbose=0)
    
    #reduction parameter is changed to NONE to get a loss value for each datapoint
    model.loss.reduction = losses_utils.ReductionV2.NONE
    rho = model.loss(tf.one_hot(y_train, n_classes), y_pred).numpy()
    
    #reduction parameter set again to AUTO to get an average of all the datapoint
    model.loss.reduction = losses_utils.ReductionV2.AUTO
    
    for i, r in enumerate(rho>=thresh):
        if r:
            y = np.append(y,y_train[i])
            X = np.append(X,[X_train[i]], axis=0)
    
    #the function can not return empty X, y
    if len(X)==0:
        try:
            nN = np.random.randint(0,len(X_train),int(len(X_train)/2))
            for n in nN:
                y = np.append(y,y_train[n])
                X = np.append(X,[X_train[n]], axis=0)
        except:
            pass
    return X, y
