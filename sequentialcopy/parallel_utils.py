import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from joblib import Parallel, delayed

from sequentialcopy.gaussian_sampler import GaussianSampler
from sequentialcopy.utils import define_loss, LambdaParameter
from sequentialcopy.models import FeedForwardModel
from sequentialcopy.sequential_copy import sequential_train

def separate_runs(original,
                  lr=0.0005,
                  n_samples_iter=100, 
                  thres=1e-9, 
                  lmda_par = 0.0,
                  automatic_lmda=False,
                  max_iter=3, 
                  X_test=None, 
                  y_test=None, 
                  layers = [64,32,10],
                  sample_selection=False,
                  plot_every=False):
    """ Function used to run in parallel. Generate the copy model, optimizer and loss from parameters and train the model"""

    d = X_test.shape[1]
    n_classes = len(np.unique(y_test))
    
    # define optimizer and loss 
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = define_loss(d, loss_name = 'UncertaintyError')

    # define new model
    seq_copy = FeedForwardModel(input_dim=d, hidden_layers=layers, output_dim=n_classes, activation='relu')
    seq_copy.compile(loss=loss, optimizer=opt)

    # define the memory (lambda) parameter
    lmda = LambdaParameter(lmda=lmda_par, automatic_lmda=automatic_lmda)

    #define the sampling process using a Gaussian distribution
    sampler = GaussianSampler(d=d, n_classes=n_classes)
    
    return sequential_train(seq_copy,
                            sampler,
                            original, 
                            n_samples_iter=n_samples_iter, 
                            thres=thres, 
                            lmda_par =lmda,
                            max_iter=max_iter, 
                            X_test=X_test, 
                            y_test=y_test, 
                            sample_selection=sample_selection,
                            plot_every=plot_every)

def decode_results(data):
    """Function used to decode the results from many parallels runs. Returns an arrays for each run"""
    n, acc_train, acc_test, rho, lmda = ([] for i in range(5))
    
    for i in range(len(data)):
        n.append(list(data[i].n))
        acc_train.append(list(data[i].acc_train))
        acc_test.append(list(data[i].acc_test))
        rho.append(list(data[i].rho))
        lmda.append(list(data[i].lmda_vector))
        
    return np.array(n), np.array(acc_train), np.array(acc_test), np.array(rho), np.array(lmda)