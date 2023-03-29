import warnings
import numpy as np
import tensorflow as tf

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

# imports from sequentialcopy package
from sequentialcopy.sampler.gaussian_sampler import GaussianSampler
from sequentialcopy.datasets.datasets import moons
from sequentialcopy.model.feedforward_model import FeedForwardModel
from sequentialcopy.model.lstm_model import LSTMmodel
from sequentialcopy.sequential_copy import sequential_train
from sequentialcopy.utils.utils import define_loss, LambdaParameter, dump_pickle, mean_sdt
from sequentialcopy.utils.plots import plot_results
from sequentialcopy.utils.parallel_utils import decode_results,separate_runs

# Create new problem
X, y = moons(1500)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=7)

# Fit model
original = MLPClassifier(random_state=42, hidden_layer_sizes=(42), learning_rate='adaptive', max_iter=100000, activation='relu', solver='adam', alpha=0.01, learning_rate_init=0.001)
original.fit(X_train, y_train)

y_test_pred = original.predict(X_test)
acc = np.average(np.where(y_test_pred==y_test,1.,0.))
print('A0:', acc)

# Define domain
min_, max_ = np.min(X_train-0.5, axis=0), np.max(X_train+0.5, axis=0)
xx, yy = np.meshgrid(np.arange(min_[0], max_[0], .025), np.arange(min_[1], max_[1], .025))
z = original.predict_proba(np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)]))[:,0].reshape(xx.shape)

# Saving figure: Original model
fig = plt.figure(figsize=(6,6)) 
ax = fig.add_subplot(111)
ax.contourf(xx, yy, z, alpha=0.5)
plot = ax.scatter(X_test[:,0], X_test[:,1], c=-y_test, s=30)
fig.savefig("plots/original_model.pdf",bbox_inches='tight')

# setting initial parameters
d = X_test.shape[1]
n_classes = len(np.unique(y_test))

# define optimizer and loss 
lr = 0.0005

# define new model
layers = [] # logistic regression
model_name = 'FeedForward'

# define the memory (lambda) parameter
lmda = 5.0
automatic_lmda = True

#define the sampling process
sampler_name = 'Gaussian'
sampler_from_file = True                              # used to get data from file
sampler_file_path = 'data/moons_synthetic_data.cvs'   # path to synthetic data
sampler_automatic_fill = True                         # allows to add new datapoints to the data file

# setting run parameters
n_samples_iter=100

thres=5e-9
sample_selection=True

max_iter = 15

## --------------------------------------------------------

opt = tf.keras.optimizers.Adam(learning_rate=lr)
loss = define_loss(d, loss_name = 'UncertaintyError')

lmda_ = LambdaParameter(lmda=lmda, automatic_lmda=automatic_lmda)

seq_copy = FeedForwardModel(input_dim=d, hidden_layers=layers, output_dim=n_classes, activation='relu')
seq_copy.compile(loss=loss, optimizer=opt)

sampler = GaussianSampler(d=d, n_classes=n_classes,from_file=sampler_from_file,
                          file_path=sampler_file_path, automatic_fill=sampler_automatic_fill)

trained_model= sequential_train(seq_copy,
                                sampler,
                                original, 
                                n_samples_iter=n_samples_iter, 
                                thres=thres, 
                                lmda_par =lmda_,
                                max_iter=max_iter, 
                                X_test=X_test, 
                                y_test=y_test, 
                                sample_selection=sample_selection,
                                plot_every=False)

# Define domain
min_, max_ = np.min(X_train-0.5, axis=0), np.max(X_train+0.5, axis=0)
xx, yy = np.meshgrid(np.arange(min_[0], max_[0], .025), np.arange(min_[1], max_[1], .025))
z = trained_model.predict(np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)]))[:,0].reshape(xx.shape)

# Saving figure: Copy model
fig = plt.figure(figsize=(6,6)) 
ax = fig.add_subplot(111)
ax.contourf(xx, yy, z, alpha=0.5)
plot = ax.scatter(X_test[:,0], X_test[:,1], c=-y_test, s=30)
fig.savefig("plots/copy_model.pdf",bbox_inches='tight')


acc_test=trained_model.acc_test
acc_train=trained_model.acc_train
rho=trained_model.rho
n=trained_model.n
lmda_vector=trained_model.lmda_vector

## --------------------------------------------------------

#saving raw data
dump_pickle(n,'n')
dump_pickle(acc_train,'acc_train')
dump_pickle(acc_test,'acc_test')
dump_pickle(rho,'rho')
dump_pickle(lmda_vector,'lmda')


# plotting and saving results
plot_results(acc_test, sdt=None, plot_type='acc_test', name='acc_test')
plot_results(acc_train, sdt=None, plot_type='acc_train', name = 'acc_train')
plot_results(rho, sdt=None, plot_type='rho', name = 'rho')
plot_results(n, sdt=None, plot_type='nN', name = 'n')
plot_results(lmda_vector, sdt=None, plot_type='lmda', name = 'lmda')