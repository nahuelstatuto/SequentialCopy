import warnings
import numpy as np
import tensorflow as tf

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

# imports from sequentialcopy package
from sequentialcopy.sampler.gaussian_sampler import GaussianSampler
from sequentialcopy.datasets.datasets import spirals
from sequentialcopy.model.feedforward_model import FeedForwardModel
from sequentialcopy.sequential_copy import sequential_train
from sequentialcopy.utils.utils import define_loss, LambdaParameter, dump_pickle, mean_sdt
from sequentialcopy.utils.plots import plot_results
from sequentialcopy.utils.parallel_utils import decode_results,separate_runs

# Create new problem
X, y = spirals(1500)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=7)

# Fit model
original = SVC(random_state=42, kernel='rbf', probability=True, gamma=10)
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
layers = [64,32,10]
model_name = 'FeedForward'
#seq_copy = FeedForwardModel(input_dim=d, hidden_layers=layers, output_dim=n_classes, activation='relu')
#seq_copy.compile(loss=loss, optimizer=opt)

# define the memory (lambda) parameter
lmda = 5.0
automatic_lmda = True
#lmda_ = LambdaParameter(lmda=lmda, automatic_lmda=automatic_lmda)

#define the sampling process
sampler_name = 'Gaussian'
#sampler = GaussianSampler(d=d, n_classes=n_classes)

# setting run parameters
n_samples_iter=100

thres=5e-9
sample_selection=True

max_iter = 30 # 30 in the original figure
n_runs = 30 # 30 in the original figure

results = Parallel(n_jobs=-1, backend='threading')(delayed(separate_runs)(original,
                                                                          model_name=model_name ,
                                                                          sampler_name=sampler_name,
                                                                          lr=lr,
                                                                          lmda=lmda,
                                                                          automatic_lmda=automatic_lmda,
                                                                          n_samples_iter=n_samples_iter, 
                                                                          thres=thres, 
                                                                          max_iter=max_iter, 
                                                                          X_test=X_test, 
                                                                          y_test=y_test, 
                                                                          layers = layers,
                                                                          sample_selection=sample_selection)  for i in range(0, n_runs))

n, acc_train, acc_test, rho, lmda_vector = decode_results(results)  

#saving raw data
dump_pickle(n,'n')
dump_pickle(acc_train,'acc_train')
dump_pickle(acc_test,'acc_test')
dump_pickle(rho,'rho')
dump_pickle(lmda_vector,'lmda')


#getting means and sdt 
acc_test_mean, acc_test_sdt = mean_sdt(acc_test)
acc_train_mean, acc_train_sdt = mean_sdt(acc_train)
rho_mean, rho_sdt = mean_sdt(rho)
n_mean, n_sdt = mean_sdt(n)
lmda_mean, lmda_sdt = mean_sdt(lmda_vector)


# plotting and saving results
plot_results(acc_test_mean, sdt=acc_test_sdt, plot_type='acc_test', name='acc_test')
plot_results(acc_train_mean, sdt=acc_train_sdt, plot_type='acc_train', name = 'acc_train')
plot_results(rho_mean, sdt=rho_sdt, plot_type='rho', name = 'rho')
plot_results(n_mean, sdt=n_sdt, plot_type='nN', name = 'n')
plot_results(lmda_mean, sdt=lmda_sdt, plot_type='lmda', name = 'lmda')