import warnings
import numpy as np
import tensorflow as tf

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from joblib import Parallel, delayed

# imports from sequentialcopy package
import sequentialcopy.sampling as sp
import sequentialcopy.parallel_utils as tt
from sequentialcopy.datasets import spirals
from sequentialcopy.utils import define_loss, LambdaParameter
from sequentialcopy.models import FeedForwardModel
from sequentialcopy.sequential_copy import sequential_train
from sequentialcopy.plots import plot_results

# Create new problem
#X, y = make_moons(**{'n_samples':1500, 'shuffle':True, 'noise':0.2, 'random_state':42})
X, y = spirals(1500)
X = StandardScaler(copy=True).fit_transform(X)
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

lr = 0.0005
layers = [64,32,10] # leave empty [] for a regression
lmda = 0.0
automatic_lmda = False
n_samples_iter=100
thres=1e-9
sample_selection=False

# define optimizer and loss 
opt = tf.keras.optimizers.Adam(learning_rate=lr)
loss = define_loss(d, loss_name = 'UncertaintyError')

# define new model
seq_copy = FeedForwardModel(input_dim=d, hidden_layers=layers, output_dim=n_classes, activation='relu')
seq_copy.compile(loss=loss, optimizer=opt)

# define the memory (lambda) parameter
lmda_ = LambdaParameter(lmda=lmda, automatic_lmda=automatic_lmda)

#define the sampling process
sampler = sp.GaussianSampler(d=d, n_classes=n_classes)

# setting run parameters
max_iter = 15 # 30 in the original figure
n_runs = 10 # 30 in the original figure

results = Parallel(n_jobs=-1, backend='threading')(delayed(tt.separate_runs)(original,
                                                                          max_iter=max_iter,
                                                                          X_test=X_test,
                                                                          y_test=y_test)  for i in range(0, n_runs))

n, acc_train, acc_test, rho, lmda_vector = tt.decode_results(results)

acc_test_mean = np.mean(acc_test, axis=0)
acc_test_sdt = np.std(acc_test, axis=0)
acc_train_mean = np.mean(acc_train, axis=0)
acc_train_sdt = np.std(acc_train, axis=0)

# plotting and saving results
plot_results(acc_test_mean, sdt=acc_test_sdt, plot_type='acc_test')
plot_results(acc_train_mean, sdt=acc_train_sdt, plot_type='acc_train')

print('n: ',n)
print('acc_train: ',acc_train)
print('acc_test: ',acc_test)
print('rho: ',rho)
print('lmda_vector: ',lmda_vector)

