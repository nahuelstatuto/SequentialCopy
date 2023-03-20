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
import sequentialcopy.tutorials_utils as tt
from sequentialcopy.datasets import spirals
from sequentialcopy.utils import define_loss
from sequentialcopy.utils import LambdaParameter
from sequentialcopy.models import FeedForwardModel
from sequentialcopy.sequential_copy import sequential_train

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

# setting initial parameters

d = X_test.shape[1]
n_classes = len(np.unique(y_test))

lr = 0.0005
layers = [64,32,10]
lmda = 0.0
automatic_lmda = False
n_samples_iter=100
max_iter=10
thres=1e-9
sample_selection=False


# define optimizer and loss 
opt = tf.keras.optimizers.Adam(learning_rate=lr)
loss = define_loss(d, loss_name = 'UncertaintyError')

# define new model
seq_copy = FeedForwardModel(input_dim=d, hidden_layers=layers, output_dim=n_classes, activation='relu')
seq_copy.build(input_shape=(layers[0],d))
seq_copy.compile(loss=loss, optimizer=opt)

# define the memory (lambda) parameter
lmda_ = LambdaParameter(lmda=lmda, automatic_lmda=automatic_lmda)

#define the sampling process
sampler = sp.Sampler(d=d, n_classes=n_classes)

# setting initial parameters

max_iter = 8 # 30 in the original figure
n_runs = 80 # 30 in the original figure

results = Parallel(n_jobs=-1, backend='threading')(delayed(tt.separate_runs)(original,
                                                                          max_iter=max_iter,
                                                                          X_test=X_test, 
                                                                          y_test=y_test)  for i in range(0, n_runs))

n, acc_train, acc_test, rho, lmda_vector = tt.decode_results(results)

aa = np.mean(acc_test, axis=0)
bb = np.std(acc_test, axis=0)

print(aa)
print(bb)