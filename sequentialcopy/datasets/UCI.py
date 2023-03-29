import os
import joblib
import pandas as pd
import numpy as np
from sklearn.utils import check_array, check_consistent_length
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

#UCI_names = joblib.load('UCI_names.pkl')

def create_dataset(dataset, 
                   path='../data', 
                   test_size=0.2, 
                   random_state=42):
    
    if not os.path.exists(os.path.join(path, dataset)):
        os.mkdir(os.path.join(path, dataset))
    
    if not os.path.exists(os.path.join(path, dataset, '{}_data.pkl'.format(dataset))):
    
        if dataset == 'spirals':
            X, y = spirals(5000, noise=1) 
        elif dataset == 'yinyang':
            X, y = yinyang(10000)
        elif dataset == 'moons':
            X, y = moons(10000)
        elif dataset == 'iris':
            X, y = iris()
        elif dataset == 'wine':
            X, y = wine()
        elif dataset == 'covertype':
            X, y = covertype()
        elif dataset in UCI_names:
            X, y = uci()
        else:
            raise NameError("The value {} is not allowed for variable dataset. Please choose spirals, yinyang, moons, iris, wine, covertype or UCI".format(dataset))

        #Split dataset into subsets that minimize the potential for bias in your evaluation and validation process.
        X_train, X_test, y_train, y_test = train_test_split(X, 
                                                            y.astype(int), 
                                                            test_size=test_size, 
                                                            random_state=random_state,
                                                            stratify=y)

        scaler = StandardScaler(copy=True)
        scaler.fit(X_train)
        X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
        y_train = y_train.ravel()
        y_test = y_test.ravel()
        data = {'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}
        
        joblib.dump(data, os.path.join(path, dataset, '{}_data.pkl'.format(dataset)))
    
    else:
        data = joblib.load(os.path.join(path, dataset, '{}_data.pkl'.format(dataset)))
    
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']

def uci():
    
    # Read raw data
    data = pd.read_table(os.path.join(path, dataset, '{}_R.dat'.format(dataset)), index_col=0)
    dtype = dtypes[dataset]
    
    # Convert to matrix format
    X = data.drop('clase', axis=1).to_numpy()
    y = data['clase'].to_numpy()
    
    # Re-order columns
    idx = dtype.argsort()
    dtype = dtype[idx[::-1]]
    X = X[:, idx[::-1]]
        
    # Preprocessing.
    X = check_array(X, accept_sparse=True, ensure_min_samples=1, dtype=np.float64)
    y = check_array(y, ensure_2d=False, ensure_min_samples=1, dtype=None)
    dtype = check_array(dtype, ensure_2d=False, ensure_min_samples=1, dtype=None)
    check_consistent_length(X, y)
    
    return X, y