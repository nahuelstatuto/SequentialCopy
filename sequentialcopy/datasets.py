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

def covertype():
    """
     Returns the covertype dataset.
    """
    COVTYPE = datasets.fetch_covtype()
    X = COVTYPE.data
    y = COVTYPE.target

    return X, y

def wine():
    """
     Returns the IRIS dataset.
    """
    WINE = datasets.load_wine()
    X = WINE.data
    y = WINE.target
    return X,y

def iris():
    """
     Returns the IRIS dataset.
    """
    IRIS = datasets.load_iris()
    X = IRIS.data
    y = IRIS.target
    return X, y

def moons(n_samples):
    """
     Returns the make_moons dataset.
    """
    X, y =  datasets.make_moons(n_samples=n_samples, noise=0.15)
    return X,y

def yinyang(n_samples):
    """
     Returns the yin-yang dataset.
    """

    r_max = 1
    r = np.random.uniform(low=0, high=r_max**2, size=n_samples)
    theta = np.random.uniform(low=0, high=1, size=n_samples) * 2 * np.pi
    x = np.sqrt(r) * np.cos(theta)
    y = np.sqrt(r) * np.sin(theta)
    X = np.dstack([x, y])[0]
    y = np.empty((len(X),))

    # Upper circle
    center_x_u = 0
    center_y_u = 0.5
    radius_u = 0.5

    # Upper circle
    center_x_l = 0
    center_y_l = -0.5
    radius_l = 0.5

    i = 0
    for xi, yi in X:
        if ((xi > 0) & ((xi - center_x_u)**2 + (yi - center_y_u)**2 >= radius_u**2)) or ((xi < 0) & ((xi - center_x_l)**2 + (yi - center_y_l)**2 < radius_l**2)):
            y[i] = 1
        else:
            y[i] = 0

        if (xi - 0)**2 + (yi - 0.5)**2 < 0.15**2:
            y[i] = 1

        if (xi - 0)**2 + (yi - (-0.5))**2 < 0.15**2:
            y[i] = 0

        i += 1

    return X, y

def spirals(n_samples, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_samples,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_samples,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_samples,1) * noise
    return np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),np.hstack((np.zeros(n_samples),np.ones(n_samples)))

def dic_for_datasets():
    dic_datasets ={'abalone':'AdaBoost',
        'acute-nephritis':'AdaBoost',
        'acute-inflammation':'AdaBoost',
        'congressional-voting':'MLP',
        'credit-approval': 'MLP',
        'haberman-survival': 'RFC',
        'ionosphere': 'RFC',
        'magic': 'LinearSVM',
        'pima': 'LinearSVM',
        'synthetic-control': 'GaussianSVM',
        'ringnorm': 'GaussianSVM',
        'tic-tac-toe': 'XgBoost',
        'waveform': 'XgBoost',
        'breast-cancer': 'AdaBoost',
        'breast-cancer-wisc': 'AdaBoost',
        'breast-cancer-wisc-diag': 'AdaBoost',
        'breast-cancer-wisc-prog': 'AdaBoost',
        'bank': 'AdaBoost',
        'breast-tissue': 'AdaBoost',
        'chess-krvkp': 'MLP',
        'conn-bench-sonar-mines-rocks': 'MLP',
        'connect-4': 'MLP',
        'contrac': 'MLP',
        'cylinder-bands': 'MLP',
        'echocardiogram': 'MLP',
        'energy-y1': 'MLP',
        'energy-y2': 'MLP',
        'fertility': 'RFC',
        'heart-hungarian': 'RFC',
        'hepatitis': 'RFC',
        'ilpd-indian-liver': 'RFC',
        'iris': 'RFC',
        'mammographic': 'RFC',
        'miniboone': 'RFC',
        'molec-biol-splice': 'RFC',
        'mushroom': 'LinearSVM',
        'musk-1': 'LinearSVM',
        'musk-2': 'LinearSVM',
        'oocytes_merluccius_nucleus_4d': 'LinearSVM',
        'oocytes_trisopterus_nucleus_2f': 'LinearSVM',
        'parkinsons': 'LinearSVM',
        'pittsburg-bridges-MATERIAL': 'LinearSVM',
        'pittsburg-bridges-REL-L': 'LinearSVM',
        'pittsburg-bridges-T-OR-D': 'GaussianSVM',
        'planning': 'GaussianSVM',
        'seeds': 'GaussianSVM',
        'spambase': 'GaussianSVM',
        'statlog-australian-credit': 'GaussianSVM',
        'statlog-german-credit': 'GaussianSVM',
        'statlog-heart': 'GaussianSVM',
        'statlog-image': 'GaussianSVM',
        'statlog-vehicle': 'XgBoost',
        'teaching': 'XgBoost',
        'titanic': 'XgBoost',
        'twonorm': 'XgBoost',
        'vertebral-column-2clases': 'XgBoost',
        'vertebral-column-3clases': 'XgBoost',
        'waveform-noise': 'XgBoost',
        'wine': 'XgBoost',
        'spirals':'GaussianSVM',
        'yinyang':'GaussianSVM',
        'moons':'GaussianSVM'}
    return dic_datasets

def original_info():
    return {'abalone': [8, 3, 0.5454545454545454, 4177], 'acute-nephritis': [6, 2, 1.0, 120], 'acute-inflammation': [6, 2, 1.0, 120],'congressional-voting': [16, 2, 0.6091954022988506], 'credit-approval': [15, 2, 0.7898550724637681], 'haberman-survival': [3, 2, 0.6129032258064516], 'ionosphere': [33, 2, 0.9436619718309859], 'magic': [10, 2, 0.8009989484752892], 'pima': [8, 2, 0.7207792207792207], 'synthetic-control': [60, 6, 1.0], 'ringnorm': [18, 2, 0.9831081081081081], 'tic-tac-toe': [9, 2, 0.9739583333333334], 'waveform': [21, 3, 0.843], 'breast-cancer': [9, 2, 0.6896551724137931], 'breast-cancer-wisc': [9, 2, 0.9142857142857143], 'breast-cancer-wisc-diag': [30, 2, 0.9210526315789473], 'breast-cancer-wisc-prog': [33, 2, 0.7], 'bank': [16, 2, 0.8718232044198895], 'breast-tissue': [9, 6, 0.5454545454545454], 'chess-krvkp': [36, 2, 0.9953125], 'conn-bench-sonar-mines-rocks': [60, 2, 0.8333333333333334], 'connect-4': [42, 2, 0.8748519834221433], 'contrac': [9, 3, 0.5728813559322034], 'cylinder-bands': [35, 2, 0.7766990291262136], 'echocardiogram': [10, 2, 0.8148148148148148], 'energy-y1': [8, 3, 0.974025974025974], 'energy-y2': [8, 3, 0.922077922077922], 'fertility': [9, 2, 0.9], 'heart-hungarian': [12, 2, 0.7627118644067796], 'hepatitis': [19, 2, 0.7419354838709677], 'ilpd-indian-liver': [9, 2, 0.6153846153846154], 'iris': [4, 3, 0.9333333333333333], 'mammographic': [5, 2, 0.8031088082901554], 'miniboone': [50, 2, 0.9359550993733903], 'molec-biol-splice': [60, 3, 0.9435736677115988], 'mushroom': [21, 2, 0.9790769230769231], 'musk-1': [166, 2, 0.8125], 'musk-2': [166, 2, 0.9583333333333334], 'oocytes_merluccius_nucleus_4d': [41, 2, 0.7707317073170732], 'oocytes_trisopterus_nucleus_2f': [25, 2, 0.8032786885245902], 'parkinsons': [22, 2, 0.9230769230769231], 'pittsburg-bridges-MATERIAL': [7, 3, 0.9090909090909091], 'pittsburg-bridges-REL-L': [7, 3, 0.6666666666666666], 'pittsburg-bridges-T-OR-D': [7, 2, 0.8571428571428571], 'planning': [12, 2, 0.7027027027027027], 'seeds': [7, 3, 0.8809523809523809], 'spambase': [57, 2, 0.9261672095548317], 'statlog-australian-credit': [14, 2, 0.6811594202898551], 'statlog-german-credit': [24, 2, 0.765], 'statlog-heart': [13, 2, 0.8518518518518519], 'statlog-image': [18, 7, 0.9523809523809523], 'statlog-vehicle': [18, 4, 0.7647058823529411], 'teaching': [5, 3, 0.5483870967741935], 'titanic': [3, 2, 0.7777777777777778], 'twonorm': [20, 2, 0.9763513513513513], 'vertebral-column-2clases': [6, 2, 0.8387096774193549], 'vertebral-column-3clases': [6, 3, 0.8064516129032258], 'waveform-noise': [40, 3, 0.843], 'wine': [11, 3, 0.9444444444444444], 'spirals': [2, 2, 1.0], 'yinyang': [2, 2, 0.995], 'moons': [2, 2, 0.989]}