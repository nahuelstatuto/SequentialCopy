import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler


def spirals(n_samples, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_samples,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_samples,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_samples,1) * noise
    
    return StandardScaler(copy=True).fit_transform( np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))) ),np.hstack((np.zeros(n_samples),np.ones(n_samples)))

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
    X, y =  datasets.make_moons(n_samples=n_samples, noise=0.2, shuffle=True)
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