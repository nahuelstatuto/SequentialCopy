import os
import numpy as np
import tensorflow as tf
import logging

from numpy import genfromtxt
from numpy import savetxt

class BaseSampler():
    """Base Sampler class. Generates new samples and saves them to a file if desired."""
    
    # Set the initial values for class variables
    data = None
    iteration = 0
    file = None
       
    def __init__(self, d, n_classes, target=-1, from_file=False, 
                 to_file=False, file_path=None, automatic_fill=False):
        """Initializer for Sampler class. 
        
        Args:
        d (int): number of dimensions in the data
        n_classes (int): number of classes in the problem
        target (int): the index of the target column in the file with data labels (default: -1, last column)
        from_file (bool): whether to read samples from a file
        to_file (bool): whether to save generated samples to a file
        file_path (str): path to the file with samples to read from or save to
        automatic_fill (bool): whether to create a file and save the data in it or not
        """
        self.d = d
        self.n_classes = n_classes
        self.target = target
        
        self.from_file = from_file
        self.to_file = to_file
        self.file_path = file_path
        self.automatic_fill = automatic_fill
        
        self.set_file()
        
    def set_file(self):
        """Opens the file with samples for reading or creates a new file if it does not exist."""
        if self.from_file:
            if self.file_path:
                self.file = self.open_file_for_read()
            else:
                logging.error('No such file: \'{}\'. Define \'file_path\'.'.format(self.file_path))
                self.from_file, self.to_file, self.automatic_fill  = False, False, False
        
    def open_file_for_read(self):
        """Opens the file with samples for reading."""
        try:
            return open(os.path.join(self.file_path), 'rb')
        except:
            self.from_file = False
            if self.automatic_fill:
                fp = self.create_sampling_file()
                return fp
            else:
                logging.error('No such file: \'{}\'. Set \'automatic_fill\' = True to create it.'.format(self.file_path))
                self.to_file = False
                return None

    def create_sampling_file(self):
        """Creates a new file for saving generated samples."""
        try:
            fp = open(os.path.join(self.file_path), 'wb')
            self.to_file = True
            logging.warning('No such file: \'{}\'. File created!'.format(self.file_path))
            return fp
        except:
            logging.error('No file_path defined. File can not be created.')
            return None
            
    def close_file(self):
        """Close file."""
        if self.file:
            self.file.close()        
        
    def get_samples(self, original, num_samples = 100):
        """Get new samples from either file or generate them.

        Args:
        original: original model to label synthetic data
        num_samples (int): number of samples to generate

        Returns:
        X_new: generated data
        y_new: generated labels
        """
        if self.from_file and self.file:
            X, y = self.read_samples_from_file(original, num_samples)
        elif self.from_file and not(self.file):
            logging.error('Define \'file_path\'.'.format(self.file_path))
            self.from_file, self.to_file, self.automatic_fill = False, False, False
            return np.empty((0, self.d)), np.empty((0), dtype=int)
        else:
            X, y = self.generate_samples(original, num_samples)
            if self.to_file:
                self.add_samples_to_file(X, y)
            
        return X, y

    def read_samples_from_file(self, original, num_samples):
        """Read new samples from file. 

        Args:
        original: original model to label synthetic data
        num_samples (int): number of samples to generate

        Returns:
        X_new: generated data
        y_new: generated labels
        """
        #read the whole file the first time
        if self.iteration==0:
            self.data = genfromtxt(self.file, delimiter=',') 
        
        X_new = np.empty((0, self.d))
        y_new = np.empty((0), dtype=int)
        
        #read only "num_samples" lines
        for line in self.data[self.iteration*num_samples:(self.iteration+1)*num_samples]:
            X_new, y_new = np.vstack((X_new,np.asarray(line[:self.d]))), np.append(y_new,line[self.target])

        self.iteration+=1
        
        if len(X_new)==num_samples:
            return X_new, y_new 

        logging.warning('Not enough data points from file. {} data points were generated.'.format(num_samples-len(X_new)))
        self.from_file, self.to_file = False, True  
        
        X, y = self.generate_samples(original, num_samples-len(X_new))
        self.add_samples_to_file(X, y)      
        X_new, y_new = np.vstack((X_new,X)), np.append(y_new,y)
        
        return X_new, y_new

    def generate_samples(self, original, num_samples, **kwargs):
        """Generate new samples from selected distributions (default: Gaussian.)

        Args:
        original: original model to label synthetic data
        num_samples (int): number of samples to generate

        Returns:
        X_new: generated data
        y_new: generated labels
        """
        
        X_new = np.empty((0, self.d))
        y_new = np.empty((0), dtype=int)
        
        return X_new, y_new
    
    def add_samples_to_file(self, X, y):
        """Add data to a file."""
        self.file = open(os.path.join(self.file_path), 'a+')
        for X_,y_ in zip(X,y):            
            savetxt(self.file, np.asarray([ np.append(X_[:],y_)]), delimiter=',')
        self.close_file()


class GaussianSampler(BaseSampler):
    """Sampler class that generates new samples from a Gaussian Dsitribution and saves them to a file if desired."""
    
    def generate_samples(self, original, num_samples, **kwargs):
        """Generate new samples from a Gaussian distribution

        Args:
        original: original model to label synthetic data
        num_samples (int): number of samples to generate
        **kwargs (dict): standard deviation  

        Returns:
        X_new: generated data
        y_new: generated labels
        """
        std = 1.5
        if 'std' in kwargs:
            if isinstance(kwargs.get("std"), float):
                std = kwargs.get("std")
        
        X_new = np.random.multivariate_normal(np.zeros((self.d,)),
                                                  np.eye(self.d,self.d)*std,
                                                  size=num_samples)
        y_new = original.predict(X_new)
        return X_new, y_new
    

class BalancerSampler(BaseSampler):
    """Sampler class that generates new samples from a Gaussian Dsitribution and saves them to a file if desired."""
    
    def generate_samples(self, original, num_samples, **kwargs):
        """Generate new samples from Balancer distribution

        Args:
        original: original model to label synthetic data
        num_samples (int): number of samples to generate

        Returns:
        X_new: generated data
        y_new: generated labels
        """
        
        high = 2*np.sqrt(self.d)
        if 'high' in kwargs:
            if isinstance(kwargs.get("high"), float):
                high = kwargs.get("high")
        
        X_new, y_new = self.rBalancer(model=original, N_batch=num_samples, high=high)
        return X_new, y_new
    
    
    def rBalancer(self,model,N=0,max_iter=10,N_batch=100, low=0, high=1):
        """Funtion that generate samples balanced for each class in a d-dimensional ball. Useful for larger d. 
        
        Args:
        
        model: the original model used to generate the synthetic samples.
        N: the number of samples required per class.
        max_iter: the maximum number of iterations to generate the samples.
        N_batch: the number of elements sampled at each iteration.
        low and high: the minimum and maximum values used to scale the direction between them.
        
        """
        bins = np.arange(self.n_classes+1)-0.5
        classes = np.arange(self.n_classes)
        
        #Generate random direction
        v = np.random.multivariate_normal(np.zeros((self.d,)),np.eye(self.d,self.d),size = N_batch)
        v = v/np.linalg.norm(v,axis=1)[:,np.newaxis]
        
        #Scale the direction between low and high
        alpha = np.random.uniform(low=low,high=high,size = N_batch)

        qsynth = np.dot(alpha[:,np.newaxis],np.ones((1,self.d)))*v
        y_synth = model.predict(qsynth)
        nSamplesPerClass=np.histogram(y_synth,bins=bins)[0]
        
        toAdd = classes[nSamplesPerClass<N]
        toDelete = classes[nSamplesPerClass>N]
        
        ## hay que deletear!!!
        
        
        
        for i in range(max_iter):
            #Generate random direction
            v = np.random.multivariate_normal(np.zeros((self.d,)),np.eye(self.d,self.d),size = N_batch)
            v = v/np.linalg.norm(v,axis=1)[:,np.newaxis]
            
            alpha = np.random.uniform(low=low,high=high,size = N_batch)
            qtmp = np.dot(alpha[:,np.newaxis],np.ones((1,self.d)))*v
            y_synth = model.predict(qtmp)

            #Select samples to add
            idx = [j for j in range(N_batch) if y_synth[j] in toAdd]
            
            #Add samples to the synthetic set
            qsynth = np.r_[qsynth,qtmp[idx,:]]
            
            y_synth = model.predict(qsynth)

            nSamplesPerClass=np.histogram(y_synth,bins=bins)[0]
            toAdd = classes[nSamplesPerClass<N]
            
            if len(toAdd)<1:
                return qsynth,y_synth
        
        return qsynth,y_synth
