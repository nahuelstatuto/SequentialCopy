import numpy as np
from sequentialcopy.base_sampler import BaseSampler

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
