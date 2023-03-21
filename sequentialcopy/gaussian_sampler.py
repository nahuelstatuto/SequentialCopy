import numpy as np
from sequentialcopy.base_sampler import BaseSampler

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