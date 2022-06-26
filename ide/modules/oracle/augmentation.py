from dataclasses import dataclass
from unittest import result
from ide.core.oracle.augmentation import Augmentation
import tensorflow as tf
import numpy as np
from scipy  import signal

@dataclass
class NoiseAugmentation(Augmentation):
    noise_ratio: float = 0.01

    rng = np.random.default_rng()

    def apply(self, data_points):

        queries, results = data_points
        
        augmented = self.rng.normal(results, self.noise_ratio)

        return queries, augmented

class NoiseConvolution(Augmentation):

    def apply(self, data_points):
        queries, results = data_points
        
        #convolution with random data
        t = np.linspace(-1, 1, len(results))
        bump = np.exp(-0.1*t**2)
        #bump /= np.trapz(bump) # normalize the integral to 1
        kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
        augmented = signal.fftconvolve(results, kernel, mode='same')

        return queries, augmented
