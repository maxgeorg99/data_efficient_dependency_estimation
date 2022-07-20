from dataclasses import dataclass, field
import dataclasses
from typing import Dict, List
from typing_extensions import Self
from importlib_metadata import distribution
from nptyping import Float
import numpy as np
from ide.building_blocks.dependency_measure import DependencyMeasure
from ide.building_blocks.dependency_test import DependencyTest
from sklearn.utils import resample, shuffle

from ide.modules.query.query_sampler import RandomChoiceQuerySampler, UniformQuerySampler

@dataclass
class DependencyTestAdapter(DependencyTest):
    dependency_measure: DependencyMeasure 
    distribution:Dict = field(init=False, default_factory=dict,repr=False)
    scores = []
    iterations:int = field(default=0,repr=False)
    
    def __init__(self, dependency_measure: DependencyMeasure, iterations:int = 100):
        self.dependency_measure = dependency_measure
        self.iterations = iterations

    def calc_p_value(self,t):
        greater = sum([v for k,v in self.distribution.items() if k >= t])
        return greater / len(self.scores)

    def calc_var(self):
        estVar = self.dependency_measure.variance()
        if (estVar == 0):
            variance = np.var(self.scores)
        else:
            variance = estVar
        return variance

    def test(self):
        self.scores = []

        queries = self.exp_modules.queried_data_pool.queries

        samples = self.exp_modules.queried_data_pool.results

        self.scores.append(self.dependency_measure.apply(queries, samples))
        
        for i in range(1,self.iterations):
            shuffled_samples = self.shuffle_samples(samples)
            self.scores.append(self.dependency_measure.apply(queries, shuffled_samples))

        self.distribution = {item:self.scores.count(item) for item in self.scores}

        t, p, v = self.executeTest(queries, samples)

        return t, p, v

    def shuffle_samples(self, samples):
        shuffled_samples = np.copy(samples)
        for i in range(shuffled_samples.shape[1]):
            shuffled_samples[:,i] = shuffle(shuffled_samples[:,i]) #DevSkim: ignore DS148264 
        return shuffled_samples


    def executeTest(self, queries, samples):
        t = self.dependency_measure.apply(queries, samples)
        p = self.calc_p_value(t)
        v = self.calc_var()
        return t,p,v 