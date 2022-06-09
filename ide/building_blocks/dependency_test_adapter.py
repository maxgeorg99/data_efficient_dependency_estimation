from dataclasses import dataclass, field
import dataclasses
from typing import List
from importlib_metadata import distribution
import numpy as np
from ide.building_blocks.dependency_measure import DependencyMeasure
from ide.building_blocks.dependency_test import DependencyTest
from ide.core.data_sampler import DataSampler
from ide.core.oracle.data_source import DataSource
from ide.core.query.query_sampler import QuerySampler
from ide.modules.oracle.data_source import IndependentDataSetDataSource
from sklearn.utils import resample, shuffle

from ide.modules.query.query_sampler import RandomChoiceQuerySampler, UniformQuerySampler

@dataclass
class DependencyTestAdapter(DependencyTest):
    dependency_measure: DependencyMeasure
    distribution: field(init=False, default_factory=dict)
    scores = []
    def __init__(self, dependency_measure: DependencyMeasure, datasource: DataSource, samplesize = 100, iterations:int = 100):
        
        if (datasource.data_pool.query_count is None):
            queries = datasource.query_pool.queries_from_norm_pos(np.random.uniform(size=(samplesize, *datasource.query_pool.query_shape)))
            samples = datasource.query(queries)
        else:
            samples = datasource.data_pool.all_queries()
        self.dependency_measure = dependency_measure
        self.scores.append(dependency_measure.apply(samples))
        
        for i in range(1,iterations):
            samples = tuple(shuffle(x) for x in samples) #DevSkim: ignore DS148264 
            self.scores.append(dependency_measure.apply(samples))

        self.distribution = {item:self.scores.count(item) for item in self.scores}

    def calc_p_value(self,t):
        greater = sum([v for k,v in self.distribution.items() if k >= t])
        return greater / len(self.scores)

    def calc_var(self):
        variance = np.var(self.scores)
        return variance

    def test(self):
        queries = self.exp_modules.queried_data_pool.queries

        samples = self.exp_modules.queried_data_pool.results

        t, p, v = self.executeTest(samples)

        return t, p, v

    def executeTest(self, samples):
        t = self.dependency_measure.apply(samples)
        p = self.calc_p_value(t)
        v = self.calc_var()
        return t,p,v 
    