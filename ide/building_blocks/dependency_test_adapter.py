from importlib_metadata import distribution
import numpy as np
from ide.building_blocks.dependency_measure import DependencyMeasure
from ide.building_blocks.dependency_test import DependencyTest
from ide.core.oracle.data_source import DataSource
from ide.modules.oracle.data_source import IndependentDataSetDataSource
from sklearn.utils import resample, shuffle


class DependencyTestAdapter(DependencyTest):
    distribution: dict
    scores = []
    total = 0
    dependency_measure: DependencyMeasure

    def __init__(self, dependency_measure: DependencyMeasure, datasource: DataSource,iterations:int = 100):
        if (datasource.data_pool.query_count is None):
            queries = datasource.query_pool.queries_from_norm_pos(np.random.uniform(size=(100, *datasource.query_pool.query_shape)))
            samples = datasource.query(queries)
        else:
            samples = datasource.data_pool.all_queries()
        self.dependency_measure = dependency_measure
        self.scores.append(dependency_measure.apply(samples))
        
        for i in range(1,iterations):
            samples = tuple(shuffle(x) for x in samples) #DevSkim: ignore DS148264 
            self.scores.append(dependency_measure.apply(samples))

        self.distribution = {item:self.scores.count(item) for item in self.scores}
        self.total = len(self.scores)

    def calc_p_value(self,t):
        greater = sum([v for k,v in self.distribution.items() if k >= t])
        return greater / self.total

    def calc_var(self):
        variance = np.var(self.scores)
        return variance

    def test(self, samples):
        t = self.dependency_measure.apply(samples,samples)
        p = self.calc_p_value(t)
        v = self.calc_var()
        return t,p,v 
    