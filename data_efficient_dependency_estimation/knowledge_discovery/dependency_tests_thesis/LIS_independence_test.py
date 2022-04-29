import numpy
import tensorflow as tf
import rpy2
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector

from data_efficient_dependency_estimation.knowledge_discovery.dependency_tests_thesis.RKnowledgeDiscoveryTask import RKnowledgeDiscoveryTask
from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.queryable import Queryable

class LISIndependenceTest(RKnowledgeDiscoveryTask):

    def __init__(self) -> None:
        self.packageName = ('LIStest')
        super().__init__()
        self.test = self.package.JLMn

    def learn(self, num_queries):
        self.sampler.update_pool(self.surrogate_model.get_query_pool())
        query = self.sampler.sample(num_queries)
        xs, ys = self.surrogate_model.query(query)
        xs = xs[:,0]
        ys = ys[:,0]

        r, p = self.test(xs, ys)
        self.global_uncertainty = p

        return r, p


    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        return tf.fill(points.shape, self.global_uncertainty)
