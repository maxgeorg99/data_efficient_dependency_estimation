import numpy
import tensorflow as tf
import dcor

from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.queryable import Queryable

from data_efficient_dependency_estimation.knowledge_discovery.dependency_tests_thesis.independence_test_for_measure import IndependenceTestForMeasure


class DistanceCorrelationTest(IndependenceTestForMeasure):

    def __init__(self) -> None:
        super().__init__()
        self.global_uncertainty = 0

    def learn(self, num_queries):
        self.sampler.update_pool(self.surrogate_model.get_query_pool())
        query = self.sampler.sample(num_queries)
        xs, ys = self.surrogate_model.query(query)

        r = dcor.distance_correlation(xs[:,0], ys[:,0])
        self.global_uncertainty = None

        return r


    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        return tf.fill(points.shape, self.global_uncertainty)
