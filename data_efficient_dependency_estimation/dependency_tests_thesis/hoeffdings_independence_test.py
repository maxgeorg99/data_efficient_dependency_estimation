import numpy
import tensorflow as tf
from  XtendedCorrel import hoeffding

from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.queryable import Queryable

from ide.building_blocks.dependency_test import DependencyTest

class ConditionalIndependenceTest(DependencyTest):

    def __init__(self) -> None:
        super().__init__()
        self.global_uncertainty = 0

    def learn(self, num_queries):
        self.sampler.update_pool(self.surrogate_model.get_query_pool())
        query = self.sampler.sample(num_queries)
        xs, ys = self.surrogate_model.query(query)

        p = hoeffding(xs[:,0], ys[:,0])
        self.global_uncertainty = p

        return p


    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        return tf.fill(points.shape, self.global_uncertainty)
