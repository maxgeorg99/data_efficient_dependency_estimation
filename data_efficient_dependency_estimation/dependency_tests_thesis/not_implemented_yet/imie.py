from random import random
import numpy
from scipy import rand
import tensorflow as tf
from random import randrange

from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.queryable import Queryable


class IMIE(KnowledgeDiscoveryTask):
    mean: int
    var: int
    k: int
    iterations: int
    offset: int

    def digamma (self, x: int):
        value = -0.577215664901532860606512090082
        for i in range (1,x):
            value += 1.0 / i
        return value

    def __init__(self) -> None:
        super().__init__()
        self.global_uncertainty = 0
        mean = 0
        var = 0
        k = 0 #ksg
        iterations = 0

    def MC_SortedIndexVectors():
        return 0

    def imie(self,x: tf.Tensor,y: tf.Tensor, OrderR):
        self.offset = self.digamma(x.shape) + self.digamma(self.k) - (1.0 / self.k)
        if (self.iterations < x.shape):
            randomIdx = randrange(self.iterations, x.shape - 1)
            OrderR[randomIdx], OrderR[self.iterations] = OrderR[self.iterations], OrderR[randomIdx]
            index = OrderR[self.iterations]
            self.iterations = self.iterations + 1
            MCs = self.MC_SortedIndexVectors(x[index], y[index], self.k, self.OrderX, self.OrderY, x, y)
            v = self.digamma(MCs.first) + self.digamma(MCs.second)
            delta = v - mean
            mean += (delta / (self.iterations))
            delta2 = v - mean
            var += (delta * delta2)
	
        return self.offset - self.mean

    def learn(self, num_queries):
        self.sampler.update_pool(self.surrogate_model.get_query_pool())
        query = self.sampler.sample(num_queries)
        xs, ys = self.surrogate_model.query(query)

        r = self.imie(xs,ys)
        self.global_uncertainty = self.var / self.iterations

        return r


    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        return tf.fill(points.shape, self.global_uncertainty)
