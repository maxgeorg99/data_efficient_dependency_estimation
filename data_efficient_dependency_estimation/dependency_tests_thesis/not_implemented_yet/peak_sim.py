import numpy
import tensorflow as tf

from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.queryable import Queryable


class PeakSimilarity (KnowledgeDiscoveryTask):

    def __init__(self) -> None:
        super().__init__()
        self.global_uncertainty = 0

    def peakSim(x,y,n:int):
        X = numpy.fft.fft(x,n)
        Y = numpy.fft.fft(y,n)
        #select first n high-amplitude DFT coefficients, where n << N
        #normalized to mean zero and variance one
        return 0

    def learn(self, num_queries):
        self.sampler.update_pool(self.surrogate_model.get_query_pool())
        query = self.sampler.sample(num_queries)
        xs, ys = self.surrogate_model.query(query)
        n = xs.shape / 1000

        r, p = self.peakSim(xs,ys,n)

        self.global_uncertainty = p

        return r, p

    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        return tf.fill(points.shape, self.global_uncertainty)
