import subprocess
import numpy
import pandas as pd
import tensorflow as tf
from py4j.java_gateway import JavaGateway

from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.queryable import Queryable

from ide.building_blocks.dependency_test import DependencyTest

class HICS(DependencyTest):

    def __init__(self) -> None:
        super().__init__()
        self.global_uncertainty = 0

    def learn(self, num_queries):
        self.sampler.update_pool(self.surrogate_model.get_query_pool())
        query = self.sampler.sample(num_queries)
        xs, ys = self.surrogate_model.query(query)
        data = xs[:,0],ys[:,0]
        #a comma-separated text file with 1 line header
        dataFile = 'hicsData.csv'
        df = pd.DataFrame(data)
        df.to_csv(dataFile, sep=",", header='true')

        p = subprocess.check_output(['java', '-jar', 'MCDE-experiments-1.0', '-t EstimateDependency', '-f ' + dataFile ,'-a HiCS'])
        self.global_uncertainty = p

        return p


    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        return tf.fill(points.shape, self.global_uncertainty)
