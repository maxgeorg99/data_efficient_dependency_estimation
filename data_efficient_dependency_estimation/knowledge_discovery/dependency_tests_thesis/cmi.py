import subprocess
import numpy
import pandas as pd
import tensorflow as tf

from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.queryable import Queryable


class CMI(KnowledgeDiscoveryTask):

    def __init__(self) -> None:
        super().__init__()
        self.global_uncertainty = 0

    def learn(self, num_queries):
        self.sampler.update_pool(self.surrogate_model.get_query_pool())
        query = self.sampler.sample(num_queries)
        xs, ys = self.surrogate_model.query(query)
        data = xs[:,0],ys[:,0]
        #a comma-separated text file with 1 line header
        dataFile = 'cmiData.csv'
        df = pd.DataFrame(data)
        df.to_csv(dataFile, sep=",", header='true')

        p = subprocess.check_output(['java', '-jar', 'MCDE-experiments-1.0.jar', '-t EstimateDependency', '-f ' + dataFile ,'-a CMI'])        
        self.global_uncertainty = p

        return p


    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        return tf.fill(points.shape, self.global_uncertainty)
