from numpy import var
import tensorflow as tf
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector

from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.queryable import Queryable

class RKnowledgeDiscoveryTask(KnowledgeDiscoveryTask):

    packageName: var
    package: var
    test: var

    def __init__(self) -> None:
        super().__init__()
        self.global_uncertainty = 0
        self.package = rpackages.importr(self.packageName)

    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        return tf.fill(points.shape, self.global_uncertainty)