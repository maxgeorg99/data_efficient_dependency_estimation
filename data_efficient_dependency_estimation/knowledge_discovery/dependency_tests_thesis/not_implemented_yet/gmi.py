import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

import tensorflow as tf

from active_learning_ts.knowledge_discovery.knowledge_discovery_task import KnowledgeDiscoveryTask
from active_learning_ts.query_selection.query_sampler import QuerySampler
from active_learning_ts.queryable import Queryable


class GMI(KnowledgeDiscoveryTask):

    def __init__(self) -> None:
        super().__init__()
        self.global_uncertainty = 0

    def calcAlpha():
        aL = 0
        aU = 0
        #f(α) := d dα (∆(α, e∗XY) + cd(1 − α) n−1)
        f_a = 0
        x = 0
        if f_a(aU) < 0: 
            return aU
        if f_a(aL) > 0: 
            return aL
        if aL <= x <= aU: 
            return x        
        return 0

    #constructs a adjacency matrix with data points as nodes and pairs are conneced via edges
    def constructGraph(z:tf.Tensor):
        n = z.shape
        a = np.zeros(shape=(2 * n, 2 * n))
        for i in z:
            a = 0    
        return a

    def gmi(self, x: tf.Tensor ,y: tf.Tensor ):
        a = self.calcAlpha() 
        n = x.shape
        n1 = a * n
        n2 = (1 - a) * n
        #Divide Zn into two subsets Z ′ n′ and Z ′′ n′′ such that α = n′/n and β = n′′/n, where α + β = 1. 
        z = zip(x,y)
        z1 = z[0:n1-1]
        z2 = z[n1:]
        #Zn′′ ← {(xik, yjk)n′′ k=1: shuffle first and second elements of pairs in Z ′′ n′′ } 
        z2_shuffled = [(sub[1], sub[0]) for sub in z2]
        #Z ← Z′ n′ ∪  ̃ Z ′′ n′′ 
        z_for_MST = z1 + z2_shuffled
        #Construct MST on ̂ Z
        graph = self.constructGraph(z_for_MST)
        tree = minimum_spanning_tree(graph)
        #Rn′,n′′ ← # edges connecting a node in Z ′ n′ to a node of Zn′′
        r = tree[n1:,0:n1-1].sum()
        return 1 - r * (n1 + n2)/2*n1*n2 

    def learn(self, num_queries):
        self.sampler.update_pool(self.surrogate_model.get_query_pool())
        query = self.sampler.sample(num_queries)
        xs, ys = self.surrogate_model.query(query)

        r, p = self.gmi(xs,ys)
        self.global_uncertainty = p

        return r, p


    def uncertainty(self, points: tf.Tensor) -> tf.Tensor:
        return tf.fill(points.shape, self.global_uncertainty)
