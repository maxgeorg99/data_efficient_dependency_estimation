from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from ide.building_blocks.query_sampler import QuerySampler
from ide.building_blocks.selection_criteria import SelectionCriteria
from ide.core.configuration import Configurable

if TYPE_CHECKING:
    from typing import Tuple, List
    
@dataclass
class QueryOptimizer(Configurable):
    selection_criteria: SelectionCriteria
    num_queries: int

    def select(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries
        ...
    
@dataclass
class MaxMCQueryOptimizer(QueryOptimizer):
    query_sampler: QuerySampler
    num_tries: int

    def select(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries

        query_candidates = self.query_sampler.sample(self.selection_criteria.pool, self.num_tries)
        scores = self.selection_criteria.query(query_candidates)

        queries = tf.gather(query_candidates, tf.math.top_k(scores, num_queries).indices)

        return queries.as_numpy()

@dataclass
class ProbWeightedMCQueryOptimizer(QueryOptimizer):
    query_sampler: QuerySampler
    num_tries: int

    _rng = np.random.default_rng()


    def select(self, num_queries = None):
        if num_queries is None: num_queries = self.num_queries
        query_candidates = self.query_sampler.sample(self.selection_criteria.pool, self.num_tries)
        scores = self.selection_criteria.query(query_candidates)

        queries = self._rng.choice(a=query_candidates, size=num_queries, replace=False, p=scores)

        return queries