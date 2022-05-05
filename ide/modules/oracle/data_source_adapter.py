from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass

import tensorflow as tf
import numpy as np
from distribution_data_generation.data_source import DataSource as oldDataSource
from active_learning_ts.data_retrievement.pools.continuous_vector_pool import ContinuousVectorPool

from ide.core.data.data_pool import DataPool

from ide.core.oracle.data_source import DataSource
from ide.core.query.query_pool import QueryPool

if TYPE_CHECKING:
    from typing import Tuple, List, Any
    from ide.core.oracle.interpolation_strategy import InterpolationStrategy
    from ide.core.data_sampler import DataSampler

class DataSourceAdapter(DataSource):
    distribution_data_source: oldDataSource
    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)

    def __init__(self, distribution_data_source: oldDataSource):
        self.distribution_data_source  = distribution_data_source

    def query(self, queries):
        q, r = self.distribution_data_source.query(queries)
        results = np.asarray(r)
        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)

    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)