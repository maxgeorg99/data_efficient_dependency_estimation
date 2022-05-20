from __future__ import annotations
import dataclasses
from random import random
from tkinter import W
from typing import TYPE_CHECKING, Dict

from dataclasses import dataclass, field
from unittest import result
from nptyping import NDArray

import pandas as pd
import wfdb
import numpy as np
from ide.core.data.data_pool import DataPool

from ide.core.oracle.data_source import DataSource
from ide.core.query.query_pool import QueryPool

if TYPE_CHECKING:
    from typing import Tuple, List, Any
    from ide.core.oracle.interpolation_strategy import InterpolationStrategy
    from ide.core.data_sampler import DataSampler



@dataclass
class LineDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    a: float = 1
    b: float = 0


    def query(self, queries):
        results = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) + np.ones(self.result_shape)*self.b
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


@dataclass
class SquareDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    x0: float = 0.5
    y0: float = 0
    s: float = 5


    def query(self, queries):
        results = np.dot((queries - self.x0)**2, np.ones((*self.query_shape,*self.result_shape))*self.s) + np.ones(self.result_shape)*self.y0
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

    
@dataclass
class HyperSphereDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)

    def query(self, queries):
        x = np.dot(-1*np.square(queries), np.ones((*self.query_shape,*self.result_shape)))
        y = x + np.ones(self.result_shape)
        results = np.sqrt(np.abs(y))
        if (random() <= 0.5):
            results = np.negative(results)
        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = -1
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)
    
    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)
@dataclass
class InterpolatingDataSource(DataSource):
    data_sampler: DataSampler
    interpolation_strategy: InterpolationStrategy

    def query(self, queries):
        data_points = self.data_sampler.sample(queries)
        data_points = self.interpolation_strategy.interpolate(data_points)
        return data_points

    @property
    def query_pool(self) -> QueryPool:
        return self.interpolation_strategy.query_pool

    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

@dataclass
class RealWorldDataSetDataSource(DataSource):
    data: Dict = dataclasses.field(default_factory=dict)
    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)

    def __init__ (self, keys, values):
        self.data = dict(zip(keys, values))

    def query(self, queries):
        results = np.asarray([self.data.get(x) for sublist in queries for x in sublist]).reshape(-1, 1)
        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        keys = list(self.data.keys())
        qp = QueryPool(query_count=0, query_shape=self.query_shape, query_ranges=None)
        for i in keys:
            qp.add_queries(np.asarray([i]).reshape(-1, 1))
        return qp

    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)
       