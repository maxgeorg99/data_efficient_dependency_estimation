from __future__ import annotations
import dataclasses
from random import random
from tkinter import W
from typing import TYPE_CHECKING, Dict

from dataclasses import dataclass, field
from typing_extensions import Self
from unittest import result
from importlib_metadata import distribution
from nptyping import NDArray

import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import wfdb
import numpy as np
from ide.core.data.data_pool import DataPool
from ide.core.data.queried_data_pool import QueriedDataPool
from ide.core.experiment_modules import ExperimentModules

from ide.core.oracle.data_source import DataSource
from ide.core.query.query_pool import QueryPool
from ide.modules.data_sampler import KDTreeKNNDataSampler, KDTreeRegionDataSampler, RTreeKNNDataSampler
from ide.modules.oracle.interpolation_strategy.interpolation_strategy import AverageInterpolationStrategy
from ide.modules.queried_data_pool import FlatQueriedDataPool

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
class IndependentDataSetDataSource(DataSource):
    id: int = field(default = 1, repr=False)
    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    width: int = field(default = 1, repr=False)
    number_of_distributions: int = field(default = 10, repr=False)
    distribution_function = None
    dim: int = field(default = 1, repr=False)
    all_distributions = [np.random.normal,np.random.uniform,np.random.gamma,np.random.beta]

    def __init__(self,number_of_distributions = 10, dims = 1, id = 1):
        self.id = id
        self.dim = dims
        self.number_of_distributions = number_of_distributions
        self.distribution_function = GaussianMixture(n_components=self.number_of_distributions)

    def query(self, queries):
        distributions = []
        for i in range(self.number_of_distributions):
            loc = np.random.randint(100,size=1)
            scale = np.random.randint(100,size=1)
            distribution = np.random.choice(self.all_distributions)
            if distribution is np.random.normal:
                distributions.append({"type": distribution, "kwargs": {"loc": loc, "scale": scale}})
            elif distribution is np.random.uniform:
                distributions.append({"type": distribution, "kwargs": {"low": loc, "high": scale}})
            elif distribution is np.random.gamma:
                distributions.append({"type": distribution, "kwargs": {"shape":self.dim, "scale": scale}})
            elif distribution is np.random.beta:
                distributions.append({"type": distribution, "kwargs": {"a": 1+ loc, "b": 1 + scale}})
        sample_size = len(queries)
        coefficients = np.array([random() for i in range(sample_size)])
        coefficients /= coefficients.sum()      # in case these did not add up to 1
        data = np.zeros((sample_size, self.number_of_distributions))
        for idx, distr in enumerate(distributions):
            data[:, idx] = np.asarray( distr["type"](size=(sample_size,), **distr["kwargs"]))
        random_idx = np.random.choice(np.arange(sample_size), size=(sample_size,), p=coefficients)
        results = [np.asarray(np.random.choice(data[x])) for x in random_idx]
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
class InterpolatingDataSource(DataSource):
    interpolation_strategy: InterpolationStrategy = dataclasses.field(default=AverageInterpolationStrategy(), repr=False)
    data: Dict = dataclasses.field(default_factory=dict, repr=False)
    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    data_set: str = dataclasses.field(default="", repr=False)

    def query(self, queries):

        keys = list(self.data.keys())
        values = list(self.data.values())
        all_queries = np.asarray(keys).reshape(-1, 1)
        all_results = np.asarray(values).reshape(-1, 1)

        knn = NearestNeighbors(n_neighbors=5)
        knn.fit(all_queries, all_results)
        kneighbor_indexes = knn.kneighbors(queries, n_neighbors=5, return_distance=False)
        
        neighbor_queries = all_queries[kneighbor_indexes]
        kneighbors = all_results[kneighbor_indexes]
        
        data_points = (np.asarray(neighbor_queries), kneighbors)
        data_points = self.interpolation_strategy.interpolate(data_points)
        return data_points

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