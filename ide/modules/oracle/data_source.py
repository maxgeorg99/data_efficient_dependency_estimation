from __future__ import annotations
import dataclasses
from random import Random, random
from tkinter import W
from typing import TYPE_CHECKING, Dict

from dataclasses import dataclass, field
from typing_extensions import Self
from unittest import result
from importlib_metadata import distribution
from nptyping import NDArray, Number, Shape

import pandas as pd
from scipy import rand
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RationalQuadratic
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

from hyppo.tools import spiral
from hyppo.tools import diamond
from hyppo.tools import w_shaped
from hyppo.tools import logarithmic
from hyppo.tools import two_parabolas

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
        N = len(queries)
        dim = self.result_shape[0] + self.query_shape[0]
        norm = np.random.normal
        normal_deviates = norm(size=(dim, N))
        radius = np.sqrt((normal_deviates**2).sum(axis=0))
        results = np.transpose(normal_deviates/radius)
        return results[:,0:1], results[:,1:]

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
        self.result_shape = (dims,)

    def query(self, queries):
        distributions = []
        for i in range(self.number_of_distributions):
            loc = random()
            scale = random()
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
        results = [np.random.choice(data[x],size=self.dim) for x in random_idx]
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
        data_points = self.interpolation_strategy.interpolate(queries,data_points)
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

@dataclass
class DiamondDataSource(DataSource):
    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)

    def query(self, queries):
        period = -np.pi / 4
        u = queries
        v = queries
        p=queries.shape[1]
        sig = np.identity(p)
        gauss_noise = np.random.multivariate_normal(np.zeros(p), sig, size=len(queries))

        x = u * np.cos(period) + v * np.sin(period) + 0.05 * p * gauss_noise
        results = -u * np.sin(period) + v * np.cos(period)

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
class WshapeDataSource(DataSource):
    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    r: int = 1

    def query(self, queries):
        p=self.result_shape[0]
        n = len(queries)

        x = np.array(np.random.uniform(-self.r, self.r, size=(n, p)))
        #x = np.dot(queries, np.ones((*self.query_shape,*self.result_shape)))
        u = np.array(np.random.uniform(0, 1, size=(len(queries), p)))
        coeffs = np.array([1 / (i + 1) for i in range(p)]).reshape(-1, 1)
        x_coeffs = x @ coeffs
        u_coeffs = u @ coeffs
        y = 4 * ((x_coeffs**2 - 0.5) ** 2 + u_coeffs / 500)

        y = np.concatenate((x[:,1:], y),axis=-1)
        x = x[:,0:1]
        return x ,y

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
class SpiralDataSource(DataSource):
    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    r: int = 5

    def query(self, queries):
        p=self.result_shape[0]
        n = len(queries)

        rx = np.array(np.random.uniform(0, self.r, size=(n, 1)))
        ry = rx
        rx = np.repeat(rx, p, axis=1)
        z = rx
        x = np.zeros((n, p))
        x[:, 0] = np.cos(z[:, 0] * np.pi)
        for i in range(p - 1):
            x[:, i + 1] = np.multiply(x[:, i], np.cos(z[:, i + 1] * np.pi))
            x[:, i] = np.multiply(x[:, i], np.sin(z[:, i + 1] * np.pi))
        x = np.multiply(rx, x)
        y = np.multiply(ry, np.sin(z[:, 0].reshape(-1, 1) * np.pi))

        y = np.concatenate((x[:,1:], y),axis=-1)
        x = x[:,0:1]
        return x, y

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
class LogarithmicDataSource(DataSource):
    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    def query(self, queries):
        p=self.result_shape[0]

        x = np.dot(queries, np.ones((*self.query_shape,*self.result_shape)))

        y = np.log(x**2)

        x = x[:,0:1]
        return x, y

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
class TwoParabolasDataSource(DataSource):
    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    r: int = 1

    def query(self, queries):
        n = len(queries)
        p=self.result_shape[0]
        
        x = np.array(np.random.uniform(-self.r, self.r, size=(n, p)))
        #x = np.dot(queries, np.ones((*self.query_shape,*self.result_shape)))
        n = len(x)
        u = np.random.binomial(1, 0.5, size=(n, 1))
        coeffs = np.array([1 / (i + 1) for i in range(p)]).reshape(-1, 1)
        x_coeffs = x @ coeffs
        y = (x_coeffs**2) * (u - 0.5)

        y = np.concatenate((x[:,1:], y),axis=-1)
        x = x[:,0:1]
        return x, y

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
class CrossDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    a: float = 1

    def query(self, queries):

        direction = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_up = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) 
        results_up = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) 
        results_down = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*(-self.a))

        results = (1- direction)*results_up + direction*results_down

        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)
    
    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

@dataclass
class DoubleLinearDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    a: float = 1
    slope_factor = 0.5

    def query(self, queries):

        slope = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_steap = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) 
        results_flat = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.slope_factor*self.a)

        results = (1- slope)*results_steap + slope*results_flat

        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)
    
    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

        
@dataclass
class HourglassDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    a: float = 1

    def query(self, queries):

        kind = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_up = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) 
        results_down = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*(-self.a))

        results_dir = (1- kind)*results_up + kind*results_down
        results_const = (1- kind)*0.5 + kind*-0.5

        const = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results = (1- const)*results_dir + const*results_const

        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)
    
    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

@dataclass
class ZDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    a: float = 1

    def query(self, queries):

        kind = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_const = (1- kind)*0.5 + kind*-0.5
        results_up = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*self.a) 


        const = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results = (1- const)*results_up + const*results_const

        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)
    
    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

@dataclass
class ZInvDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    a: float = 1

    def query(self, queries):

        kind = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_const = (1- kind)*0.5 + kind*-0.5
        results_down = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))*(-self.a))

        const = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results = (1- const)*results_down + const*results_const

        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)
    
    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

@dataclass
class LinearPeriodicDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    a: float = 1
    p: float = 0.2


    def query(self, queries):
        results = np.dot(queries % self.p, np.ones((*self.query_shape, *self.result_shape))*self.a)
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
class SineDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    a: float = 1
    p: float = 1
    y0: float = 0
    x0: float = 0


    def query(self, queries):
        results = np.dot(np.sin((queries-self.x0) * 2 * np.pi * self.p), np.ones((*self.query_shape,*self.result_shape))*self.a) + np.ones(self.result_shape)*self.y0
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
class HypercubeDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    w: float = 0.5

    def query(self, queries):
        dim = self.result_shape[0]
        results = []
        for i in range(len(queries)):
            results_for_dim = []
            y = Random().randint(0,dim)
            for j in range(dim):
                if j == y:
                    results_for_dim.append(np.asarray([Random().randint(0,1)]))
                else:
                    results_for_dim.append(queries[i])
            results.append(results_for_dim)
        results = np.asarray(results)
        results = np.squeeze(results,axis=2)
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
class HypercubeGraphDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    w: float = 0.5

    def query(self, queries):
        dim = self.result_shape[0]
        results = []
        for i in range(len(queries)):
            results_for_dim = []
            y = Random().randint(0,dim)
            for j in range(dim):
                if j == y:
                    results_for_dim.append(queries[i])
                else:
                    results_for_dim.append(np.asarray([Random().randint(0,1)]))
            results.append(results_for_dim)
        results = np.asarray(results)
        results = np.squeeze(results,axis=2)
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
class StarDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    w: float = 0.1

    def query(self, queries):

        direction = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        results_up = np.dot(queries, np.ones((*self.query_shape,*self.result_shape))) 
        results_down = np.dot(queries, -np.ones((*self.query_shape,*self.result_shape)))

        results_dir = (1- direction)*results_up + direction*results_down

        const = np.random.randint(2,size=(queries.shape[0], *self.result_shape))

        result_const = const*results_dir

        random = np.random.uniform(-0.5,0.5, size=(queries.shape[0], *self.result_shape))

        mask = np.equal(queries, 0)

        results = mask * random + (1 - mask) * result_const

        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = -0.5
        x_max = 0.5
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)
    
    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)


from threading import Lock
@dataclass
class GausianProcessDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    s: float = 0.1
    r: int = np.random.randint(1, 10000000)

    gpr: GaussianProcessRegressor = field(default=None,init=False)
    queries: NDArray[Number, Shape["query_nr, ... query_dim"]] = field(default=None,init=False)
    results: NDArray[Number, Shape["query_nr, ... result_dim"]] = field(default=None,init=False)
    singleton: GaussianProcessRegressor = field(default=None,init=False)

    def query(self, queries):
        if self.results.size == 0:
            results = self.gpr.sample_y(queries,*self.result_shape)
        else:
            results = np.squeeze(self.gpr.sample_y(queries))
        self.queries = np.concatenate((self.queries, queries))
        self.results = np.concatenate((self.results, results))
        self.gpr.fit(self.queries, self.results)

        return self.queries, self.results

    @property
    def query_pool(self) -> QueryPool:
        x_min = 0
        x_max = 1
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)

    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

    def __call__(self, **kwargs) -> Self:
        obj = super().__call__( **kwargs)
        if self.singleton is None:
            obj.gpr = GaussianProcessRegressor(kernel=RationalQuadratic(length_scale=obj.s), optimizer=None, random_state=obj.r)
            obj.queries = np.empty((0,*obj.query_shape))
            obj.results = np.empty((0,*obj.result_shape))
            obj.singleton = obj
            self.singleton = obj
        return self.singleton

