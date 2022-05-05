from __future__ import annotations
from random import random
from tkinter import W
from typing import TYPE_CHECKING

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
        results = queries*np.ones(self.result_shape)*self.a + np.ones(self.result_shape)*self.b
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
        results = (queries - self.x0)**2*np.ones(self.result_shape)*self.s + np.ones(self.result_shape)*self.y0
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
    radius = 1


    def query(self, queries):
        results = np.sqrt(-1*queries**2*np.ones(self.result_shape) + self.radius**2*np.ones(self.result_shape))
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
class CrossDataSource(DataSource):
    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)

    def query(self, queries):
        results = queries*np.ones(self.result_shape) if (np.random.randint(2) == 0) else np.ones(self.result_shape)-queries*np.ones(self.result_shape)
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
class HourglasDataSource(DataSource):
    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)

    def query(self, queries):
        r = np.random.randint(4)
        o  = np.ones(self.result_shape)
        if r == 0:
            results = np.zeros(self.result_shape)
        elif r == 1:
            results = queries*o
        elif r == 2:
            results = o - queries*o
        else:
            results = o
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
class ZDataSource(DataSource):
    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)

    def query(self, queries):
        r = np.random.randint(3)
        if r == 0:
            results = queries*np.ones(self.result_shape)
        elif r == 1:
            results = np.ones(queries.shape)
        else:
            results = queries*np.zeros(self.result_shape)
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
class CHF_data_source(DataSource):
    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)

    def query(self, queries):
        record = wfdb.rdrecord( wfdb.get_record_list(db_dir='chfdb')[0],pn_dir='chfdb')
        r = record.p_signal
        queries = r[:,0]
        results = r[:,1]
        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = -6
        x_max = 6
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)

    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

@dataclass
class HIPEDataSource(DataSource):
    df: pd.DataFrame = pd.DataFrame()
    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)

    def __init__(self):
        filename1 = 'real_world_data_sets/HIPE/PickAndPlaceUnit_PhaseCount_2_geq_2017-10-01_lt_2018-01-01.csv'
        filename2 = 'real_world_data_sets/HIPE/ScreenPrinter_PhaseCount_2_geq_2017-10-01_lt_2018-01-01.csv'
        filename3 = 'real_world_data_sets/HIPE/VacuumPump2_PhaseCount_2_geq_2017-10-01_lt_2018-01-01.csv'
        df = pd.read_csv(filename1)
        df2 = pd.read_csv(filename2)
        df3 = pd.read_csv(filename3)
        df.merge(df2,on = df.columns.values.tolist(), how = 'right')
        df.merge(df3,on = df.columns.values.tolist(), how = 'right')
        self.df = df

    def query(self, queries):
        results = self.df.to_numpy()
        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = 0
        x_max = 1450317
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)

    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)
@dataclass
class NASDAQDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)

    def __init__(self):
        pass

    def query(self, queries):
        filename = '/real_world_data_sets/NASDAQ/WIKI_PRICES.csv'
        df = pd.read_csv(filename)
        results = df.to_numpy()
        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = 0
        x_max = 10000
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)

    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

@dataclass
class OfficeDataSource(DataSource):
    df:pd.DataFrame = pd.DataFrame()

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)

    def __init__(self):
        connectivity_file = '/real_world_data_sets/Office/connectivity.txt'
        data_file = '/real_world_data_sets/Office/data.txt'
        locs_file = '/real_world_data_sets/Office/mote_locs.txt'

        connectivity_df = pd.read_csv(connectivity_file, sep=" ", header=None)
        data_df = pd.read_csv(data_file, sep=" ", header=None)
        locs_df = pd.read_csv(locs_file, sep=" ", header=None)
        self.df = data_df

    def query(self, queries):
        results = self.df.to_numpy()
        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = 0
        x_max = self.df.size
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)

    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

@dataclass
class PersonalActivityDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)


    def query(self, queries):
        pass

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
class SmartphoneDataSource(DataSource):
    df: pd.DataFrame = pd.DataFrame()
    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)

    def __init__(self):
        df1 = pd.read_csv('/real_world_data_source/Smartphone/final_acc_test.tx', sep=" ", header=None)
        df2 = pd.read_csv('/real_world_data_source/Smartphone/final_gyro_test.txt', sep=" ", header=None)
        df3 = pd.read_csv('/real_world_data_source/Smartphone/final_X_test.txt', sep=" ", header=None)
        df4 = pd.read_csv('/real_world_data_source/Smartphone/final_Y _test.txt', sep=" ", header=None)

    def query(self, queries):
        results = self.df.to_numpy()
        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = 0
        x_max = self.df.size
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)

    @property
    def data_pool(self) -> DataPool:
        return DataPool(self.query_pool, self.result_shape)

@dataclass
class SunspotDataSource(DataSource):

    df: pd.DataFrame = pd.DataFrame()
    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)

    def __init__(self):
        filename = '/real_world_data_sets/Sunspot/SN_d_tot_V2.0.txt'
        self.df = pd.read_csv(filename, sep=" ", header=None)

    def query(self, queries):
        results = self.df.to_numpy()
        return queries, results

    @property
    def query_pool(self) -> QueryPool:
        x_min = 0
        x_max = self.df.size
        query_ranges = np.asarray(tuple((x_min, x_max) for i in range(self.query_shape[0])))
        return QueryPool(query_count=None, query_shape=self.query_shape, query_ranges=query_ranges)

@dataclass
class HydraulicDataSource(DataSource):

    query_shape: Tuple[int,...] = (1,)
    result_shape: Tuple[int,...] = (1,)
    a: float = 1
    b: float = 0


    def query(self, queries):
        results = queries*np.ones(self.result_shape)*self.a + np.ones(self.result_shape)*self.b
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
       