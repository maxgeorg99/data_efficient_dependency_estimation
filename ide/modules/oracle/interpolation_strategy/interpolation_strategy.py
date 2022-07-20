from __future__ import annotations
from random import randint
import random
from typing import TYPE_CHECKING
from ide.core.data_sampler import DataSampler
import numpy as np
from ide.core.oracle.interpolation_strategy import InterpolationStrategy

if TYPE_CHECKING:
    from typing import Tuple, List, Dict
    from nptyping import NDArray, Number, Shape

from ide.core.query.query_pool import QueryPool

class AverageInterpolationStrategy(InterpolationStrategy):
    def interpolate(self, real_queries, data_points: Tuple[NDArray[Number, Shape["query_nr, sample_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, sample_nr, ... result_dim"]]]) -> Tuple[NDArray[Number, Shape["query_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, ... result_dim"]]]:
        avg_quries = []
        avg_results = []
        for queries in data_points[0]:
            avg_quries.append(np.asarray([np.average(queries)]))
        for results in data_points[1]:
            avg_results.append(np.asarray([np.average(results)]))
        return (np.asarray(avg_quries),np.asarray(avg_results))

class WeightedAverageInterpolationStrategy(InterpolationStrategy):
    def interpolate(self, real_queries, data_points: Tuple[NDArray[Number, Shape["query_nr, sample_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, sample_nr, ... result_dim"]]]) -> Tuple[NDArray[Number, Shape["query_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, ... result_dim"]]]:
        queries = data_points[0]
        results = data_points[1]
        query_output = []
        result_output = []
        for i in range(len(queries)):
            #calc distance from querie 
            distances = np.abs(queries[i] - real_queries[i])
            query_output.append(np.asarray([np.average(queries[i], weights=distances)]))
            result_output.append(np.asarray([np.average(results[i], weights=distances)]))
        q = np.asarray(query_output)
        r = np.asarray(result_output)        
        return (q,r)

class RandomInterpolationStrategy(InterpolationStrategy):
    def interpolate(self, real_queries, data_points: Tuple[NDArray[Number, Shape["query_nr, sample_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, sample_nr, ... result_dim"]]]) -> Tuple[NDArray[Number, Shape["query_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, ... result_dim"]]]:
        queries = data_points[0]
        results = data_points[1]
        query_output = []
        result_output = []
        for i in range(len(queries)):
            randIdx = randint(0,len(queries[i])-1)
            query_output.append(np.asarray(queries[i][randIdx]))
            result_output.append(np.asarray(results[i][randIdx]))
        q = np.asarray(query_output)
        r = np.asarray(result_output)  
        return (q,r)

class WeightedRandomInterpolationStrategy(InterpolationStrategy):
    def interpolate(self, real_queries, data_points: Tuple[NDArray[Number, Shape["query_nr, sample_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, sample_nr, ... result_dim"]]]) -> Tuple[NDArray[Number, Shape["query_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, ... result_dim"]]]:
        queries = data_points[0]
        results = data_points[1]
        query_output = []
        result_output = []
        for i in range(len(queries)):
            #calc distance from querie
            distances = np.abs(queries[i] - real_queries[i])
            randIdx = random.choices(list(range(len(queries[i]))), weights=np.concatenate(distances))
            query_output.append(queries[i][randIdx])
            result_output.append(results[i][randIdx])
        q = np.asarray(query_output).reshape(-1,1)
        r = np.asarray(result_output).reshape(-1,1)
        return (q,r)
