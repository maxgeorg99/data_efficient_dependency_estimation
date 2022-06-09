from __future__ import annotations
from typing import TYPE_CHECKING
from ide.core.data_sampler import DataSampler
import numpy as np
from ide.core.oracle.interpolation_strategy import InterpolationStrategy

if TYPE_CHECKING:
    from typing import Tuple, List, Dict
    from nptyping import NDArray, Number, Shape

from ide.core.query.query_pool import QueryPool

class AverageInterpolationStrategy(InterpolationStrategy):
    def interpolate(self, data_points: Tuple[NDArray[Number, Shape["query_nr, sample_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, sample_nr, ... result_dim"]]]) -> Tuple[NDArray[Number, Shape["query_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, ... result_dim"]]]:
        avg_quries = []
        avg_results = []
        for queries in data_points[0]:
            avg_quries.append(np.asarray([np.average(queries)]))
        for results in data_points[1]:
            avg_results.append(np.asarray([np.average(results)]))
        return (np.asarray(avg_quries),np.asarray(avg_results))

class WeightedAverageInterpolationStrategy(InterpolationStrategy):
    def interpolate(self, data_points: Tuple[NDArray[Number, Shape["query_nr, sample_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, sample_nr, ... result_dim"]]]) -> Tuple[NDArray[Number, Shape["query_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, ... result_dim"]]]:
        quries = data_points[0]
        results = []
        for qurie, result  in data_points:
            #calc distance from querie
            distances = np.abs(qurie - result)
            results.append(np.average(result, weights=distances))
        return (quries,results)

class RandomInterpolationStrategy(InterpolationStrategy):
    def interpolate(self, data_points: Tuple[NDArray[Number, Shape["query_nr, sample_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, sample_nr, ... result_dim"]]]) -> Tuple[NDArray[Number, Shape["query_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, ... result_dim"]]]:
        quries = data_points[0]
        results = []
        for qurie, result  in data_points:
            results.append(np.random.choice(result,1))
        return (quries,results)

class WeightedRandomInterpolationStrategy(InterpolationStrategy):
    def interpolate(self, data_points: Tuple[NDArray[Number, Shape["query_nr, sample_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, sample_nr, ... result_dim"]]]) -> Tuple[NDArray[Number, Shape["query_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, ... result_dim"]]]:
        quries = data_points[0]
        results = []
        for qurie, result  in data_points:
            #calc distance from querie
            distances = np.abs(qurie - result)
            results.append(np.random.choice(result,1, p=distances))
        return (quries,results)

