from __future__ import annotations
from typing import TYPE_CHECKING
from ide.core.data_sampler import DataSampler
import numpy as np
from ide.core.oracle.interpolation_strategy import InterpolationStrategy

if TYPE_CHECKING:
    from typing import Tuple, List, Dict
    from nptyping import NDArray, Number, Shape

from ide.core.query.query_pool import QueryPool

class MinimumDistanceInterpolation(InterpolationStrategy):
    def interpolate(self, data_points: Tuple[NDArray[Number, Shape["query_nr, sample_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, sample_nr, ... result_dim"]]]) -> Tuple[NDArray[Number, Shape["query_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, ... result_dim"]]]:
        real_quries = self.query_pool().all_queries()
        q = []
        r = []
        for i in data_points[0]:
            minIndex = real_quries.argmin(real_quries, key=lambda x:abs(x-i))
            q.append(real_quries[minIndex])
            r.append(self.query_pool().queries_from_index(minIndex))
        return (q,r)

class AverageInterpolation(InterpolationStrategy):
    def interpolate(self, data_points: Tuple[NDArray[Number, Shape["query_nr, sample_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, sample_nr, ... result_dim"]]]) -> Tuple[NDArray[Number, Shape["query_nr, ... query_dim"]], NDArray[Number, Shape["query_nr, ... result_dim"]]]:
        real_quries = self.query_pool().all_queries()
        results =  [self.query_pool().queries_from_index(x) for x in self.query_pool().all_queries()]
        q = []
        r = []
        for i in data_points[0]:
            q.append(i)
            r.append(np.interp(i, real_quries, results))
        return (q,r)