from __future__ import annotations
import subprocess
from typing import TYPE_CHECKING
#from rpy2.robjects.packages import rpackages

from dataclasses import dataclass
from nptyping import NDArray
import numpy as np
import pandas as pd
import dcor

from xicor.xicor import Xi
from data_efficient_dependency_estimation.dependency_tests_thesis.XtendedCorrel import hoeffding

from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import chi2_contingency
from scipy.stats import kruskal #type: ignore

from fcit import fcit

from ide.core.data_sampler import DataSampler
from ide.core.configuration import Configurable

if TYPE_CHECKING:
    from ide.core.configuration import Configurable
    from typing import Tuple, List

@dataclass
class MultiSampleTest(Configurable):


    def test(self, samples: Tuple[float,...]) -> Tuple[List[float],List[float]]:
        raise NotImplementedError


@dataclass
class KWHMultiSampleTest(MultiSampleTest):
    
    def test(self, samples):
        t, p = kruskal(*samples)
        return t, p