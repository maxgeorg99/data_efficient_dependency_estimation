from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass
from nptyping import NDArray
import numpy as np

from xicor.xicor import Xi
from XtendedCorrel import hoeffding


from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
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

@dataclass
class Pearson(MultiSampleTest):

    def test(self, samples: NDArray):
        x = [item for sublist in samples for item in sublist]
        y = [item for sublist in x for item in sublist]
        t, p = pearsonr(y, y)
        return t,p
@dataclass
class Spearmanr(MultiSampleTest):

    def test(self, samples: NDArray):
        x = [item for sublist in samples for item in sublist]
        y = [item for sublist in x for item in sublist]
        t, p = spearmanr(y, y)
        return t,p
@dataclass
class Kendalltau(MultiSampleTest):

    def test(self, samples: NDArray):
        x = [item for sublist in samples for item in sublist]
        y = [item for sublist in x for item in sublist]
        t, p = kendalltau(y, y)
        return t,p

@dataclass
class FIT(MultiSampleTest):

    def test(self, samples: NDArray):
        p = fcit.test(samples, samples)
        return 0,p
@dataclass
class XiCor(MultiSampleTest):

    def test(self, samples: NDArray):
        x = [item for sublist in samples for item in sublist]
        y = [item for sublist in x for item in sublist]
        xi_obj = Xi(y,y)
        #t = 0 if and only if X and Y are independent
        t = xi_obj.correlation
        p = xi_obj.pval_asymptotic(ties=False, nperm=1000)        
        return t, p

@dataclass
class Hoeffdings(MultiSampleTest):

    def test(self, samples: NDArray):
        p = hoeffding(*samples)    
        return 0, p