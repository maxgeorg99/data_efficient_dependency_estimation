from __future__ import annotations
import subprocess
from typing import TYPE_CHECKING

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

@dataclass
class dHSIC(MultiSampleTest):

    def test(self, samples):
        pass

        #packageName = ('dHSIC')
        #package = rpackages.importr(packageName)
        #t, p = package.dHSIC(samples, samples,kernel=("gaussian","discrete"))
        #return t,p

@dataclass
class CMI(MultiSampleTest):

    def test(self, samples):
        dataFile = 'cmiData.csv'
        df = pd.DataFrame(samples)
        df.to_csv(dataFile, sep=",", header='true')

        t = 0
        p = subprocess.check_output(['java', '-jar', 'MCDE-experiments-1.0.jar', '-t EstimateDependency', '-f ' + dataFile ,'-a CMI'])

        return t,p

@dataclass
class GMI(MultiSampleTest):

    def test(self, samples: NDArray):
        pass

@dataclass
class DIMID(MultiSampleTest):

    def test(self, samples: NDArray):
        pass

@dataclass
class IMIE(MultiSampleTest):

    def test(self, samples: NDArray):
        pass

@dataclass
class PeakSim(MultiSampleTest):

    def test(self, samples: NDArray):
        pass
@dataclass
class Pearson(MultiSampleTest):

    def test(self, samples: NDArray):
        x = [item for sublist in samples for item in sublist]
        y = [item for sublist in x for item in sublist]
        t, p = pearsonr(y, y)
        return [t],[p]
@dataclass
class Spearmanr(MultiSampleTest):

    def test(self, samples: NDArray):
        x = [item for sublist in samples for item in sublist]
        y = [item for sublist in x for item in sublist]
        t, p = spearmanr(y, y)
        return [t],[p]
@dataclass
class Kendalltau(MultiSampleTest):

    def test(self, samples: NDArray):
        x = [item for sublist in samples for item in sublist]
        y = [item for sublist in x for item in sublist]
        t, p = kendalltau(y, y)
        return [t],[p]

@dataclass
class HiCS(MultiSampleTest):

    def test(self, samples: NDArray):
        dataFile = 'hicsData.csv'
        df = pd.DataFrame(samples)
        df.to_csv(dataFile, sep=",", header='true')

        p = subprocess.check_output(['java', '-jar', 'MCDE-experiments-1.0', '-t EstimateDependency', '-f ' + dataFile ,'-a HiCS'])

@dataclass
class MCDE(MultiSampleTest):

    def test(self, samples: NDArray):
        dataFile = 'mcdeData.csv'
        df = pd.DataFrame(samples)
        df.to_csv(dataFile, sep=",", header='true')
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
class A_dep_test(MultiSampleTest):

    def test(self, samples: NDArray):
        pass

@dataclass
class Hoeffdings(MultiSampleTest):

    def test(self, samples: NDArray):
        p = hoeffding(*samples)    
        return 0, p

@dataclass
class dCor(MultiSampleTest):

    def test(self, samples: NDArray):
        r = dcor.distance_correlation(samples, samples)
        #calculate p value
        return r,0

@dataclass
class chi_square(MultiSampleTest):

    def test(self, samples: NDArray):
        r, p = chi2_contingency(samples, samples)
        return r,p


@dataclass
class IndepTest(MultiSampleTest):

    def test(self, samples: NDArray):
        pass

@dataclass
class CondIndTest(MultiSampleTest):

    def test(self, samples: NDArray):
        pass

@dataclass
class LISTest(MultiSampleTest):

    def test(self, samples: NDArray):
        pass