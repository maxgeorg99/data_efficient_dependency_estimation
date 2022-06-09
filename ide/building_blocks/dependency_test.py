from __future__ import annotations
from abc import abstractmethod
import subprocess
from typing import TYPE_CHECKING
import dcor

from xicor.xicor import Xi
from data_efficient_dependency_estimation.dependency_tests_thesis.XtendedCorrel import hoeffding
from fcit import fcit

from dataclasses import dataclass
from nptyping import NDArray
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
from ide.building_blocks.multi_sample_test import MultiSampleTest

from ide.core.experiment_module import ExperimentModule

if TYPE_CHECKING:
    from typing_extensions import Self #type: ignore
    from ide.core.data_sampler import DataSampler
    from ide.core.query.query_sampler import QuerySampler


@dataclass
class DependencyTest(ExperimentModule):
    
    @abstractmethod 
    def test(self):
        raise NotImplementedError

@dataclass
class GMI(DependencyTest):

    def test(self, samples: NDArray):
        pass

@dataclass
class DIMID(DependencyTest):

    def test(self, samples: NDArray):
        pass

@dataclass
class IMIE(DependencyTest):

    def test(self, samples: NDArray):
        pass

@dataclass
class PeakSim(DependencyTest):

    def test(self, samples: NDArray):
        pass
@dataclass
class Pearson(DependencyTest):

    def test(self):
        results = self.exp_modules.queried_data_pool.results
        t, p = pearsonr(results, results)
        return [t],[p],[0]
@dataclass
class Spearmanr(DependencyTest):

    def test(self, samples: NDArray):
        x = [item for sublist in samples for item in sublist]
        y = [item for sublist in x for item in sublist]
        t, p = spearmanr(y, y)
        return [t],[p],[0]
@dataclass
class Kendalltau(DependencyTest):

    def test(self, samples: NDArray):
        x = [item for sublist in samples for item in sublist]
        y = [item for sublist in x for item in sublist]
        t, p = kendalltau(y, y)
        return [t],[p],[0]

@dataclass
class HiCS(DependencyTest):

    def test(self, samples: NDArray):
        x = np.asarray([item.tolist() for sublist in samples for item in sublist])
        dataFile = 'run_data_store/hicsData.csv'
        df = pd.DataFrame(x)
        df.to_csv(dataFile, sep=",", header='true', index=False)

        #output = subprocess.check_output('java -jar target/scala-2.12/MCDE-experiments-1.0.jar -t EstimateDependency -f src/test/resources/data/Independent-5-0.0.csv -a HiCS -m 1 -p 1')
        output = subprocess.check_output('java -jar target/scala-2.12/MCDE-experiments-1.0.jar -t EstimateDependency -f run_data_store/hicsData.csv -a HiCS -m 1 -p 1')
        score = float(output.splitlines()[5])
        return score,0

@dataclass
class MCDE(DependencyTest):

    def test(self, samples: NDArray):
        x = np.asarray([item.tolist() for sublist in samples for item in sublist])
        dataFile = 'run_data_store/mcdeData.csv'
        df = pd.DataFrame(x)
        df.to_csv(dataFile, sep=",", header='true', index=False)

        #output = subprocess.check_output('java -jar target/scala-2.12/MCDE-experiments-1.0.jar -t EstimateDependency -f src/test/resources/data/Independent-5-0.0.csv -a MWP -m 1 -p 1')
        output = subprocess.check_output('java -jar target/scala-2.12/MCDE-experiments-1.0.jar -t EstimateDependency -f run_data_store/mcdeData.csv -a MWP -m 1 -p 1')
        score = float(output.splitlines()[4])
        return score,0
@dataclass
class FIT(DependencyTest):

    def test(self, samples: NDArray):
        samples = np.array(samples)
        p = fcit.test(samples, samples)
        return 0,p,0
@dataclass
class XiCor(DependencyTest):

    def test(self, samples: NDArray):
        x = [item for sublist in samples for item in sublist]
        y = [item for sublist in x for item in sublist]
        xi_obj = Xi(y,y)
        #t = 0 if and only if X and Y are independent
        t = xi_obj.correlation
        p = xi_obj.pval_asymptotic(ties=False, nperm=1000)      
        return t, p, 0
@dataclass
class Hoeffdings(DependencyTest):

    def test(self, samples: NDArray):
        x = [item for sublist in samples for item in sublist]
        y = [item for sublist in x for item in sublist]
        samples = np.array([y])
        p = hoeffding(samples,samples)    
        return 0, p,0

@dataclass
class dCor(DependencyTest):

    def test(self, samples: NDArray):
        x = [item for sublist in samples for item in sublist]
        y = [item for sublist in x for item in sublist]
        samples = np.array([y])
        r = dcor.distance_correlation(samples, samples)
        #calculate p value
        return r,0,0

@dataclass
class chi_square(DependencyTest):

    def test(self, samples: NDArray):
        r, p, dof, expected = chi2_contingency(samples, samples)
        return r,p

@dataclass
class A_dep_test(DependencyTest):

    def test(self, samples: NDArray):
        packageName = 'IndependenceTests'
        #IndependenceTests = rpackages.importr('IndependenceTests')
        #IndependenceTests.A.dep.tests(samples)

@dataclass
class IndepTest(DependencyTest):

    def test(self, samples: NDArray):
        pass

@dataclass
class CondIndTest(DependencyTest):

    def test(self, samples: NDArray):
        pass

@dataclass
class LISTest(DependencyTest):

    def test(self, samples: NDArray):
        pass

@dataclass
class NaivDependencyTest(DependencyTest):
    query_sampler: QuerySampler
    data_sampler: DataSampler
    multi_sample_test : MultiSampleTest

    def test(self):

        queries = self.query_sampler.sample(self.data_sampler.query_pool)

        sample_queries, samples = self.data_sampler.sample(queries)

        t, p = self.multi_sample_test.test(samples)

        return t, p, 0

    def __call__(self, exp_modules = None, **kwargs) -> Self:
        obj = super().__call__(exp_modules, **kwargs)
        obj.data_sampler = obj.data_sampler(exp_modules)
        obj.multi_sample_test = obj.multi_sample_test()
        obj.query_sampler = obj.query_sampler()
        return obj
