from __future__ import annotations
from abc import abstractmethod
import subprocess
from typing import TYPE_CHECKING
from unittest import result
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
from ide.core.query.query_pool import QueryPool

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

    def test(self):
        pass

@dataclass
class DIMID(DependencyTest):

    def test(self):
        pass

@dataclass
class IMIE(DependencyTest):

    def test(self):
        pass

@dataclass
class PeakSim(DependencyTest):

    def test(self):
        pass
@dataclass
class Pearson(DependencyTest):

    def test(self):
        results = self.exp_modules.queried_data_pool.results
        x = [item for sublist in results for item in sublist]
        t, p = pearsonr(x, x)
        return t,p,0
@dataclass
class Spearmanr(DependencyTest):

    def test(self):
        results = self.exp_modules.queried_data_pool.results
        x = [item for sublist in results for item in sublist]
        t, p = spearmanr(x, x)
        return t,p,0
@dataclass
class Kendalltau(DependencyTest):

    def test(self):
        results = self.exp_modules.queried_data_pool.results
        x = [item for sublist in results for item in sublist]
        t, p = kendalltau(x, x)
        return t,p,0
@dataclass
class FIT(DependencyTest):

    def test(self):
        queries = self.exp_modules.queried_data_pool.queries

        samples = self.exp_modules.queried_data_pool.results
        samples = np.array(samples)
        p = fcit.test(samples, samples)
        return 0,p,0
@dataclass
class XiCor(DependencyTest):

    def test(self):
        queries = self.exp_modules.queried_data_pool.queries

        samples = self.exp_modules.queried_data_pool.results
        x = [item for sublist in samples for item in sublist]
        xi_obj = Xi(x,x)
        #t = 0 if and only if X and Y are independent
        t = xi_obj.correlation
        p = xi_obj.pval_asymptotic(ties=False, nperm=1000)      
        return t, p, 0
@dataclass
class Hoeffdings(DependencyTest):

    def test(self):
        queries = self.exp_modules.queried_data_pool.queries

        samples = self.exp_modules.queried_data_pool.results
        x = [item for sublist in samples for item in sublist]
        y = [item for item in x]
        samples = np.array(y)
        p = hoeffding(samples,samples)    
        return 0, p,0

@dataclass
class dCor(DependencyTest):

    def test(self):
        queries = self.exp_modules.queried_data_pool.queries

        samples = self.exp_modules.queried_data_pool.results
        x = [item for sublist in samples for item in sublist]
        y = [item for sublist in x for item in sublist]
        samples = np.array([y])
        r = dcor.distance_correlation(samples, samples)
        #calculate p value
        return r,0,0

@dataclass
class chi_square(DependencyTest):

    def test(self):
        queries = self.exp_modules.queried_data_pool.queries

        samples = self.exp_modules.queried_data_pool.results
        r, p, dof, expected = chi2_contingency(samples, samples)
        return r,p,0

@dataclass
class A_dep_test(DependencyTest):

    def test(self):
        queries = self.exp_modules.queried_data_pool.queries

        samples = self.exp_modules.queried_data_pool.results
        return 0, 0, 0

@dataclass
class IndepTest(DependencyTest):

    def test(self):
        queries = self.exp_modules.queried_data_pool.queries

        samples = self.exp_modules.queried_data_pool.results
        x = np.asarray([item.tolist() for sublist in samples for item in sublist])
        dataFile = 'run_data_store/indepTestData.csv'
        df = pd.DataFrame(x)
        df.to_csv(dataFile, sep=",", header='true', index=False)
 
        command = 'Rscript'
        path = 'C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/r_scripts/IndepTest.r'
        cmd = [command, path, '--vanilla'] 
        if(len(x)>50):
            output = subprocess.check_output(cmd)
            if(output.splitlines()[8].split()[0].startswith(b'[1]')):
                p = float(output.splitlines()[8].split()[1])
                t = float(output.splitlines()[8].split()[2])
            elif(output.splitlines()[12].split()[0].startswith(b'[1]')):
                p = float(output.splitlines()[12].split()[1])
                t = float(output.splitlines()[12].split()[2])
        else:
            p = 0
            t = 0
        v = 0
        return t, p,v

@dataclass
class CondIndTest(DependencyTest):

    def test(self):
        queries = self.exp_modules.queried_data_pool.queries

        samples = self.exp_modules.queried_data_pool.results
        x = np.asarray([item.tolist() for sublist in samples for item in sublist])
        dataFile = 'run_data_store/condIndTestData.csv'
        df = pd.DataFrame(x)
        df.to_csv(dataFile, sep=",", header='true', index=False)

        output = subprocess.check_output(["Rscript",  "--vanilla", "C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/r_scripts/CondIndTest.r"])
        
        t = 0
        p = float(output.splitlines()[4].split()[1])
        v = 0
        return t, p,v
@dataclass
class LISTest(DependencyTest):

    def test(self):
        queries = self.exp_modules.queried_data_pool.queries

        samples = self.exp_modules.queried_data_pool.results
        x = np.asarray([item.tolist() for sublist in samples for item in sublist])
        dataFile = 'run_data_store/LISTestData.csv'
        df = pd.DataFrame(x)
        df.to_csv(dataFile, sep=",", header='true', index=False)

        if(len(x)>20 and len(x)<200):
            output = subprocess.check_output(["Rscript",  "--vanilla", "C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/r_scripts/LISTest.r"])
            p = float(output.splitlines()[5].split()[1])
            t = float(output.splitlines()[8].split()[1])
        else:
            p = 0
            t = 0
        v = 0
        return t, p,v

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
