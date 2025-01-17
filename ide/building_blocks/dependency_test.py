from __future__ import annotations
from abc import abstractmethod
import imp
import itertools
from math import perm
from random import random, randrange
import subprocess
from typing import TYPE_CHECKING
from unittest import result
import dcor
from sklearn.model_selection import permutation_test_score
from sklearn.neighbors import NearestNeighbors

from xicor.xicor import Xi
from data_efficient_dependency_estimation.dependency_tests_thesis.XtendedCorrel import hoeffding
from fcit import fcit
from tensorflow.math import digamma 
from dataclasses import dataclass
from nptyping import NDArray
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import chi2_contingency
import numpy as np
import pandas as pd
from ide.building_blocks.multi_sample_test import MultiSampleTest
from hyppo.independence.dcorr import Dcorr
from hyppo.independence.hsic import Hsic
from hyppo.independence.hhg import HHG
from hyppo.independence.mgc import MGC
from hyppo.independence.kmerf import KMERF
from hyppo.independence.base import IndependenceTest
from hyppo.tools.common import perm_test

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
class NoDependencyTest(DependencyTest):

    def test(self):
        return 0,0,0

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
        quries = self.exp_modules.queried_data_pool.queries
        
        if results.shape[1] == 1:
            x = quries
            y = results
            x = [item for sublist in x for item in sublist]
            y = [item for sublist in y for item in sublist]
            t, p = pearsonr(x, y)
        elif results.shape[1] == 2:
            x = results[:,:1]
            y = results[:,1:2]
            x = [item for sublist in x for item in sublist]
            y = [item for sublist in y for item in sublist]
            t, p = pearsonr(x, y)
        else:
            pair_wise = []
            for a, b in itertools.combinations(results.transpose(),2):
                pair_wise.append(pearsonr(a, b))
            ts = [i[0] for i in pair_wise]
            max_val = max(ts, key=abs)
            max_index = ts.index(max_val)
            t,p = pair_wise[max_index]
        return t,p,0

@dataclass
class Spearmanr(DependencyTest):

    def test(self):
        results = self.exp_modules.queried_data_pool.results
        quries = self.exp_modules.queried_data_pool.queries
        
        if results.shape[1] == 1:
            x = quries
            y = results
            x = [item for sublist in x for item in sublist]
            y = [item for sublist in y for item in sublist]
            t, p = spearmanr(x, y)
        elif results.shape[1] == 2:
            x = results[:,:1]
            y = results[:,1:2]
            t, p = spearmanr(x, y)
        else:
            pair_wise = []
            for a, b in itertools.combinations(results.transpose(),2):
                pair_wise.append(spearmanr(a, b))
            ts = [i[0] for i in pair_wise]
            max_val = max(ts, key=abs)
            max_index = ts.index(max_val)
            t,p = pair_wise[max_index]
        return t,p,0
@dataclass
class Kendalltau(DependencyTest):

    def test(self):
        results = self.exp_modules.queried_data_pool.results
        quries = self.exp_modules.queried_data_pool.queries
        
        if results.shape[1] == 1:
            x = quries
            y = results
            x = [item for sublist in x for item in sublist]
            y = [item for sublist in y for item in sublist]
            t, p = kendalltau(x, y)
        elif results.shape[1] == 2:
            x = results[:,:1]
            y = results[:,1:2]
            t, p = kendalltau(x, y)
        else:    
            pair_wise = []
            for a, b in itertools.combinations(results.transpose(),2):
                pair_wise.append(kendalltau(a, b))
            ts = [i[0] for i in pair_wise]
            max_val = max(ts, key=abs)
            max_index = ts.index(max_val)
            t,p = pair_wise[max_index]
        return t,p,0
@dataclass
class FIT(DependencyTest):

    def test(self):
        results = self.exp_modules.queried_data_pool.results
        quries = self.exp_modules.queried_data_pool.queries
        if results.shape[1] == 1:
            x = quries
            y = results
            x = [item for sublist in x for item in sublist]
            y = [item for sublist in y for item in sublist]
            t, p = fcit.test(x, y)
        elif results.shape[1] == 2:
            x = results[:,:1]
            y = results[:,1:2]
            p = fcit.test(x, y)
        else:    
            ps = []
            for a, b in itertools.combinations(results.transpose(),2):
                ps.append(fcit.test(a, b))
            p = max(ps)
        return 0,p,0
@dataclass
class XiCor(DependencyTest):

    def test(self):
        results = self.exp_modules.queried_data_pool.results
        quries = self.exp_modules.queried_data_pool.queries
        if results.shape[1] == 1:
            x = quries
            y = results
            x = [item for sublist in x for item in sublist]
            y = [item for sublist in y for item in sublist]
            t, p = self.xi(x, y)
        elif results.shape[1] == 2:
            x = results[:,:1]
            y = results[:,1:2]
            t, p = self.xi(x, y)
        else:    
            pair_wise = []
            for a, b in itertools.combinations(results.transpose(),2):
                pair_wise.append(self.xi(a, b))
            ts = [i[0] for i in pair_wise]
            max_val = max(ts, key=abs)
            max_index = ts.index(max_val)
            t,p = pair_wise[max_index]      
        return t, p, 0

    def xi(self,a,b):
        xi_obj = Xi(a,b)
        t = xi_obj.correlation
        p = xi_obj.pval_asymptotic(ties=False, nperm=100)
        return t,p
@dataclass
class chi_square(DependencyTest):

    def test(self):
        queries = self.exp_modules.queried_data_pool.queries

        samples = self.exp_modules.queried_data_pool.results
        r, p, dof, expected = chi2_contingency(samples)
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
        dataFile = 'run_data_store/indepTestData.csv'
        data = np.concatenate((queries,samples),axis=1)
        df = pd.DataFrame(data)
        df.to_csv(dataFile, sep=",", header='true', index=False)
 
        command = 'Rscript'
        path = 'C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/r_scripts/IndepTest.r'
        cmd = [command, path, '--vanilla'] 
        if(len(data)>50):
            output = subprocess.check_output(cmd)
            line = next(x for x in output.splitlines() if x.startswith(b'[1]'))
            p = float(line.split()[1])
        else:
            p = 0
        return 0, p,0

@dataclass
class CondIndTest(DependencyTest):

    def test(self):
        queries = self.exp_modules.queried_data_pool.queries

        samples = self.exp_modules.queried_data_pool.results
        data = np.concatenate((queries,samples),axis=1)
        dataFile = 'run_data_store/condIndTestData.csv'
        df = pd.DataFrame(data)
        df.to_csv(dataFile, sep=",", header='true', index=False)

        output = subprocess.check_output(["Rscript",  "--vanilla", "C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/r_scripts/CondIndTest.r"])
        line = next(x for x in output.splitlines() if x.startswith(b'[1]'))
        p = float(line.split()[1])
        return 0, p, 0
@dataclass
class LISTest(DependencyTest):

    def test(self):
        queries = self.exp_modules.queried_data_pool.queries

        samples = self.exp_modules.queried_data_pool.results
        data = np.concatenate((queries,samples),axis=1)
        dataFile = 'run_data_store/LISTestData.csv'
        df = pd.DataFrame(data)
        df.to_csv(dataFile, sep=",", header='true', index=False)

        if(len(data)>20):
            output = subprocess.check_output(["Rscript",  "--vanilla", "C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/r_scripts/LISTest.r"])
            p = float(output.splitlines()[1].split()[1])
            t = float(output.splitlines()[4].split()[1]) 
        else:
            p = 0
            t = 0
        return t, p,0

@dataclass
class NaivDependencyTest(DependencyTest):
    query_sampler: QuerySampler
    data_sampler: DataSampler
    multi_sample_test : MultiSampleTest

    def test(self):

        queries = self.query_sampler.sample()

        sample_queries, samples = self.data_sampler.query(queries)

        t, p = self.multi_sample_test().test(samples)

        return t, p, 0

    def __call__(self, exp_modules = None, **kwargs) -> Self:
        obj = super().__call__(exp_modules, **kwargs)
        obj.data_sampler = obj.data_sampler(exp_modules)
        obj.multi_sample_test = obj.multi_sample_test()
        obj.query_sampler = obj.query_sampler(obj.data_sampler)
        return obj

@dataclass
class hypoDcorr(DependencyTest):

    def test(self):
        results = self.exp_modules.queried_data_pool.results
        queries = self.exp_modules.queried_data_pool.queries
        x = queries
        y = results
        stat,pvalue = Dcorr().test(x,y,workers=-1,reps=100)
        return stat, pvalue, 0

@dataclass
class hypoHsic(DependencyTest):

    def test(self):
        results = self.exp_modules.queried_data_pool.results
        queries = self.exp_modules.queried_data_pool.queries
        x = queries
        y = results
        stat,pvalue = Hsic().test(x,y,workers=-1,reps=100)
        return stat, pvalue, 0

@dataclass
class hypoHHG(DependencyTest):

    def test(self):
        results = self.exp_modules.queried_data_pool.results
        queries = self.exp_modules.queried_data_pool.queries
        x = queries
        y = results
        stat,pvalue = HHG().test(x,y,workers=-1,reps=100)
        return stat, pvalue, 0

@dataclass
class hypoMGC(DependencyTest):

    def test(self):
        results = self.exp_modules.queried_data_pool.results
        queries = self.exp_modules.queried_data_pool.queries
        x = queries
        y = results
        stat,pvalue,_ = MGC().test(x,y,workers=-1,reps=100)
        return stat, pvalue, 0

@dataclass
class hypoKMERF(DependencyTest):

    def test(self):
        results = self.exp_modules.queried_data_pool.results
        queries = self.exp_modules.queried_data_pool.queries
        x = queries
        y = results
        stat,pvalue,_ = KMERF().test(y,x,workers=-1,reps=100)
        return stat, pvalue, 0
