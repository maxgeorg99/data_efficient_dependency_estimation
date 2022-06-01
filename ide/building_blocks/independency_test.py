from __future__ import annotations
from random import random
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
from ide.building_blocks.multi_sample_test import MultiSampleTest

from ide.core.data_sampler import DataSampler
from ide.core.configuration import Configurable
from ide.modules.oracle.data_source_adapter import DataSourceAdapter

if TYPE_CHECKING:
    from ide.core.configuration import Configurable
    from typing import Tuple, List

@dataclass
class IndependenceTest(Configurable):
    #return list of predictions and p values
    def test(self, samples: Tuple[float,...]) -> Tuple[List[int],List[float]]:
        raise NotImplementedError

@dataclass
class IndependenceTestForMeasure(Configurable):
    iterations: int
    dataset: List
    independentDataset: List
    dependencyMeasure: MultiSampleTest
    scores = []

    def __init__(self, number_iterations, dataset, test:MultiSampleTest):
        self.iterations = number_iterations
        self.test = test
        self.dataset = dataset
        self.independentDataset = np.random.random_sample(len(dataset))
        self.createTest()

    def createTest(self):
        t,p = self.dependencyMeasure.test(self.dataset)
        self.scores.append(t)
        #measure independt data
        for i in range(self.iterations-1):
            t,p = self.dependencyMeasure.test(self.independentDataset)
            self.scores.append(t)

    def runMeasures(self):
        pass
    
    def rankByScore(self):
        pass

    def test(self, samples: Tuple[float,...]) -> Tuple[List[int],List[float]]:
        raise NotImplementedError