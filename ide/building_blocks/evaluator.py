from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING

from sklearn.metrics import f1_score, roc_auc_score

from ide.building_blocks.experiment_modules import DependencyExperiment
from dataclasses import dataclass, field
from ide.core.evaluator import Evaluator, Evaluate
from statsmodels.stats.power import TTestIndPower

import numpy as np
from matplotlib import pyplot as plot

from ide.modules.oracle.data_source_adapter import DataSourceAdapter # type: ignore


if TYPE_CHECKING:
    from typing import List
    from ide.core.experiment import Experiment
    from nptyping import  NDArray, Number, Shape
    from ide.building_blocks.dependency_test import DependencyTest


@dataclass
class PlotQueriesEvaluator(Evaluator):
    interactive: bool = False
    folder: str = "fig"
    fig_name:str = "QueryDistribution2d"

    queries: NDArray[Number, Shape["2, query_nr, ... query_dim"]] = field(init=False, default = None)
    iteration: int = field(init = False, default = 0)

    def register(self, experiment: Experiment):
        super().register(experiment)

        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

        self.experiment.oracle.query = Evaluate(self.experiment.oracle.query)
        self.experiment.oracle.query.pre(self.plot_queries)

        self.queries: NDArray[Number, Shape["2, query_nr, ... query_dim"]] = None


    def plot_queries(self, queries):

        size = queries.shape[0] // 2
        queries = np.reshape(queries, (2, size,-1))

        if self.queries is None:
            self.queries = queries
        else:
            self.queries = np.concatenate((self.queries, queries), axis=1)


        heatmap, xedges, yedges = np.histogram2d(self.queries[0,:,0], self.queries[1,:,0], bins=10)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        
        fig = plot.figure(self.fig_name)
        plot.imshow(heatmap.T, extent=extent, origin='lower')
        plot.title(self.fig_name)
        plot.colorbar()
        if self.interactive: plot.show()
        else:
            plot.savefig(f'{self.folder}/{self.fig_name}_{self.iteration:05d}.png')
            plot.clf()

        self.iteration += 1

@dataclass
class PlotScoresEvaluator(Evaluator):
    interactive: bool = False
    folder: str = "fig"
    fig_name:str = "Scores 2d"

    iteration: int = field(init = False, default = 0)

    def register(self, experiment: Experiment):
        super().register(experiment)

        if not os.path.isdir(self.folder):
            os.makedirs(self.folder)

        self.experiment.query_optimizer.selection_criteria.query = Evaluate(self.experiment.query_optimizer.selection_criteria.query)
        self.experiment.query_optimizer.selection_criteria.query.warp(self.plot_scores)

    def plot_scores(self, func, queries):

        scores = func(queries)

        size = queries.shape[0] // 2
        test_queries = np.reshape(queries, (size,2,-1))
        test_scores = scores[:size]

        fig = plot.figure(self.fig_name)
        plot.scatter(test_queries[:,0,0], test_queries[:,1,0], c=test_scores)
        plot.title(self.fig_name)
        plot.colorbar()
        if self.interactive: plot.show()
        else:
            plot.savefig(f'{self.folder}/{self.fig_name}_{self.iteration:05d}.png')
            plot.clf()

        self.iteration += 1

        return scores



@dataclass
class PlotTestPEvaluator(Evaluator):
    interactive: bool = False
    folder: str = "fig"
    fig_name:str = "P-value"

    ps: List[float] = field(init=False, default_factory=list)
    iteration: int = field(init = False, default = 0)

    def register(self, experiment: Experiment):
        super().register(experiment)

        if isinstance(self.experiment.experiment_modules, DependencyExperiment):
            self.experiment.experiment_modules.dependency_test.test = Evaluate(self.experiment.experiment_modules.dependency_test.test)
            self.experiment.experiment_modules.dependency_test.test.post(self.plot_test_result)
        else:
            raise ValueError

        self.ps = []


    def plot_test_result(self, result):
        t,p = result

        self.ps.append(p)

        fig = plot.figure(self.fig_name)
        plot.plot([i for i in range(len(self.ps))], self.ps)
        plot.title(self.fig_name)
        if self.interactive: plot.show()
        else:
            plot.savefig(f'{self.folder}/{self.fig_name}_{self.experiment.exp_name}_{self.experiment.exp_nr}_{self.iteration:05d}.png')
            plot.clf()

        self.iteration += 1


@dataclass
class BoxPlotTestPEvaluator(Evaluator):
    interactive: bool = False
    folder: str = "fig"
    fig_name:str = "Boxplot p-value"

    ps: List[float] = field(init=False, default_factory=list)
    pss: List[float] = field(init=False, default_factory=list)
    iteration: int = field(init = False, default = 0)

    def register(self, experiment: Experiment):
        super().register(experiment)

        if isinstance(self.experiment.experiment_modules, DependencyExperiment):
            self.experiment.experiment_modules.dependency_test.test = Evaluate(self.experiment.experiment_modules.dependency_test.test)
            self.experiment.experiment_modules.dependency_test.test.post(self.save_test_result)
        else:
            raise ValueError

        self.experiment.run = Evaluate(self.experiment.run)
        self.experiment.run.post(self.plot_test_results)

        self.ps = []
    
    def save_test_result(self, result):
        t,p = result

        self.ps.append(p)

    def plot_test_results(self, _):

        self.pss.append(self.ps)

        data = np.asarray(self.pss)
        positions = np.arange(data.shape[1]) + 1

        fig = plot.figure(self.fig_name)

        plot.boxplot(data, positions=positions, meanline=False, showmeans=False, showfliers=False)
        means = np.mean(data, axis=0)
        plot.plot(positions, means)
        plot.xticks(np.arange(data.shape[1], step=10),np.arange(data.shape[1], step=10))
        plot.title(self.fig_name)
        if self.interactive: plot.show()
        else:
            plot.savefig(f'{self.folder}/{self.fig_name}_{self.iteration:05d}.png',dpi=500)
            plot.clf()

        self.iteration += 1
class DataEfficiencyEvaluator(Evaluator):
    """
    Measures data efficiency metrics of an dependency estimation in an experiment.
    The metric includes 
    * Statistical Power
    * Distribution of dependency estimation scores
    * F1
    * AUC
    in propotion to the amount of samples.
    """
    folder: str = "log"
    evaluator:str = "DataEfficiency"

    evaluation_metrics: None
    effect: None
    power: None
    alpha: None
    y_true: None
    ts: List[float] = field(init=False, default_factory=list)
    ps: List[float] = field(init=False, default_factory=list)
    pt: List[float] = field(init=False, default_factory=list)
    pss: List[float] = field(init=False, default_factory=list)
    iteration: int = field(init = False, default = 0)

    def setup_logger(self, l_name, log_file_name):

        handler = logging.FileHandler(log_file_name)        

        logger = logging.getLogger(l_name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        return logger

    def register(self, experiment: Experiment):
        super().register(experiment)

        self.ps = []
        self.ts = []

        self.iteration = 0

        if isinstance(self.experiment.experiment_modules, DependencyExperiment):
            self.experiment.experiment_modules.dependency_test.test = Evaluate(self.experiment.experiment_modules.dependency_test.test)

            self.experiment.experiment_modules.dependency_test.test.post(self.save_test_result)
        else:
            raise ValueError

        self.experiment.run = Evaluate(self.experiment.run)
        self.experiment.run.post(self.log_test_results)
        """
        calculate metrics
        """
        #self.evaluation_metrics['Power'] = TTestIndPower().solve_power(self.effect, power=self.power, nobs1=None, ratio=1.0, alpha=self.alpha)
        #self.evaluation_metrics['Dependency score'] = self.ps
        #self.evaluation_metrics['F1'] = f1_score(self.y_true, result)
        #self.evaluation_metrics['AUC'] = roc_auc_score(self.y_true, result)
    
    def save_test_result(self, result):
        t,p = result

        self.ps.append(p[0])
        self.ts.append(t[0])

    def log_test_results(self, _):
        if isinstance(self.experiment.oracle.data_source, DataSourceAdapter) :
            self.name = type(self.experiment.oracle.data_source.distribution_data_source).__name__
        else:
            self.name = type(self.experiment.oracle.data_source).__name__
        l_name = f'{self.experiment.exp_name}_{self.name}'
        f_name = f'{self.folder}/{self.evaluator}_{self.experiment.exp_name}_{self.name}_{self.experiment.exp_nr}.txt'
        logger = self.setup_logger(l_name, f_name)
        logger.info('P-value:')
        logger.info(self.ps)
        logger.info('Score:')
        logger.info(self.ts)
        self.iteration += 1