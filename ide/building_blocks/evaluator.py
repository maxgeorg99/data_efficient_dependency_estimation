from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING, Dict
from os.path import exists as file_exists

from ide.building_blocks.experiment_modules import DependencyExperiment
from dataclasses import dataclass, field
from ide.core.blueprint import Blueprint
from ide.core.evaluator import Evaluator, Evaluate
from statsmodels.stats.power import TTestIndPower
from sklearn.metrics import f1_score, roc_auc_score

import numpy as np
from matplotlib import pyplot as plot
from ide.modules.oracle.data_source import IndependentDataSetDataSource

from ide.modules.oracle.data_source_adapter import DataSourceAdapter # type: ignore
from ide.core.evaluator import LogingEvaluator, Evaluate

import numpy as np
from matplotlib import pyplot as plot # type: ignore
import os


if TYPE_CHECKING:
    from typing import List, Tuple
    from ide.core.experiment import Experiment
    from nptyping import  NDArray, Number, Shape
    from ide.building_blocks.dependency_test import DependencyTest


@dataclass
class PlotQueriesEvaluator(LogingEvaluator):
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
        queries = np.reshape(queries, (size, 2,-1))

        if self.queries is None:
            self.queries = queries
        else:
            self.queries = np.concatenate((self.queries, queries), axis=0)


        heatmap, xedges, yedges = np.histogram2d(self.queries[:,0,0], self.queries[:,1,0], bins=10)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

        
        fig = plot.figure(self.fig_name)
        plot.imshow(heatmap.T, extent=extent, origin='lower')
        plot.title(self.fig_name)
        plot.colorbar()
        if self.interactive: plot.show()
        else:
            plot.savefig(f'{self.path}/{self.fig_name}_{self.iteration:05d}.png')
            plot.clf()

        self.iteration += 1

@dataclass
class PlotScoresEvaluator(LogingEvaluator):
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
        test_scores = np.reshape(scores, (size,2,-1))

        fig = plot.figure(self.fig_name)
        plot.scatter(test_queries[:,0,0], test_queries[:,1,0], c=test_scores[:,0,0])
        plot.title(self.fig_name)
        plot.colorbar()
        if self.interactive: plot.show()
        else:
            plot.savefig(f'{self.path}/{self.fig_name}_{self.iteration:05d}.png')
            plot.clf()

        self.iteration += 1

        return scores



@dataclass
class PlotTestPEvaluator(LogingEvaluator):
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
class BoxPlotTestPEvaluator(LogingEvaluator):
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
        t,p,v = result

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
            plot.savefig(f'{self.path}/{self.fig_name}_{self.iteration:05d}.png',dpi=500)
            plot.clf()

        self.iteration += 1
@dataclass
class DataEfficiencyEvaluator(Evaluator):
    folder: str = field(default = "log",repr=False)
    evaluator:str = field(default = "DataEfficiency",repr=False)

    ts: List[float] = field(init=False, default_factory=list,repr=False)
    ps: List[float] = field(init=False, default_factory=list,repr=False)
    vs: List[float] = field(init=False, default_factory=list,repr=False)
    pt: List[float] = field(init=False, default_factory=list,repr=False)
    pss: List[float] = field(init=False, default_factory=list,repr=False)
    predictions: List[int] = field(init=False, default_factory=list,repr=False)
    iteration: int = field(init = False, default = 0,repr=False)

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
        self.vs = []

        self.iteration = 0

        if isinstance(self.experiment.experiment_modules, DependencyExperiment):
            self.experiment.experiment_modules.dependency_test.test = Evaluate(self.experiment.experiment_modules.dependency_test.test)
            self.experiment.experiment_modules.dependency_test.test.post(self.save_test_result)
        if isinstance(self.experiment.oracle.data_source, DataSourceAdapter) :
            self.name = type(self.experiment.oracle.data_source.distribution_data_source).__name__ + str(self.experiment.oracle.data_source.query_shape[0]) + 'x' + str(self.experiment.oracle.data_source.result_shape[0])
        elif isinstance(self.experiment.oracle.data_source, IndependentDataSetDataSource) :
            self.name = type(self.experiment.oracle.data_source).__name__ + str(self.experiment.oracle.data_source.id)
        else:
            self.name = type(self.experiment.oracle.data_source).__name__ + str(self.experiment.oracle.data_source.query_shape[0]) + 'x' + str(self.experiment.oracle.data_source.result_shape[0])

        self.experiment.run = Evaluate(self.experiment.run)
        self.experiment.run.post(self.numpy_save_results)
    
    def save_test_result(self, result):
        t,p,v = result

        self.vs.append(v)
        tvalue = 0
        if (type(p) is list or isinstance(p, np.ndarray)):
            tvalue = t[0]
            self.ps.append(p[0])
            self.ts.append(tvalue)
        else: 
            tvalue = t
            self.ps.append(p)
            self.ts.append(tvalue)

   
    def numpy_save_results(self, _):
        file_name = f'{self.folder}/{self.experiment.exp_name}_{self.name}_{self.experiment.exp_nr}.npz'
        np.savez(file_name, pValues=self.ps, score=self.ts, var=self.vs)

@dataclass
class LogBluePrint(Evaluator):
    folder:str = field(default="log", repr=False)
    evaluator:str = field(default="Experiment", repr=False)
    printed:bool = field(default=False, repr=False)

    def setup_logger(self, l_name, log_file_name):

        handler = logging.FileHandler(log_file_name)        

        logger = logging.getLogger(l_name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        return logger

    def register(self, experiment: Experiment):
        super().register(experiment)

        self.experiment.run = Evaluate(self.experiment.run)
        self.experiment.run.post(self.log_blueprint)

    def log_blueprint(self, _):
        if(not self.printed):
            self.printed =  True
            experiment = self.experiment
            blueprint = Blueprint(
                        repeat=experiment.repeat,
                        stopping_criteria= experiment.stopping_criteria,
                        oracle = experiment.oracle,
                        queried_data_pool=experiment.queried_data_pool,
                        initial_query_sampler=experiment.initial_query_sampler,
                        query_optimizer=experiment.query_optimizer,
                        experiment_modules=experiment.experiment_modules,
                        evaluators=experiment.evaluators,
                        exp_name=experiment.exp_name,
            )
            l_name = f'{self.experiment.exp_name}'
            f_name = f'{self.folder}/{self.evaluator}_{self.experiment.exp_name}.txt'
            logger = self.setup_logger(l_name, f_name)
            logger.info(blueprint)
