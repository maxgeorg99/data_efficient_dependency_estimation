from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable
from ide.core.data.queried_data_pool import QueriedDataPool
from ide.core.evaluator import Evaluator
from ide.core.experiment_modules import ExperimentModules
from ide.core.oracle.oracle import Oracle
from ide.core.query.query_optimizer import QueryOptimizer
from ide.core.query.query_sampler import QuerySampler

from ide.core.stopping_criteria import StoppingCriteria

if TYPE_CHECKING:
    from ide.core.blueprint import Blueprint
    from nptyping import NDArray, Number, Shape

@dataclass
class Experiment():
    exp_nr: int
    exp_name: str
    repeat: int
    stopping_criteria: StoppingCriteria
    oracle: Oracle
    initial_query_sampler: QuerySampler
    experiment_modules: ExperimentModules
    evaluators: Iterable[Evaluator]

    def __init__(self, bp: Blueprint, exp_nr: int) -> None:
        self.repeat = bp.repeat
        self.exp_nr = exp_nr
        self.exp_path = bp.exp_path
        self.exp_name = bp.exp_name

        self.oracle = bp.oracle()

        self.queried_data_pool = bp.queried_data_pool(self.oracle.query_pool, self.oracle.data_pool)
        self.experiment_modules = bp.experiment_modules(self.queried_data_pool, self.oracle.data_pool)

        self.initial_query_sampler = bp.initial_query_sampler(self.oracle)
        self.query_optimizer = bp.query_optimizer(self.experiment_modules)
        self.stopping_criteria = bp.stopping_criteria()
        self.evaluators = bp.evaluators



    def run(self):
        iteration = 0
        queries = self.initial_query_sampler.sample()
        while self.stopping_criteria.next(iteration):
            queries = self.loop(iteration, queries)
            self.experiment_modules.run()
            iteration += 1
        return self.exp_nr

    def loop(self, iteration: int, queries: NDArray[Number, Shape["query_nr, ... query_dims"]]) -> NDArray[Number, Shape["query_nr, ... query_dims"]]:
        data_points = self.oracle.query(queries)
        self.queried_data_pool.add(data_points)
        queries, scores = self.query_optimizer.select()
        return queries
            