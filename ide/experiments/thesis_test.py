
from ide.building_blocks.dependency_test import NoDependencyTest
from ide.building_blocks.experiment_modules import DependencyExperiment
from ide.building_blocks.selection_criteria import QueryTestNoSelectionCritera
from ide.core.blueprint import Blueprint
from ide.core.oracle.augmentation import NoAugmentation
from ide.core.oracle.oracle import Oracle
from ide.core.query.query_optimizer import NoQueryOptimizer
from ide.modules.evaluator import PlotNewDataPointsEvaluator
from ide.modules.oracle.data_source import StarDataSource
from ide.modules.queried_data_pool import FlatQueriedDataPool
from ide.modules.query.query_sampler import RandomChoiceQuerySampler, UniformQuerySampler
from ide.modules.stopping_criteria import LearningStepStoppingCriteria

synthetic_data_sources = []
synthetic_data_sources.append(StarDataSource((1,),(1,)))
blueprints = []
for d in synthetic_data_sources:
    blueprints.append(Blueprint(
                        repeat=10,
                        stopping_criteria= LearningStepStoppingCriteria(100),
                        queried_data_pool=FlatQueriedDataPool(),
                        initial_query_sampler=UniformQuerySampler(num_queries=10),
                        query_optimizer=NoQueryOptimizer(
                            selection_criteria=QueryTestNoSelectionCritera(),
                            num_queries=10,
                            query_sampler=RandomChoiceQuerySampler(),
                        ),
                        experiment_modules=
                        DependencyExperiment(
                            dependency_test=NoDependencyTest()
                        ),
                        oracle = Oracle(
                            data_source=d,
                            augmentation=NoAugmentation()
                        ),
                        evaluators=[
                            PlotNewDataPointsEvaluator()
                            ],
                        exp_name = 'test'
                        ))