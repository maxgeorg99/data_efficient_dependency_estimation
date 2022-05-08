from typing import List
from ide.building_blocks.dependency_test import DependencyTest
from ide.building_blocks.experiment_modules import DependencyExperiment
from ide.building_blocks.multi_sample_test import MultiSampleTest
from ide.building_blocks.selection_criteria import QueryTestNoSelectionCritera
from ide.core.blueprint import Blueprint
from ide.core.evaluator import Evaluator
from ide.core.oracle.data_source import DataSource
from ide.core.oracle.oracle import Oracle
from ide.core.query.query_optimizer import NoQueryOptimizer
from ide.modules.data_sampler import KDTreeRegionDataSampler
from ide.modules.oracle.augmentation import NoiseAugmentation
from ide.modules.queried_data_pool import FlatQueriedDataPool
from ide.modules.query.query_sampler import LatinHypercubeQuerySampler, RandomChoiceQuerySampler, UniformQuerySampler
from ide.modules.stopping_criteria import LearningStepStoppingCriteria

class BlueprintFactory():
    blueprints = []
    
    def __init__(self, algorithms: List[MultiSampleTest], dataSources: List[DataSource], evaluators: List[Evaluator]):
        for dataSource in dataSources:
            for test in algorithms:
                self.blueprints.append(Blueprint(
                    #define fitting repeat and querie nums
                    repeat=10,
                    stopping_criteria= LearningStepStoppingCriteria(10),
                    oracle = Oracle(
                        data_source=dataSource,
                        augmentation=NoiseAugmentation
                    ),
                    queried_data_pool=FlatQueriedDataPool(),
                    initial_query_sampler=LatinHypercubeQuerySampler(num_queries=2),
                    query_optimizer=NoQueryOptimizer(
                        selection_criteria=QueryTestNoSelectionCritera(),
                        num_queries=4,
                        query_sampler=UniformQuerySampler(),
                    ),
                    experiment_modules=DependencyExperiment(
                        dependency_test=DependencyTest(
                            query_sampler = RandomChoiceQuerySampler(num_queries=10),
                            #use default option ehre
                            data_sampler = KDTreeRegionDataSampler(0.05),
                            multi_sample_test = test 
                            ),
                        ),
                    evaluators=evaluators,
                    exp_name=type(test).__name__,
                    )
                )

    def getBlueprints(self):
        return self.blueprints        