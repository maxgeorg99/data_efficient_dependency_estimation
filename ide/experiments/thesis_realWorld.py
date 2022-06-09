from data_efficient_dependency_estimation.dependency_tests_thesis.Kendall import Kendall
from data_efficient_dependency_estimation.dependency_tests_thesis.Pearson import Pearson
from ide.modules.oracle.data_source import RealWorldDataSetDataSource
from ide.modules.oracle.real_world_data_source_factory import RealWorldDataSourceFactory
from ide.modules.queried_data_pool import FlatQueriedDataPool
from ide.modules.data_sampler import KDTreeKNNDataSampler, KDTreeRegionDataSampler
from ide.core.oracle.oracle import Oracle
from ide.core.query.query_optimizer import NoQueryOptimizer
from ide.modules.query.query_sampler import RandomChoiceQuerySampler, UniformQuerySampler, LatinHypercubeQuerySampler
from ide.building_blocks.selection_criteria import QueryTestNoSelectionCritera
from ide.building_blocks.test_interpolation import KNNTestInterpolator
from ide.building_blocks.two_sample_test import MWUTwoSampleTest
from ide.building_blocks.experiment_modules import DependencyExperiment
from ide.modules.oracle.augmentation import NoiseAugmentation
from ide.modules.stopping_criteria import LearningStepStoppingCriteria
from ide.core.blueprint import Blueprint
from ide.modules.oracle.data_source_adapter import DataSourceAdapter
from ide.modules.evaluator import LogNewDataPointsEvaluator, PlotNewDataPointsEvaluator, PrintNewDataPointsEvaluator, PlotQueryDistEvaluator
from ide.building_blocks.evaluator import PlotScoresEvaluator, PlotQueriesEvaluator, PlotTestPEvaluator, BoxPlotTestPEvaluator
from ide.building_blocks.dependency_test import DependencyTest, NaivDependencyTest
from ide.building_blocks.multi_sample_test import KWHMultiSampleTest

from ide.core.blueprint_factory import BlueprintFactory

real_world_data_sources = RealWorldDataSourceFactory().get_data_source('chf')
test = NaivDependencyTest(
            query_sampler = RandomChoiceQuerySampler(num_queries=20),
            data_sampler = KDTreeRegionDataSampler(0.05),
            multi_sample_test=KWHMultiSampleTest()
        )

blueprints = BlueprintFactory().getBlueprintsForRealWorldData(algorithms=[test],dataSources=[real_world_data_sources])