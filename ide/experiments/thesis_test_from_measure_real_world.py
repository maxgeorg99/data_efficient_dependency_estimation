from ide.building_blocks.dependency_measure import dCor
from ide.building_blocks.dependency_test_adapter import DependencyTestAdapter
from ide.core.blueprint_factory import BlueprintFactory
from ide.core.oracle.augmentation import NoAugmentation
from ide.core.oracle.interpolation_strategy import InterpolationStrategy
from ide.modules.oracle.interpolation_strategy.interpolation_strategy import AverageInterpolationStrategy
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
from ide.modules.oracle.data_source import InterpolatingDataSource, LineDataSource, SquareDataSource
from ide.modules.evaluator import LogNewDataPointsEvaluator, PlotNewDataPointsEvaluator, PrintNewDataPointsEvaluator, PlotQueryDistEvaluator
from ide.building_blocks.evaluator import DataEfficiencyEvaluator, LogBluePrint, PlotScoresEvaluator, PlotQueriesEvaluator, PlotTestPEvaluator, BoxPlotTestPEvaluator
from ide.building_blocks.dependency_test import DependencyTest, Pearson

real_world_data_sources = RealWorldDataSourceFactory().get_data_source('chf')
test = DependencyTestAdapter(
                dependency_measure=dCor(),
                datasource=real_world_data_sources
            )

blueprints = BlueprintFactory().getBlueprintsForRealWorldData(dataSources=[real_world_data_sources],algorithms=[test])