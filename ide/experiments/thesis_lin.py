from ide.core.oracle.augmentation import NoAugmentation
from ide.core.oracle.interpolation_strategy import InterpolationStrategy
from ide.modules.oracle.interpolation_strategy.interpolation_strategy import AverageInterpolationStrategy
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


blueprint = Blueprint(
    repeat=10,
    stopping_criteria= LearningStepStoppingCriteria(290),
    queried_data_pool=FlatQueriedDataPool(),
    initial_query_sampler=RandomChoiceQuerySampler(num_queries=10),
    query_optimizer=NoQueryOptimizer(
        selection_criteria=QueryTestNoSelectionCritera(),
        num_queries=4,
        query_sampler=RandomChoiceQuerySampler(),
    ),
    experiment_modules=DependencyExperiment(
        dependency_test=Pearson(),
        ),
        oracle = Oracle(
        data_source=InterpolatingDataSource(
            query_shape = (1,),
            result_shape = (1,),
            data_sampler = KDTreeRegionDataSampler(),
            interpolation_strategy = AverageInterpolationStrategy()
        ),
        augmentation= NoAugmentation()
    ),
    #evaluators=[PlotQueryDistEvaluator(), PlotNewDataPointsEvaluator(), PlotQueriesEvaluator(), PlotTestPEvaluator(), BoxPlotTestPEvaluator()],
    evaluators=[PlotNewDataPointsEvaluator(),LogBluePrint(),DataEfficiencyEvaluator()],
    #evaluators=[PlotNewDataPointsEvaluator(), PlotScoresEvaluator(), PlotQueriesEvaluator(), PlotTestPEvaluator(), BoxPlotTestPEvaluator()],
    exp_name='InterpolatingDataSource'
)