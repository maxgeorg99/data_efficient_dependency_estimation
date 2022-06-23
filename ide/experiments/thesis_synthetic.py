from distribution_data_generation.data_sources.double_linear_data_source import DoubleLinearDataSource
from distribution_data_generation.data_sources.hourglass_data_source import HourglassDataSource
from distribution_data_generation.data_sources.hypercube_data_source import HypercubeDataSource
from distribution_data_generation.data_sources.graph_data_source import GraphDataSource
from distribution_data_generation.data_sources.sine_data_source import SineDataSource
from distribution_data_generation.data_sources.star_data_source import StarDataSource
from distribution_data_generation.data_sources.z_data_source import ZDataSource
from distribution_data_generation.data_sources.inv_z_data_source import InvZDataSource
from distribution_data_generation.data_sources.cross_data_source import CrossDataSource
from ide.building_blocks.dependency_measure import CMI, MCDE, HiCS, dCor, dHSIC
from ide.building_blocks.dependency_test_adapter import DependencyTestAdapter
from ide.building_blocks.multi_sample_test import KWHMultiSampleTest

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
from ide.modules.oracle.data_source import IndependentDataSetDataSource, LineDataSource, SquareDataSource
from ide.modules.oracle.data_source_adapter import DataSourceAdapter
from ide.modules.evaluator import LogNewDataPointsEvaluator, PlotNewDataPointsEvaluator, PrintNewDataPointsEvaluator, PlotQueryDistEvaluator
from ide.building_blocks.evaluator import PlotScoresEvaluator, PlotQueriesEvaluator, PlotTestPEvaluator, BoxPlotTestPEvaluator
from ide.building_blocks.dependency_test import FIT, DependencyTest, Kendalltau, NaivDependencyTest, Pearson, Spearmanr, XiCor, chi_square

from ide.core.blueprint_factory import BlueprintFactory

synthetic_data_sources = []

for i in range(2,3):
        synthetic_data_sources.append(LineDataSource((1,),(i,))),
        synthetic_data_sources.append(SquareDataSource((1,),(i,))),
        synthetic_data_sources.append(DataSourceAdapter(CrossDataSource(1,i))),
        synthetic_data_sources.append(DataSourceAdapter(DoubleLinearDataSource(1,i))),
        synthetic_data_sources.append(DataSourceAdapter(HourglassDataSource(1,i))),
        #synthetic_data_sources.append(DataSourceAdapter(HypercubeDataSource(1,i))),
        synthetic_data_sources.append(DataSourceAdapter(SineDataSource(1,i))),
        synthetic_data_sources.append(DataSourceAdapter(StarDataSource(1,i))),
        synthetic_data_sources.append(DataSourceAdapter(ZDataSource(1,i))),
        synthetic_data_sources.append(DataSourceAdapter(InvZDataSource(1,i)))

#for i in range(2,5):
#    for j in range(1,5):
#        synthetic_data_sources.append(IndependentDataSetDataSource(dims=i,id= i*j ))

blueprints = BlueprintFactory.getBlueprintsForSyntheticData(dataSources=synthetic_data_sources)