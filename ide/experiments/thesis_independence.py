import imp
from random import randint, random
from distribution_data_generation.data_sources.double_linear_data_source import DoubleLinearDataSource
from distribution_data_generation.data_sources.hourglass_data_source import HourglassDataSource
from distribution_data_generation.data_sources.hypercube_data_source import HypercubeDataSource
from distribution_data_generation.data_sources.graph_data_source import GraphDataSource
from distribution_data_generation.data_sources.star_data_source import StarDataSource
from distribution_data_generation.data_sources.z_data_source import ZDataSource
from distribution_data_generation.data_sources.inv_z_data_source import InvZDataSource
from distribution_data_generation.data_sources.cross_data_source import CrossDataSource
from distribution_data_generation.data_sources.random_data_source import RandomDataSource
from distribution_data_generation.data_sources.multi_gausian_data_source import MultiGausianDataSource
from ide.building_blocks.dependency_measure import CMI, IMIE, MCDE, HiCS, Hoeffdings, dCor, dHSIC
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
from ide.modules.oracle.data_source import IndependentDataSetDataSource, LineDataSource, SineDataSource, SquareDataSource
from ide.modules.oracle.data_source_adapter import DataSourceAdapter
from ide.modules.evaluator import LogNewDataPointsEvaluator, PlotNewDataPointsEvaluator, PrintNewDataPointsEvaluator, PlotQueryDistEvaluator
from ide.building_blocks.evaluator import PlotScoresEvaluator, PlotQueriesEvaluator, PlotTestPEvaluator, BoxPlotTestPEvaluator
from ide.building_blocks.dependency_test import FIT, CondIndTest, DependencyTest, IndepTest, Kendalltau, LISTest, NaivDependencyTest, Pearson, Spearmanr, XiCor, chi_square, hypoDcorr, hypoHHG, hypoHsic, hypoKMERF, hypoMGC
from hyppo.independence import Dcorr
from hyppo.independence import Hsic

from ide.core.blueprint_factory import BlueprintFactory

#class 1
algorithms = [Pearson(),Spearmanr(),Kendalltau()]
#class 2
#algorithms = [hypoDcorr(),hypoHsic(),XiCor(),DependencyTestAdapter(Hoeffdings())]
#class 3
#algorithms = [DependencyTestAdapter(IMIE()),DependencyTestAdapter(CMI()),DependencyTestAdapter(HiCS()),DependencyTestAdapter(MCDE())]
#class 4
#algorithms = [hypoKMERF(),hypoMGC(),hypoHHG()]
#class 5
#algorithms = [CondIndTest(),IndepTest(),LISTest(),FIT()]

synthetic_data_sources = []
for i in range(0,16):
    synthetic_data_sources.append(IndependentDataSetDataSource(id=i))

blueprints=BlueprintFactory.getBlueprintsForSyntheticData(algorithms=algorithms,dataSources=synthetic_data_sources, noiseRatio=0.0)