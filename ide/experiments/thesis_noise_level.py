from random import randint, random
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
from ide.modules.oracle.data_source import CrossDataSource, DoubleLinearDataSource, GausianProcessDataSource, HourglassDataSource, HyperSphereDataSource, HypercubeDataSource, HypercubeGraphDataSource, IndependentDataSetDataSource, LineDataSource, LinearPeriodicDataSource, LogarithmicDataSource, SineDataSource, SpiralDataSource, SquareDataSource, StarDataSource, TwoParabolasDataSource, WshapeDataSource, ZDataSource, ZInvDataSource
from ide.modules.oracle.data_source_adapter import DataSourceAdapter
from ide.modules.evaluator import LogNewDataPointsEvaluator, PlotNewDataPointsEvaluator, PrintNewDataPointsEvaluator, PlotQueryDistEvaluator
from ide.building_blocks.evaluator import PlotScoresEvaluator, PlotQueriesEvaluator, PlotTestPEvaluator, BoxPlotTestPEvaluator
from ide.building_blocks.dependency_test import FIT, CondIndTest, DependencyTest, IndepTest, Kendalltau, LISTest, NaivDependencyTest, Pearson, Spearmanr, XiCor, chi_square, hypoDcorr, hypoHHG, hypoHsic, hypoKMERF, hypoMGC

from ide.core.blueprint_factory import BlueprintFactory

#class 1
#algorithms = [Spearmanr(),Kendalltau()]
#class 2
#algorithms = [hypoDcorr(),hypoHsic(),XiCor(),DependencyTestAdapter(Hoeffdings())]
#class 3
#algorithms = [DependencyTestAdapter(IMIE()),DependencyTestAdapter(CMI()),DependencyTestAdapter(HiCS()),DependencyTestAdapter(MCDE()),DependencyTestAdapter(Hoeffdings())]
#class 4
algorithms = [hypoKMERF(),hypoMGC(),hypoHHG()]
#class 5
#algorithms = [hypoDcorr(),hypoHsic()]
#algorithms = [FIT()]

synthetic_data_sources = []
for i in range(3,4,5):
    synthetic_data_sources.append(LineDataSource((1,),(2,),a=randint(-100,100),b=randint(-100,100))), 
    synthetic_data_sources.append(SquareDataSource((1,),(2,),x0=randint(-100,100)*random(),y0=randint(-100,100)*random(),s=randint(-100,100)*random())),
    synthetic_data_sources.append(SineDataSource((1,),(2,),p=randint(0,10),a=randint(0,100)))
    synthetic_data_sources.append(CrossDataSource((1,),(2,),a=100*random())),
    synthetic_data_sources.append(DoubleLinearDataSource((1,),(2,),a=100*random())), 
    synthetic_data_sources.append(HourglassDataSource((1,),(2,),a=100*random())),
    synthetic_data_sources.append(StarDataSource((1,),(2,),w=random())),
    synthetic_data_sources.append(ZDataSource((1,),(2,),a=100*random())),
    synthetic_data_sources.append(ZInvDataSource((1,),(2,),100*random()))
    #synthetic_data_sources.append(GausianProcessDataSource((1,),(2,)))
    synthetic_data_sources.append(LinearPeriodicDataSource((1,),(2,),p=10*random(),a=100*random()))
    synthetic_data_sources.append(HypercubeDataSource((1,),(2,),w=random()))
    synthetic_data_sources.append(HypercubeGraphDataSource((1,),(2,),w=random()))
    synthetic_data_sources.append(HyperSphereDataSource((1,),(2,)))
    synthetic_data_sources.append(WshapeDataSource((1,),(2,),r=100*random()))
    synthetic_data_sources.append(SpiralDataSource((1,),(2,),r=randint(1,10)))
    synthetic_data_sources.append(TwoParabolasDataSource((1,),(2,),r=randint(1,10)))
    synthetic_data_sources.append(LogarithmicDataSource((1,),(2,)))

blueprints = []
noise_array = [0.0]
for noise in noise_array: 
    blueprints+=BlueprintFactory.getBlueprintsForSyntheticDataCompuationIntensive(algorithms=algorithms, dataSources=synthetic_data_sources, noiseRatio=noise)