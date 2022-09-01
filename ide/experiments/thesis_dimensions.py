from distribution_data_generation.data_sources.random_data_source import RandomDataSource
from ide.building_blocks.dependency_measure import CMI, IMIE, MCDE, HiCS, Hoeffdings
from ide.building_blocks.dependency_test import FIT, Kendalltau, Pearson, Spearmanr, XiCor, hypoDcorr, hypoHHG, hypoHsic, hypoKMERF, hypoMGC
from ide.building_blocks.dependency_test_adapter import DependencyTestAdapter
from ide.core.blueprint_factory import BlueprintFactory
from ide.modules.oracle.data_source import CrossDataSource, DoubleLinearDataSource, GausianProcessDataSource, HourglassDataSource, HyperSphereDataSource, HypercubeDataSource, HypercubeGraphDataSource, IndependentDataSetDataSource, LineDataSource, LinearPeriodicDataSource, LogarithmicDataSource, SineDataSource, SpiralDataSource, SquareDataSource, StarDataSource, TwoParabolasDataSource, WshapeDataSource, ZDataSource, ZInvDataSource
from ide.modules.oracle.data_source_adapter import DataSourceAdapter

#class 1
algorithms = [Pearson(),Spearmanr(),Kendalltau()]
#class 2
#algorithms = [hypoDcorr(),hypoHsic()]
#class 3
#algorithms = [DependencyTestAdapter(IMIE()),DependencyTestAdapter(CMI()),DependencyTestAdapter(HiCS()),DependencyTestAdapter(MCDE())]
#class 4
#algorithms = [hypoMGC(),hypoHHG(),hypoKMERF()]
#class 5
#algorithms = [hypoDcorr(),hypoHsic()]

synthetic_data_sources = []

for i in [1,2,3,4,5]:
        synthetic_data_sources.append(LineDataSource((1,),(i,)))
        synthetic_data_sources.append(SquareDataSource((1,),(i,)))
        synthetic_data_sources.append(SineDataSource((1,),(i,),p=4))
        synthetic_data_sources.append(CrossDataSource((1,),(i,)))
        synthetic_data_sources.append(DoubleLinearDataSource((1,),(i,)))
        synthetic_data_sources.append(HourglassDataSource((1,),(i,)))
        synthetic_data_sources.append(StarDataSource((1,),(i,)))
        synthetic_data_sources.append(ZDataSource((1,),(i,)))
        synthetic_data_sources.append(ZInvDataSource((1,),(i,)))
        synthetic_data_sources.append(WshapeDataSource((1,),(i,)))
        synthetic_data_sources.append(SpiralDataSource((1,),(i,)))
        synthetic_data_sources.append(TwoParabolasDataSource((1,),(i,)))
        synthetic_data_sources.append(LogarithmicDataSource((1,),(i,)))
        synthetic_data_sources.append(GausianProcessDataSource((1,),(i,)))
        synthetic_data_sources.append(LinearPeriodicDataSource((1,),(i,)))
        synthetic_data_sources.append(HypercubeDataSource((1,),(i,)))
        synthetic_data_sources.append(HypercubeGraphDataSource((1,),(i,)))
        synthetic_data_sources.append(HyperSphereDataSource((1,),(i,)))

blueprints=BlueprintFactory.getBlueprintsForSyntheticData(algorithms=algorithms,dataSources=synthetic_data_sources)