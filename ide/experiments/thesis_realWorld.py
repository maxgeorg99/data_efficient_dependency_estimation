from ide.building_blocks.dependency_measure import CMI, IMIE, MCDE, HiCS, Hoeffdings
from ide.building_blocks.dependency_test_adapter import DependencyTestAdapter
from ide.core.oracle.interpolation_strategy import InterpolationStrategy
from ide.modules.oracle.interpolation_strategy.interpolation_strategy import RandomInterpolationStrategy
from ide.modules.oracle.real_world_data_source_factory import RealWorldDataSourceFactory
from ide.core.blueprint import Blueprint
from ide.modules.oracle.data_source_adapter import DataSourceAdapter
from ide.building_blocks.dependency_test import FIT, CondIndTest, DependencyTest, IndepTest, Kendalltau, LISTest, NaivDependencyTest, Pearson, Spearmanr, XiCor, chi_square, hypoDcorr, hypoHHG, hypoHsic, hypoKMERF, hypoMGC

from ide.core.blueprint_factory import BlueprintFactory

#class 1
#algorithms = [Pearson(),Spearmanr(),Kendalltau()]
#class 2
#algorithms = [hypoDcorr(),hypoHsic(),XiCor(),DependencyTestAdapter(Hoeffdings())]
#class 3
#algorithms = [DependencyTestAdapter(IMIE()),DependencyTestAdapter(CMI()),DependencyTestAdapter(HiCS()),DependencyTestAdapter(MCDE())]
#class 4
#algorithms = [hypoKMERF(),hypoMGC(),hypoHHG()]
#class 5
algorithms = [CondIndTest(),IndepTest(),LISTest(),FIT()]

#real_world_data_sources = RealWorldDataSourceFactory().get_all_data_sources()
#interpolation_strat = RandomInterpolationStrategy()
#real_world_data_sources =[RealWorldDataSourceFactory().sunspot(interpolation_strategy=interpolation_strat),RealWorldDataSourceFactory().NASDAQ(interpolation_strategy=interpolation_strat), RealWorldDataSourceFactory().hipe(interpolation_strategy=interpolation_strat), RealWorldDataSourceFactory().smartphone(interpolation_strategy=interpolation_strat), RealWorldDataSourceFactory().hydraulic(interpolation_strategy=interpolation_strat)]
real_world_data_sources =[RealWorldDataSourceFactory().hydraulic()]


blueprints = BlueprintFactory.getBlueprintsForRealWorldData(algorithms=algorithms, dataSources=real_world_data_sources)