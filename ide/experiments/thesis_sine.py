from ide.building_blocks.dependency_test import NaivDependencyTest
from ide.building_blocks.multi_sample_test import KWHMultiSampleTest
from ide.core.blueprint_factory import BlueprintFactory
from ide.modules.data_sampler import KDTreeRegionDataSampler
from ide.modules.oracle.data_source import LineDataSource, SineDataSource

from ide.modules.query.query_sampler import LatinHypercubeQuerySampler, RandomChoiceQuerySampler

algos = [NaivDependencyTest(
            query_sampler = RandomChoiceQuerySampler(num_queries=20),
            data_sampler = KDTreeRegionDataSampler(0.05),
            multi_sample_test=KWHMultiSampleTest())]
synthetic_data_sources = []
synthetic_data_sources.append(SineDataSource((1,),(2,),amplitude=1,period=3))

blueprints = BlueprintFactory.getBlueprintsForSyntheticData(algorithms=algos ,dataSources=synthetic_data_sources, noiseRatio=4.0)