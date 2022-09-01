from random import randint, random

from ide.modules.oracle.data_source import CrossDataSource, HourglassDataSource, LineDataSource
from ide.building_blocks.dependency_test import Kendalltau, Pearson, Spearmanr

from ide.core.blueprint_factory import BlueprintFactory

algorithms = [Pearson(),Spearmanr(),Kendalltau()]
synthetic_data_sources = []
noise = 0.5

synthetic_data_sources.append(LineDataSource((1,),(3,),a=randint(-100,100),b=randint(-100,100))), 
synthetic_data_sources.append(CrossDataSource((1,),(3,),a=100*random())),

blueprints=BlueprintFactory.getBlueprintsForSyntheticData(algorithms=algorithms,dataSources=synthetic_data_sources,noiseRatio=noise)