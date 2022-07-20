import numpy as np
import pandas as pd
from ide.core.oracle.interpolation_strategy import InterpolationStrategy
from ide.modules.data_sampler import KDTreeKNNDataSampler
from ide.modules.oracle.data_source import InterpolatingDataSource, RealWorldDataSetDataSource
from ide.modules.oracle.data_source_adapter import DataSourceAdapter
import wfdb
import tensorflow as tf

from ide.modules.oracle.interpolation_strategy.interpolation_strategy import AverageInterpolationStrategy
from ide.modules.queried_data_pool import FlatQueriedDataPool

class RealWorldDataSourceFactory():

    def chf(self,interpolation_strategy:InterpolationStrategy = AverageInterpolationStrategy()):
        record = wfdb.rdrecord( wfdb.get_record_list(db_dir='chfdb')[0],pn_dir='chfdb')
        p_signal = record.p_signal
        q = p_signal[:,0]
        r = p_signal[:,1]
        data_source=InterpolatingDataSource(
            data= dict(zip(q, r)),
            interpolation_strategy=interpolation_strategy,
            data_set='CHF_'+ type(interpolation_strategy).__name__,
        )
        return data_source

    def office(self,interpolation_strategy:InterpolationStrategy = AverageInterpolationStrategy()):
        data_file = 'C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/Office/data.txt'

        df = pd.read_csv(data_file, sep=" ", header=None)
        df = df[~df.isnull().any(axis=1)]
        q = df.index.to_numpy()
        r = df.iloc[:,2:]
        r = r.to_numpy()
        r = r[np.random.choice(r.shape[0], 10000, replace=False)]
        data_source=InterpolatingDataSource(
            data= dict(zip(q, r)),
            interpolation_strategy=interpolation_strategy,
            data_set='office'+ type(interpolation_strategy).__name__,
        )
        return data_source

    def sunspot(self,interpolation_strategy:InterpolationStrategy = AverageInterpolationStrategy()):
        filename = 'C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/Sunspot/SN_d_tot_V2.0.txt'
        df = pd.read_csv(filename, sep=" ", header=None)
        q = df.index.to_numpy()
        r = df.to_numpy()
        data_source=InterpolatingDataSource(
            data= dict(zip(q, r)),
            interpolation_strategy=interpolation_strategy,
            data_set='sunspot'+ type(interpolation_strategy).__name__,
        )
        return data_source
    
    def personal_activity(self,interpolation_strategy:InterpolationStrategy = AverageInterpolationStrategy()):
        return NotImplemented
    
    def NASDAQ(self,interpolation_strategy:InterpolationStrategy = AverageInterpolationStrategy()):
        filename = 'C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/NASDAQ/WIKI-PRICES.csv'
        df = pd.read_csv(filename)
        q = df.index.to_numpy()
        r = df.iloc[:,2:]
        r = r.to_numpy()

        data_source=InterpolatingDataSource(
            data= dict(zip(q, r)),
            interpolation_strategy=interpolation_strategy,
            data_set='NASDAQ'+ type(interpolation_strategy).__name__,
        )
        return data_source
    
    def smartphone(self,interpolation_strategy:InterpolationStrategy = AverageInterpolationStrategy()):
        df = pd.read_csv('C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/Smartphone/final_X_test.txt')
        df = df[~df.isnull().any(axis=1)]
        df2 = pd.read_csv('C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/Smartphone/final_Y_test.txt', sep=",", header=None)
        df['y'] = df2
        q = df.index.to_numpy()
        r = df.to_numpy()
        r = r.astype(np.float32)
        data_source=InterpolatingDataSource(
            data= dict(zip(q, r)),
            interpolation_strategy=interpolation_strategy,
            data_set='smartphone'+ type(interpolation_strategy).__name__,
        )
        return data_source

    def hipe(self,interpolation_strategy:InterpolationStrategy = AverageInterpolationStrategy()):
        filename1 = 'C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/HIPE/PickAndPlaceUnit_PhaseCount_2_geq_2017-10-01_lt_2018-01-01.csv'
        filename2 = 'C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/HIPE/ScreenPrinter_PhaseCount_2_geq_2017-10-01_lt_2018-01-01.csv'
        filename3 = 'C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/HIPE/VacuumPump2_PhaseCount_2_geq_2017-10-01_lt_2018-01-01.csv'
        df = pd.read_csv(filename1)
        df2 = pd.read_csv(filename2)
        df3 = pd.read_csv(filename3)
        df.append(df2, ignore_index=True)
        df.append(df3, ignore_index=True)
        q = df.index.to_numpy()
        r = df.iloc[:,2:]
        r = r.to_numpy()
        r = r[np.random.choice(r.shape[0], 10000, replace=False)]
        data_source=InterpolatingDataSource(
            data= dict(zip(q, r)),
            interpolation_strategy=interpolation_strategy,
            data_set='hipe'+ type(interpolation_strategy).__name__,
        )
        return data_source

    def hydraulic(self,interpolation_strategy:InterpolationStrategy = AverageInterpolationStrategy()):
        df = pd.read_csv('C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/HydraulicSystemsDataSet/PS1.txt', sep=" ", header=None)
        df2 = pd.read_csv('C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/HydraulicSystemsDataSet/PS2.txt', sep=" ", header=None)
        df3 = pd.read_csv('C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/HydraulicSystemsDataSet/PS3.txt', sep=" ", header=None)
        df4 = pd.read_csv('C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/HydraulicSystemsDataSet/PS4.txt', sep=" ", header=None)
        df5 = pd.read_csv('C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/HydraulicSystemsDataSet/PS4.txt', sep=" ", header=None)
        df6 = pd.read_csv('C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/HydraulicSystemsDataSet/PS4.txt', sep=" ", header=None)
        df.append(df2, ignore_index=True)
        df.append(df3, ignore_index=True)
        df.append(df4, ignore_index=True)
        df.append(df5, ignore_index=True)
        df.append(df6, ignore_index=True)
        q = df.index.to_numpy()
        r = df.to_numpy()
        data_source=InterpolatingDataSource(
            data= dict(zip(q, r)),
            interpolation_strategy=interpolation_strategy,
            data_set='hydraulic'+ type(interpolation_strategy).__name__,
        )
        return data_source


    def get_data_source(self, realWorldDataSource):
        
        switcher = {
            'chf': self.chf,
            'office': self.office,
            'sunspot': self.sunspot,
            'personalActivity': self.personal_activity,
            'NASDAQ': self.NASDAQ,
            'hipe': self.hipe,
            'smartphone': self.smartphone,
            'hydraulic': self.hydraulic,
        }

        r = switcher.get(realWorldDataSource, lambda: "Invalid data source")

        return r()

    def get_all_data_sources(self):
        return [self.chf(),self.office(), self.sunspot(), self.NASDAQ(), self.hipe(), self.smartphone(), self.hydraulic()]
