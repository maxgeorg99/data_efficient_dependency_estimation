import pandas as pd
from ide.modules.oracle.data_source import RealWorldDataSetDataSource
from ide.modules.oracle.data_source_adapter import DataSourceAdapter
import wfdb
import tensorflow as tf

class RealWorldDataSourceFactory():
    def chf(self):
        record = wfdb.rdrecord( wfdb.get_record_list(db_dir='chfdb')[0],pn_dir='chfdb')
        p_signal = record.p_signal
        q = p_signal[:,0]
        r = p_signal[:,1]
        return RealWorldDataSetDataSource(q,r)

    def office(self):
        connectivity_file = 'C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/Office/connectivity.txt'
        data_file = 'C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/Office/data.txt'
        locs_file = 'C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/Office/mote_locs.txt'

        df = pd.read_csv(connectivity_file, sep=" ", header=None)
        data_df = pd.read_csv(data_file, sep=" ", header=None)
        locs_df = pd.read_csv(locs_file, sep=" ", header=None)
        q = list(range(1, len(df.index)))
        return RealWorldDataSetDataSource(q,df.to_numpy())

    def sunspot(self):
        filename = 'C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/Sunspot/SN_d_tot_V2.0.txt'
        df = pd.read_csv(filename, sep=" ", header=None)
        q = list(range(1, len(df.index)))
        return RealWorldDataSetDataSource(q,df.to_numpy())
    
    def personal_activity(self):
        return NotImplemented
    
    def NASDAQ(self):
        filename = 'C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/NASDAQ/WIKI-PRICES.csv'
        df = pd.read_csv(filename)
        q = list(range(1, len(df.index)))
        return RealWorldDataSetDataSource(q,df.to_numpy())
    
    def smartphone(self):
        df = pd.read_csv('C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/Smartphone/final_acc_test.txt', sep=" ", header=None)
        df2 = pd.read_csv('C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/Smartphone/final_gyro_test.txt', sep=" ", header=None)
        df3 = pd.read_csv('C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/Smartphone/final_X_test.txt', sep=" ", header=None)
        df4 = pd.read_csv('C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/Smartphone/final_Y_test.txt', sep=" ", header=None)
        df.merge(df2,on = df.columns.values.tolist(), how = 'right')
        df.merge(df3,on = df.columns.values.tolist(), how = 'right')
        df.merge(df4,on = df.columns.values.tolist(), how = 'right')
        q = list(range(1, len(df.index)))
        return RealWorldDataSetDataSource(q,df.to_numpy())

    def hipe(self):
        filename1 = 'C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/HIPE/PickAndPlaceUnit_PhaseCount_2_geq_2017-10-01_lt_2018-01-01.csv'
        filename2 = 'C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/HIPE/ScreenPrinter_PhaseCount_2_geq_2017-10-01_lt_2018-01-01.csv'
        filename3 = 'C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/HIPE/VacuumPump2_PhaseCount_2_geq_2017-10-01_lt_2018-01-01.csv'
        df = pd.read_csv(filename1)
        df2 = pd.read_csv(filename2)
        df3 = pd.read_csv(filename3)
        df.merge(df2,on = df.columns.values.tolist(), how = 'right')
        df.merge(df3,on = df.columns.values.tolist(), how = 'right')
        q = list(range(1, len(df.index)))
        return RealWorldDataSetDataSource(q,df.to_numpy())

    def hydraulic(self):
        df = pd.read_csv('C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/HydraulicSystemsDataSet/PS1', sep=" ", header=None)
        df2 = pd.read_csv('C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/HydraulicSystemsDataSet/PS2', sep=" ", header=None)
        df3 = pd.read_csv('C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/HydraulicSystemsDataSet/PS3', sep=" ", header=None)
        df4 = pd.read_csv('C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/real_world_data_sets/HydraulicSystemsDataSet/PS4', sep=" ", header=None)
        df.merge(df2,on = df.columns.values.tolist(), how = 'right')
        df.merge(df3,on = df.columns.values.tolist(), how = 'right')
        df.merge(df4,on = df.columns.values.tolist(), how = 'right')
        q = list(range(1, len(df.index)))
        return RealWorldDataSetDataSource(q,df.to_numpy())


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
        #return [self.chf(),self.office(), self.sunspot(), self.NASDAQ(), self.hipe(), self.smartphone(), self.hydraulic()]
        return [self.chf(),self.office(), self.NASDAQ(), self.hipe(), self.hydraulic()]
