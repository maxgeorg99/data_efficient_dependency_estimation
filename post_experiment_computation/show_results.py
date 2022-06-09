from typing import Dict
import numpy as np
from matplotlib import pyplot as plot # type: ignore

result_folder = "./experiment_results/data_efficiency"
log_folder = "./log"
log_prefix = "DataEfficiency_"
algorithms = ["DependencyTestAdapter"]
#algorithms = ["Pearson","Kendalltau","Spearmanr","MCDE","HiCS","CMI"]
#algorithms = ["MCDE","HiCS","CMI"]
#datasources = ["LineDataSource","SquareDataSource","CrossDataSource","DoubleLinearDataSource","HourglassDataSource","HypercubeDataSource"]
datasources = ["LineDataSource","SquareDataSource"]
num_experiments = 1

def walk_algorithms():
    for algorithm in algorithms:
        for datasource in datasources:
            walk_files(algorithm, datasource)

def walk_files( algorithm, datasource):
    for i in range(num_experiments):
        f = f"{log_folder}/{algorithm}_{datasource}_{i}.npz"
        data = np.load(f)
        compute_data_efficiency(data, algorithm, datasource)

algorithm_data: Dict = {}
algorithm_pvalues: Dict = {}
algorithm_score: Dict = {}
algorithm_variance: Dict = {}
def compute_data_efficiency(data, algorithm, datasource):
    algorithm_data[(algorithm, datasource)] = data
    algorithm_pvalues[(algorithm, datasource)] = data['pValues']
    algorithm_score[(algorithm, datasource)] = data['score']
    algorithm_variance[(algorithm, datasource)] = data['var']


def show_results():    
    plot_results(algorithm_pvalues,'p')
    plot_results(algorithm_score,'t')
    plot_results(algorithm_variance,'v')

def plot_results(dict:dict,prefix):
    for key in dict.keys():
        x = range(101)
        plot.plot(x, dict.get(key).tolist())
        plot.savefig(f'{result_folder}/{key[1]}_{prefix}_results.png',dpi=500)

walk_algorithms()

show_results()