import os
from typing import Dict
import numpy as np
from matplotlib import pyplot as plot # type: ignore

result_folder = "./experiment_results/data_efficiency"
log_folder = "./log"
log_prefix = "DataEfficiency_"

num_experiments = 1

def walk_files(path):
    for dirpath, dnames, fnames in os.walk(path):
        f: str
        for f in fnames:
            if f.endswith(".npz"):
                data = np.load(os.path.join(dirpath, f))
                c = f.split("_")
                algorithm = c[0]
                datasource = c[1]
                compute_data_efficiency(data,algorithm,datasource)

algorithm_pvalues: Dict = {}
algorithm_score: Dict = {}
algorithm_variance: Dict = {}
def compute_data_efficiency(data, algorithm, datasource):
    algorithm_pvalues[(algorithm, datasource)] = data['pValues']
    algorithm_score[(algorithm, datasource)] = data['score']
    algorithm_variance[(algorithm, datasource)] = data['var']


def show_results():    
    plot_results(algorithm_pvalues,'p')
    plot_results(algorithm_score,'t')
    plot_results(algorithm_variance,'v')

def plot_results(dict:dict,prefix):
    for key in dict.keys():
        x = range(len(dict.get(key).tolist()))
        plot.plot(x, dict.get(key).tolist())
        plot.savefig(f'{result_folder}/{key[0]}_{key[1]}_{prefix}_results.png',dpi=500)
        plot.clf()

walk_files(log_folder)

show_results()