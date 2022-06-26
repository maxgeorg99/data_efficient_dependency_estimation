from cmath import exp
import itertools
from operator import le
import os
from typing import Dict
from matplotlib import pyplot as plt
import numpy as np
import statistics as stat

result_folder = "./experiment_results/concistency"
fig_name = 'concistency'
log_folder = "./log_noise_Square"
log_prefix = "DataEfficiency_Ps_"
num_experiments = 100
num_iterations = 100

def walk_files(path):
    for dirpath, dnames, fnames in os.walk(path):
        f: str
        for f in fnames:
            if f.endswith(".npz"):
                data = np.load(os.path.join(dirpath, f))
                c = f.split("_")
                algorithm = c[0] + ' noise ' +c[2]
                datasource = c[3]
                compute_data_efficiency(data,algorithm,datasource)

algorithm_data: Dict = {}
def compute_data_efficiency(data, algorithm, datasource):
    runs_data = algorithm_data.get((algorithm,datasource), [])
    runs_data.append(data)
    algorithm_data[(algorithm, datasource)] = runs_data

def plot_concistency_scores():
    scores: dict
    scores = {}
    keys = list(algorithm_data.keys())
    for source in set([x[1] for x in keys]):
        for algo in set([x[0] for x in keys]):
            ps = load_p_values(algorithm=algo,datasource=source)
            variances = load_var_values(algo,source)
            scores_iteration = []
            for i in range(num_iterations):
                var = np.var([ps[j][i] for j in range(num_experiments)])
                predticted_var = np.mean([variances[j][i] for j in range(num_experiments)])
                scores_iteration.append(predticted_var - var) 
            scores[(algo,source)]= scores_iteration  
        for k, v in scores.items():
            if k[1] == source:
                plt.plot(range(0, len(v)), v, '.-', label=k[0])
        plt.ylabel("concistency score")
        plt.xlabel("iteration")
        plt.xticks(fontsize=5)
        plt.title(source)
        plt.figlegend()
        plt.savefig(f'{result_folder}/{source}_{fig_name}_square_noise_100.png',dpi=1000)
        plt.clf()

def load_p_values(algorithm, datasource):
    return [algorithm_data[(algorithm, datasource)][i]['pValues'] for i in range(num_experiments)]

def load_var_values(algorithm, datasource):
    return [algorithm_data[(algorithm, datasource)][i]['var'] for i in range(num_experiments)]

walk_files(log_folder)

plot_concistency_scores()