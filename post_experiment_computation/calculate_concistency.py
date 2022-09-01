from ast import If
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
log_folder = "./log"
log_prefix = "DataEfficiency_Ps_"
num_experiments = 1
num_iterations = 100

def walk_files(path):
    for dirpath, dnames, fnames in os.walk(path):
        f: str
        for f in fnames:
            if f.endswith(".npz"):
                c = f.split("_")
                algorithm = c[0]
                datasource = ' '.join(c[1:4])
                with np.load(os.path.join(dirpath, f)) as data:
                    compute_data_efficiency(data,algorithm,datasource)

algorithm_p: Dict = {}
algorithm_v: Dict = {}
def compute_data_efficiency(data, algorithm, datasource):
    runs_data_p = algorithm_p.get((algorithm,datasource), [])
    runs_data_p.append(data['pValues'])        
    algorithm_p[(algorithm, datasource)] = runs_data_p

    runs_data_v = algorithm_p.get((algorithm,datasource), [])
    runs_data_v.append(data['var'])
    algorithm_v[(algorithm, datasource)] = runs_data_v

def plot_concistency_scores():
    scores: dict
    scores = {}
    keys = list(algorithm_p.keys())
    for source in set([x[1] for x in keys]):
        for algo in set([x[0] for x in keys]):
            ps = algorithm_p[(algo,source)]
            variances = algorithm_v[(algo,source)]
            scores_iteration = []
            for i in range(num_iterations):
                var = np.var([ps[j][i] for j in range(num_experiments)])
                predticted_var = np.mean([variances[j][i] for j in range(num_experiments)])
                score = predticted_var - var
                scores_iteration.append(score) 
            scores[(algo,source)] = scores_iteration
        for k, v in scores.items():
            if k[1] == source:
                plt.plot(range(0, len(v)), v, algo_marker_dict[k[0]] + '-', label=k[0])
        plt.ylabel("consistency score")
        plt.ylim(0,1)
        plt.xlabel("iteration")
        plt.title(source)
        plt.figlegend(bbox_to_anchor=(1, 1))
        class_string = os.path.dirname(log_folder).split('/')[-1]
        plt.savefig(f'{result_folder}/{source}_{fig_name}_{class_string}.png',dpi=500)
        plt.clf()

walk_files(log_folder)

keys = list(algorithm_p.keys())
datasources = set([x[1] for x in keys])
algorithms = sorted(set([x[0] for x in keys]))

markers = ['.','v','^','<','>','s','P','*','+','x','D','d','|']
algo_marker_dict = dict(zip(algorithms,markers))

plot_concistency_scores()