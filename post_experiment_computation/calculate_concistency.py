from cmath import exp
import itertools
from operator import le
from typing import Dict
from matplotlib import pyplot as plt
import numpy as np
import statistics as stat

result_folder = "./experiment_results/concistency"
fig_name = 'concistency'
log_folder = "./log"
log_prefix = "DataEfficiency_Ps_"
#algorithms = ["Pearson","Kendalltau","Spearmanr","dCor","dHSIC","FIT","Hoeffdings","XiCor","CondIndTest","LISTest","IndepTest"]
algorithms = ["NaivDependencyTest"]
datasources = ["LineDataSource1x1","SquareDataSource1x1","LineDataSource1x2","SquareDataSource1x2"]
num_experiments = 5
num_iterations = 50

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
def compute_data_efficiency(data, algorithm, datasource):
    runs_data = algorithm_data.get((algorithm,datasource), [])
    runs_data.append(data)
    algorithm_data[(algorithm, datasource)] = runs_data

def plot_concistency_scores():
    scores: dict
    scores = {}
    for source in datasources:
        for algo in algorithms:
            ps = load_p_values(algorithm=algo,datasource=source)
            variances = load_var_values(algo,source)
            scores_iteration = []
            for i in range(num_iterations):
                var = np.var([ps[j][i] for j in range(num_experiments)])
                predticted_var = np.mean([variances[j][i] for j in range(num_experiments)])
                scores_iteration.append(predticted_var - var) 
            scores[(algo,source)]= scores_iteration  
        for k, v in scores.items():
            plt.plot(range(0, len(v)), v, '.-', label=k)
        plt.ylabel("concistency score")
        plt.xlabel("iteration")
        plt.xticks(fontsize=5)
        plt.title(source)
        plt.figlegend()
        plt.savefig(f'{result_folder}/{source}_{fig_name}.png',dpi=1000)
        plt.clf()

def load_p_values(algorithm, datasource):
    return [algorithm_data[(algorithm, datasource)][i]['pValues'] for i in range(num_experiments)]

def load_var_values(algorithm, datasource):
    return [algorithm_data[(algorithm, datasource)][i]['var'] for i in range(num_experiments)]

walk_algorithms()

plot_concistency_scores()