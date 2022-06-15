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
algorithms = ["Pearson","Kendalltau","Spearmanr","dCor","dHSIC","FIT","Hoeffdings","XiCor"]
datasources = ["LineDataSource1x1","SquareDataSource1x1","LineDataSource1x2","SquareDataSource1x2"]
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
def compute_data_efficiency(data, algorithm, datasource):
    runs_data = algorithm_data.get((algorithm,datasource), [])
    runs_data.append(data)
    algorithm_data[(algorithm, datasource)] = runs_data

def plot_concistency_scores():
    for source in datasources:
        scores = []
        for algo in algorithms:
            ps = load_p_values(algorithm=algo,datasource=source)
            variances = load_var_values(algo,source)
            scores.append(calculate_concistency_score(ps,variances))

        #plot?!
        height = scores
        bars = algorithms
        y_pos = np.arange(len(bars))

        # Create bars
        plt.bar(y_pos, height)

        # Create names on the x-axis
        plt.xticks(y_pos, bars)
        plt.ylabel("concistency score")
        plt.xticks(fontsize=7)
        plt.title(source)
        plt.savefig(f'{result_folder}/{source}_{fig_name}.png')
        plt.clf()

def calculate_concistency_score(ps,vs):
    z = []
    var_ps = np.var(ps)
    for v in vs:
        z.append(v - var_ps)
    return np.mean(z)

def load_p_values(algorithm, datasource):
    return [algorithm_data[(algorithm, datasource)][i]['pValues'] for i in range(num_experiments)]

def load_var_values(algorithm, datasource):
    return [algorithm_data[(algorithm, datasource)][i]['var'] for i in range(num_experiments)]

walk_algorithms()

plot_concistency_scores()