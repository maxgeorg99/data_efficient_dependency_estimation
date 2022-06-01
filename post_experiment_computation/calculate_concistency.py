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
algorithms = ["Pearson","Kendalltau","Spearmanr"]
datasources = ["LineDataSource","SquareDataSource","HypersphereDataSource","CrossDataSource","DoubleLinearDataSource","HourglassDataSource","HypercubeDataSource","SineDataSource","StarDataSource","ZDataSource","InvZDataSource"]
num_experiments = 5

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
            ms = load_p_values(algorithm=algo,datasource=source)
            scores.append(calculate_concistency_score(ms))

        #plot?!
        height = scores
        bars = algorithms
        y_pos = np.arange(len(bars))

        # Create bars
        plt.bar(y_pos, height)

        # Create names on the x-axis
        plt.xticks(y_pos, bars)
        plt.yscale("log")
        plt.ylabel("concistency score")
        plt.ylim(top=5*(10**(-15)))
        plt.savefig(f'{result_folder}/{source}_{fig_name}.png')
        plt.clf()

def calculate_concistency_score(ms):
    z = []

    for a, b in itertools.combinations(ms, 2):
        diff = np.array(a) - np.array(b)
        z.append(np.linalg.norm(diff))
    return(stat.mean(z))

#loads p values from all experiment runs
def load_p_values(algorithm, datasource):
    return [algorithm_data[(algorithm, datasource)][i]['score'] for i in range(num_experiments)]

walk_algorithms()

plot_concistency_scores()