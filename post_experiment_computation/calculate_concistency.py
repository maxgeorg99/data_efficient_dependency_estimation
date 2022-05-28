import ast
import imp
import itertools
import os
import readline
from matplotlib import pyplot as plt
import numpy as np
import statistics as stat

result_folder = "./experiment_results/concistency"
fig_name = 'concistency'
log_folder = "./log"
log_prefix = "DataEfficiency_Ps_"
algorithms = ["Pearson","Spearmanr","Kendalltau"]
datasources = ["ZDataSource","HourglassDataSource","CrossDataSource","DoubleLinearDataSource","InvZDataSource","StarDataSource","SineDataSource","HyperCubeDataSource"]

def walk_algorithms():
    for algorithm in algorithms:
        path = f"{log_folder}/{algorithm}"
        walk_files(path, algorithm)

def walk_files(path, algorithm):
    for dirpath, dnames, fnames in os.walk(path):
        f: str
        for f in fnames:
            if f.endswith(".npz"):
                data = np.load(os.path.join(dirpath, f))

def plot_concistency_scores():
    for source in datasources:
        scores = []
        for algo in algorithms:
            ms = load_p_values(algorithm=algo,dataset=source)
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

        plt.savefig(f'{result_folder}/{source}_{fig_name}.png',dpi=500)
        plt.clf()

def calculate_concistency_score(ms):
    z = []
    q = []

    for a, b in itertools.combinations(ms, 2):
        diff = np.array(a) - np.array(b)
        z.append(np.linalg.norm(diff))
    return(stat.mean(z))

#loads p values from all experiment runs
def load_p_values(algorithm, dataset):
    pvalues = []
    
    for i in range(10):
        f = open(f"{log_folder}/{log_prefix}{algorithm}_{dataset}_{i}.txt", "r") 
        lines = f.read().splitlines()
        subpvalues = []
        for i in range(10):
            subpvalues.append(ast.literal_eval(lines[i]))
        pvalues.append(subpvalues)
    return pvalues


#for each algo : dataset combination
#walk_algorithms()

plot_concistency_scores()