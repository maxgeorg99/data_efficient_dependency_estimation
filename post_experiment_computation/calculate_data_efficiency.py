from random import random
from typing import Dict, List
from matplotlib import pyplot as plot # type: ignore
import numpy as np
import os
from requests import get
from statsmodels.stats.power import TTestIndPower
from sklearn.metrics import f1_score, roc_auc_score

result_folder = "./experiment_results/data_efficiency"
log_folder = "./log"
log_prefix = "DataEfficiency_"
algorithms = ["Pearson","Kendalltau","Spearmanr","XiCor"]
datasources = ["LineDataSource","SquareDataSource","HyperSphereDataSource"]

ignore_nans_count = -1



def walk_algorithms():
    for algorithm in algorithms:
        for datasource in datasources:
            walk_files(algorithm, datasource)

def walk_files( algorithm, datasource):
        f = f"{log_folder}/{algorithm}_{datasource}.npz"
        data = np.load(f)
        compute_data_efficiency(data, algorithm, datasource)

algorithm_data: Dict = {}
algorithm_pvalues: Dict = {}
algorithm_score: Dict = {}
def compute_data_efficiency(data, algorithm, datasource):
    algorithm_data[(algorithm, datasource)] = data
    algorithm_pvalues[(algorithm, datasource)] = data['pValues']
    algorithm_score[(algorithm, datasource)] = data['score']


algorithm_power90: Dict = {}
algorithm_power95: Dict = {}
algorithm_power99: Dict = {}
def calculate_power():
    for item in algorithm_data.items():
        results = item[1]['score']
        t90 = percentile_scala_breeze(results, 0.90)
        t95 = percentile_scala_breeze(results, 0.95)
        t99 = percentile_scala_breeze(results, 0.99)
        algorithm_power90[item[0]] = sum([r > t90 for r in results]) / len(results)
        algorithm_power95[item[0]] = sum([r > t95 for r in results]) / len(results)
        algorithm_power99[item[0]] = sum([r > t99 for r in results]) / len(results)
        

algorithm_F1: Dict = {}
def calculate_F1():
    for item in algorithm_data.items():
        y = item[1][0]['predictions']
        real_y = get_ground_truth(y)
        f1 = f1_score(real_y, y)
        algorithm_F1[item[0]] = f1

algorithm_AUC: Dict = {}
def calculate_AUC():
    for item in algorithm_data.items():
        y = item[1][0]['score']
        #handle the every class is the same case?!
        #auc = roc_auc_score(get_ground_truth(y), y)
        auc = random()
        algorithm_AUC[item[0]] = auc

def get_ground_truth(y):
    return  np.ones(len(y), dtype = int)

def plot_data_efficiency(fig_name = "data_efficiency"):
    y_pos = list(algorithms)

    for datasource in datasources:
        ys = []
        for algorithm in algorithms:
            ys.append(algorithm_F1[(algorithm, datasource)])
        plot.bar(y_pos, ys)
        plot.ylabel("F1")
        plot.savefig(f'{result_folder}/F1/{datasource}_{fig_name}_F1.png',dpi=500)
        plot.clf()

        negative_data = []
        positive_data = []
        for algorithm in algorithms:
            value = algorithm_AUC[(algorithm,datasource)]-0.5
            if (value > 0):
                positive_data.append(value)
                negative_data.append(0)
            else:
                positive_data.append(0)
                negative_data.append(value)

        ax = plot.subplot(111)
        ax.bar(y_pos, negative_data, color='r', bottom=0.5)
        ax.bar(y_pos, positive_data, color='b', bottom=0.5)
        plot.ylim(0,1)
        plot.ylabel("AUC")
        plot.savefig(f'{result_folder}/AUC/{datasource}_{fig_name}_AUC.png',dpi=500)
        plot.clf()

        ys = []
        for algorithm in algorithms:
            ys.append(algorithm_power99[(algorithm,datasource)])
        plot.bar(y_pos, ys)
        plot.ylabel("Power99")
        plot.savefig(f'{result_folder}/Power/{datasource}_{fig_name}_Power99.png',dpi=500)
        plot.clf()

def percentile_scala_breeze(list_of_floats: List[float], p: float):
        arr = sorted(list_of_floats)
        f = (len(arr) + 1) * p
        i = int(f)
        if i == 0:
            return arr[0]
        elif i >= len(arr):
            return arr[-1]
        else:
            return arr[i - 1] + (f - i) * (arr[i] - arr[i - 1])

walk_algorithms()

calculate_power()
calculate_F1()
calculate_AUC()

plot_data_efficiency()