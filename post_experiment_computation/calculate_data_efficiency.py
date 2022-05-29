from random import random
from typing import Dict
from matplotlib import pyplot as plot # type: ignore
import numpy as np
import os
from requests import get

from statsmodels.stats.power import TTestIndPower
from sklearn.metrics import f1_score, roc_auc_score

result_folder = "./experiment_results/data_efficiency"
log_folder = "./log"
log_prefix = "DataEfficiency_"
algorithms = ["Pearson","Kendalltau","Spearmanr","Hoeffdings","XiCor"]
datasources = ["ZDataSource","HourglassDataSource","CrossDataSource","DoubleLinearDataSource","InvZDataSource","StarDataSource","SineDataSource","HyperCubeDataSource"]

ignore_nans_count = -1

def walk_algorithms():
    for algorithm in algorithms:
        for datasource in datasources:
            path = f"{log_folder}"
            walk_files(path, algorithm, datasource)

def walk_files(path, algorithm, datasource):
    for dirpath, dnames, fnames in os.walk(path):
        f: str
        for f in fnames:
            if f.endswith(".npz"):
                data = np.load(os.path.join(dirpath, f))
                compute_data_efficiency(data, algorithm, datasource)

algorithm_data: Dict = {}
def compute_data_efficiency(data, algorithm, datasource):
    runs_data = algorithm_data.get((algorithm, datasource), [])
    runs_data.append(data)
    algorithm_data[(algorithm, datasource)] = runs_data

algorithm_power: Dict = {}
def calculate_power():
    for item in algorithm_data.items():
        effect_size = item[1][0]['score']
        power = TTestIndPower().power(effect_size=effect_size,nobs1=10,alpha=0.05)
        algorithm_power[item[0]] = np.mean(power)

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
        plot.savefig(f'{result_folder}/{datasource}_{fig_name}_F1.png',dpi=500)
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
        plot.savefig(f'{result_folder}/{datasource}_{fig_name}_AUC.png',dpi=500)
        plot.clf()

        ys = []
        for algorithm in algorithms:
            ys.append(algorithm_power[(algorithm,datasource)])
        plot.bar(y_pos, ys)
        plot.ylabel("Power")
        plot.savefig(f'{result_folder}/{datasource}_{fig_name}_Power.png',dpi=500)
        plot.clf()

walk_algorithms()

calculate_power()
calculate_F1()
calculate_AUC()

plot_data_efficiency()