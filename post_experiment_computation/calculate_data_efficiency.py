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
algorithms = ["Pearson"]
ignore_nans_count = -1

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

                compute_data_efficiency(data, algorithm)

algorithm_data: Dict = {}
def compute_data_efficiency(data, algorithm):
    runs_data = algorithm_data.get(algorithm, [])
    runs_data.append(data)
    algorithm_data[algorithm] = runs_data


algorithm_data: Dict = {}
def compute_data(data, algorithm):
    runs_data = algorithm_data.get(algorithm, [])
    runs_data.append(data)
    algorithm_data[algorithm] = runs_data


algorithm_means: Dict = {}
def calculate_means():
    for item in algorithm_data.items():
        mean = np.nanmean(item[1], axis=0)
        algorithm_means[item[0]] = mean

algorithm_medians: Dict = {}
def calculate_median():
    for item in algorithm_data.items():
        mean = np.nanmedian(item[1], axis=0)
        algorithm_medians[item[0]] = mean

algorithm_vars: Dict = {}
def calculate_vars():
    for item in algorithm_data.items():
        var = np.nanvar(item[1], axis=0)
        algorithm_vars[item[0]] = var

algorithm_power: Dict = {}
def calculate_power():
    for item in algorithm_data.items():
        effect_size = item[1][0]['score']
        power = TTestIndPower().power(effect_size=effect_size,nobs1=10,alpha=0.05)
        algorithm_power[item[0]] = power

algorithm_F1: Dict = {}
def calculate_F1():
    for item in algorithm_data.items():
        y = item[1][0]['score']
        real_y = get_ground_truth()
        f1 = f1_score(real_y, y)
        algorithm_F1[item[0]] = f1

algorithm_AUC: Dict = {}
def calculate_AUC():
    for item in algorithm_data.items():
        auc =  roc_auc_score(get_ground_truth(), item[1][0]['score'])
        algorithm_AUC[item[0]] = auc

def get_ground_truth():
    return  np.ones(11, dtype = int)

def plot_p_value(fig_name = "p-value"):

    for key in algorithm_data.keys():
        y = algorithm_means[key]
        x = np.asarray([i for i in range(y.shape[0])])
        y_err = algorithm_vars[key]
        plot.ylim(0,1)
        plot.errorbar(x, y, y_err, alpha=0.5, fmt=' ', label=f'{key}_var')
    
    for key in algorithm_data.keys():
        y = algorithm_means[key]
        x = np.asarray([i for i in range(y.shape[0])])
        plot.ylim(0,1)
        plot.plot(x, y, alpha=1, label=f'{key}_mean')

    for key in algorithm_data.keys():
        y = algorithm_medians[key]
        x = np.asarray([i for i in range(y.shape[0])])
        plot.ylim(0,1)
        plot.plot(x, y, alpha=1, label=f'{key}_median')

    plot.title(fig_name)
    plot.xlabel("learning iteration")
    plot.ylabel("p-value")
    plot.figlegend()
    
    plot.savefig(f'{result_folder}/{fig_name}.png',dpi=500)
    plot.clf()

def plot_data_efficiency(fig_name = "data efficiency"):

    for key in algorithm_data.keys():
        y = algorithm_means[key]
        x = np.asarray([i for i in range(y.shape[0])])
        y_err = algorithm_F1[key]
        plot.ylim(0,1)
        plot.errorbar(x, y, y_err, alpha=0.5, fmt=' ', label=f'{key}_f1')
    
    for key in algorithm_data.keys():
        y = algorithm_AUC[key]
        x = np.asarray([i for i in range(y.shape[0])])
        plot.ylim(0,1)
        plot.plot(x, y, alpha=1, label=f'{key}_AUC')

    for key in algorithm_data.keys():
        y = algorithm_power[key]
        x = np.asarray([i for i in range(y.shape[0])])
        plot.ylim(0,1)
        plot.plot(x, y, alpha=1, label=f'{key}_power')

    plot.title(fig_name)
    plot.xlabel("learning iteration")
    plot.ylabel("score")
    plot.figlegend()
    
    plot.savefig(f'{result_folder}/{fig_name}.png',dpi=500)
    plot.clf()

walk_algorithms()

calculate_power()
calculate_F1()
calculate_AUC()
calculate_means()
calculate_median()
calculate_vars()

plot_p_value()
plot_data_efficiency()