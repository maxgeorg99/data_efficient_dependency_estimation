from random import random
from typing import Dict, List
from matplotlib import pyplot as plot # type: ignore
import numpy as np
import os
from requests import get
from statsmodels.stats.power import TTestIndPower
from sklearn.metrics import auc, average_precision_score, f1_score, roc_auc_score
import statsmodels.stats.power as smp

result_folder = "./experiment_results/data_efficiency"
log_folder = "./log_noise_Square"
log_prefix = "DataEfficiency_"
num_experiments = 100
ignore_nans_count = -1
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
algorithm_pvalues: Dict = {}
algorithm_score: Dict = {}
algorithm_variance: Dict = {}
def compute_data_efficiency(data, algorithm, datasource):
    algorithm_data[(algorithm, datasource)] = data
    algorithm_pvalues[(algorithm, datasource)] = data['pValues']
    algorithm_score[(algorithm, datasource)] = data['score']
    algorithm_variance[(algorithm, datasource)] = data['var']


algorithm_power90: Dict = {}
algorithm_power95: Dict = {}
algorithm_power99: Dict = {}
def calculate_power():
    for item in algorithm_data.items():
        #TTestIndPower.power()
        results = item[1]['score']
        t90 = [smp.ttest_power(r, nobs=list(results).index(r), alpha=0.10, alternative='larger') for r in results]
        t95 = [smp.ttest_power(r, nobs=list(results).index(r), alpha=0.05, alternative='larger') for r in results]
        t99 = [smp.ttest_power(r, nobs=list(results).index(r), alpha=0.01, alternative='larger') for r in results]
        algorithm_power90[item[0]] = t90
        algorithm_power95[item[0]] = t95
        algorithm_power99[item[0]] = t99
        

algorithm_F1: Dict = {}
def calculate_F1():
    p_thresholds = [x / 100.0 for x in range(0, 26, 5)]
    for algorithm in algorithms:
        for i in range(num_iterations):
            f1_for_p = []
            for p in p_thresholds:
                predictions = [1 if algorithm_data[(algorithm, ds)]['pValues'][i] >= p else 0 for ds in datasources]
                f1 = f1_score(get_ground_truth(datasources), predictions)
                f1_for_p.append(f1)
            algorithm_F1[(algorithm,i)] = f1_for_p

algorithm_ROC_AUC: Dict = {}
def calculate_ROC_AUC():
    p_thresholds = [x / 100.0 for x in range(0, 26, 5)]
    for algorithm in algorithms:
        for i in range(num_iterations):
            auc_score_for_p = []
            for p in p_thresholds:
                predictions = [1 if algorithm_data[(algorithm, ds)]['pValues'][i] >= p else 0 for ds in datasources]
                roc_auc = roc_auc_score(get_ground_truth(datasources),predictions)
                auc_score_for_p.append(roc_auc)
            algorithm_ROC_AUC[(algorithm,i)] = auc_score_for_p

algorithm_prediction_recall_AUC: Dict = {}
def calculate_prediction_recall_AUC():    
    p_thresholds = [x / 100.0 for x in range(0, 26, 5)]
    for algorithm in algorithms:
        for i in range(num_iterations):
            auc_score_for_p = []
            for p in p_thresholds:
                predictions = [1 if algorithm_data[(algorithm, ds)]['pValues'][i] >= p else 0 for ds in datasources]
                p_r_auc = average_precision_score(get_ground_truth(datasources),predictions)
                auc_score_for_p.append(p_r_auc)
            algorithm_prediction_recall_AUC[(algorithm,i)] = auc_score_for_p

def get_ground_truth(y):
    return np.asarray([1 if x.startswith('Independent') else calc_ground_truth() if x.startswith('Real') else 0 for x in y])

def calc_ground_truth():
    return [algorithm_data.get(('MCDE',datasource))['predictions'][-1:] for datasource in datasources]

def plot_F1():
    x = np.arange(num_iterations)
    p_thresholds = [x / 100.0 for x in range(0, 10, 1)]
    for p in range(len(p_thresholds)):
        for algorithm in algorithms:
            y = [algorithm_F1[algorithm,i][p] for i in range(num_iterations)]
            plot.plot(x,y, '.-', label=algorithm)
        plot.ylabel("F1 score")
        plot.xlabel("iteration")
        plot.figlegend()
        plot.savefig(f'{result_folder}/F1/F1_{p_thresholds[p]}_2.png',dpi=500)
        plot.clf()

def plot_AUC():
    x = np.arange(num_iterations)
    p_thresholds = [x / 100.0 for x in range(0, 10, 1)]
    for p in range(len(p_thresholds)):
        for algorithm in algorithms:
            y = [algorithm_ROC_AUC[algorithm,i][p] for i in range(num_iterations)]
            plot.plot(x,y, '.-', label=algorithm)
        plot.ylabel("ROC_AUC score")
        plot.xlabel("iteration")
        plot.figlegend()
        plot.savefig(f'{result_folder}/ROC_AUC/ROC_AUC_{p_thresholds[p]}_2.png',dpi=500)
        plot.clf()

        for algorithm in algorithms:
            y = [algorithm_prediction_recall_AUC[algorithm,i][p] for i in range(num_iterations)]
            plot.plot(x,y, '.-', label=algorithm)
        plot.ylabel("Precision Recall AUC")
        plot.xlabel("iteration")
        plot.figlegend()
        plot.savefig(f'{result_folder}/Precision_Recall_AUC/AUC_{p_thresholds[p]}_2.png',dpi=500)
        plot.clf()

def plot_power():
    x = np.arange(num_iterations)
    for datasource in datasources:
        for algorithm in algorithms:
            p = algorithm_power90[(algorithm,datasource)]
            p.pop(0)
            y = np.nan_to_num(p)
            plot.plot(x,y, '.-', label=algorithm)
        plot.figlegend()
        plot.ylabel("power")
        plot.xlabel("iteration")
        plot.title(f'Power 90 {datasource}')
        plot.savefig(f'{result_folder}/Power/{algorithm}_{datasource}_Power90.png',dpi=500)
        plot.clf()

        for algorithm in algorithms:
            p = algorithm_power95[(algorithm,datasource)]
            p.pop(0)
            y = np.nan_to_num(p)
            plot.plot(x,y, '.-', label=algorithm)
        plot.figlegend()
        plot.ylabel("power")
        plot.xlabel("iteration")
        plot.title(f'Power 95 {datasource}')
        plot.savefig(f'{result_folder}/Power/{algorithm}_{datasource}_Power95.png',dpi=500)
        plot.clf()

        for algorithm in algorithms:
            p = algorithm_power99[(algorithm,datasource)]
            p.pop(0)
            y = np.nan_to_num(p)
            plot.plot(x,y, '.-', label=algorithm)
        plot.figlegend()
        plot.ylabel("power")
        plot.xlabel("iteration")
        plot.title(f'Power 99 {datasource}')
        plot.savefig(f'{result_folder}/Power/{algorithm}_{datasource}_Power99.png',dpi=500)
        plot.clf()

def plot_power_noise():
    x = np.arange(5)
    for datasource in datasources:
        y = []
        for algorithm in algorithms:
            p = algorithm_power90[(algorithm,datasource)][99]
            y.append(p)
        plot.plot(x,y)
        plot.figlegend()
        plot.ylabel("power")
        plot.xlabel("noise")
        plot.title(f'Power 90 {datasource}')
        plot.savefig(f'{result_folder}/Power/{algorithm}_{datasource}_noise_Power90.png',dpi=500)
        plot.clf()

        y = []
        for algorithm in algorithms:
            p = algorithm_power95[(algorithm,datasource)][99]
            y.append(p)
        plot.plot(x,y, '.-', label=algorithm)
        plot.figlegend()
        plot.ylabel("power")
        plot.xlabel("noise")
        plot.title(f'Power 95 {datasource}')
        plot.savefig(f'{result_folder}/Power/{algorithm}_{datasource}_noise_Power95.png',dpi=500)
        plot.clf()

        y = []
        for algorithm in algorithms:
            p = algorithm_power99[(algorithm,datasource)][99]
            y.append(p)
        plot.plot(x,y, '.-', label=algorithm)
        plot.figlegend()
        plot.ylabel("power")
        plot.xlabel("noise")
        plot.title(f'Power 99 {datasource}')
        plot.savefig(f'{result_folder}/Power/{algorithm}_{datasource}_noise_Power99.png',dpi=500)
        plot.clf()

def plot_data_efficiency(fig_name = "data_efficiency"):
    plot_F1()
    plot_AUC()
    plot_power()

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

walk_files(log_folder)

keys = list(algorithm_data.keys())
datasources = set([x[1] for x in keys])
algorithms = sorted(set([x[0] for x in keys]))

calculate_power()
#calculate_F1()
#calculate_ROC_AUC()
#calculate_prediction_recall_AUC()

plot_power_noise()
#plot_data_efficiency()