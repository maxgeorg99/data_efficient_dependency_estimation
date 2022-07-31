from random import random
from turtle import width
from typing import Dict, List
from matplotlib import pyplot as plot # type: ignore
import numpy as np
import os
import pandas as pd
from requests import get
from statsmodels.stats.power import TTestIndPower
from sklearn.metrics import auc, average_precision_score, confusion_matrix, f1_score, precision_recall_curve, roc_auc_score
import statsmodels.stats.power as smp
import Orange
from scipy.stats import rankdata

result_folder = "./experiment_results/data_efficiency"
log_folder = "./logs/class4/dim"
log_prefix = "DataEfficiency_"
num_experiments = 1
ignore_nans_count = -1
num_iterations = 100

class_string = os.path.dirname(log_folder).split('/')[-1]

def walk_files(path):
    for dirpath, dnames, fnames in os.walk(path):
        f: str
        for f in fnames:
            if f.endswith(".npz"):
                data = np.load(os.path.join(dirpath, f))
                c = f.split("_")
                algorithm = c[0]
                datasource = ' '.join(c[1:4])
                with np.load(os.path.join(dirpath, f)) as data:
                    compute_data_efficiency(data,algorithm,datasource)

algorithm_pvalues: Dict = {}
def compute_data_efficiency(data, algorithm, datasource):
    runs_data_p = algorithm_pvalues.get((algorithm,datasource), [])
    runs_data_p.append(data['pValues'][0:101])
    algorithm_pvalues[(algorithm, datasource)] = runs_data_p 

algorithm_F1_90: Dict = {}
algorithm_F1_95: Dict = {}
algorithm_F1_99: Dict = {}
def calculate_F1_noise():
    noise_levels = [0.0,0.5,2.0]
    p_thresholds = [0.01,0.05,0.1]
    for noise in noise_levels:
        for algorithm in algorithms:
            datasources_with_noise = [datasource for datasource in datasources if datasource.startswith('noise ' + str(noise)) or 'Independent' in datasource or 'Random' in datasource]
            avg_pvalues = np.stack([algorithm_pvalues[(algorithm, ds)] for ds in datasources_with_noise]).mean(1)
            F1_90 = []  
            F1_95 = []
            F1_99 = []
            for i in range(num_iterations):
                F1_scores = []
                for p in p_thresholds:
                    predictions = [1 if x >= p else 0 for x in avg_pvalues[:,i]]
                    f1 = f1_score(get_ground_truth(datasources_with_noise),predictions)
                    F1_scores.append(f1)
                F1_90.append(F1_scores[0])
                F1_95.append(F1_scores[1])
                F1_99.append(F1_scores[2])
            algorithm_F1_90[(algorithm,noise)] = F1_90
            algorithm_F1_95[(algorithm,noise)] = F1_95
            algorithm_F1_99[(algorithm,noise)] = F1_99

algorithm_ROC_AUC90: Dict = {}
algorithm_ROC_AUC95: Dict = {}
algorithm_ROC_AUC99: Dict = {}
def calculate_ROC_AUC():
    noise_levels = [0.0,0.5,2.0]
    p_thresholds = [0.01,0.05,0.1]
    for noise in noise_levels:
        for algorithm in algorithms:
            datasources_with_noise = [datasource for datasource in datasources if datasource.startswith('noise ' + str(noise)) or 'Independent' in datasource or 'Random' in datasource]
            avg_pvalues = np.stack([algorithm_pvalues[(algorithm, ds)] for ds in datasources_with_noise]).mean(1)
            AUC_90 = []
            AUC_95 = []
            AUC_99 = []
            for i in range(num_iterations):
                auc_score_for_p = []
                for p in p_thresholds:
                    predictions = [1 if x >= p else 0 for x in avg_pvalues[:,i]]
                    roc_auc = roc_auc_score(get_ground_truth(datasources_with_noise),predictions)
                    auc_score_for_p.append(roc_auc)
                AUC_90.append(auc_score_for_p[0])
                AUC_95.append(auc_score_for_p[1])
                AUC_99.append(auc_score_for_p[2])
            algorithm_ROC_AUC90[(algorithm,noise)] = AUC_90
            algorithm_ROC_AUC95[(algorithm,noise)] = AUC_95
            algorithm_ROC_AUC99[(algorithm,noise)] = AUC_99

algorithm_PR_AUC90: Dict = {}
algorithm_PR_AUC95: Dict = {}
algorithm_PR_AUC99: Dict = {}
def calculate_PR_AUC():
    noise_levels = [0.0,0.5,2.0]
    p_thresholds = [0.01,0.05,0.1]
    for noise in noise_levels:
        for algorithm in algorithms:
            datasources_with_noise = [datasource for datasource in datasources if datasource.startswith('noise ' + str(noise)) or 'Independent' in datasource or 'Random' in datasource]
            avg_pvalues = np.stack([algorithm_pvalues[(algorithm, ds)] for ds in datasources_with_noise]).mean(1)
            AUC_90 = []
            AUC_95 = []
            AUC_99 = []
            for i in range(num_iterations):
                auc_score_for_p = []
                for p in p_thresholds:
                    predictions = [1 if x >= p else 0 for x in avg_pvalues[:,i]]
                    PR_AUC = average_precision_score(get_ground_truth(datasources_with_noise),predictions)
                    auc_score_for_p.append(PR_AUC)
                AUC_90.append(auc_score_for_p[0])
                AUC_95.append(auc_score_for_p[1])
                AUC_99.append(auc_score_for_p[2])
            algorithm_PR_AUC90[(algorithm,noise)] = AUC_90
            algorithm_PR_AUC95[(algorithm,noise)] = AUC_95
            algorithm_PR_AUC99[(algorithm,noise)] = AUC_99

algorithm_power90: Dict = {}
algorithm_power95: Dict = {}
algorithm_power99: Dict = {}
def calculate_power_noise():
    noise_levels = [0.0,0.5,2.0]
    p_thresholds = [0.01,0.05,0.1]
    for noise in noise_levels:
        for algorithm in algorithms:
            datasources_with_noise = [datasource for datasource in datasources if datasource.startswith('noise ' + str(noise)) or 'Independent' in datasource or 'Random' in datasource]
            avg_pvalues = np.stack([algorithm_pvalues[(algorithm, ds)] for ds in datasources_with_noise]).mean(1)
            power90 = []
            power95 = []
            power99 = []
            for i in range(num_iterations):
                power = []
                for p in p_thresholds:
                    predictions = [1 if x >= p else 0 for x in avg_pvalues[:,i]]

                    cnf_matrix = confusion_matrix(get_ground_truth(datasources_with_noise), predictions)

                    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
                    TP = np.diag(cnf_matrix)

                    FN = FN.astype(float)
                    TP = TP.astype(float)

                    # Sensitivity, hit rate, recall, or true positive rate
                    TPR = TP/(TP+FN)
                    power.append(TPR[0])
                power90.append(power[0])
                power95.append(power[1])
                power99.append(power[2])
            algorithm_power90[(algorithm,noise)] = power90
            algorithm_power95[(algorithm,noise)] = power95
            algorithm_power99[(algorithm,noise)] = power99

algorithm_power95_dim: Dict = {}
def calculate_power_dim():
    dimensions = [1,2,3,4,5]
    for algorithm in algorithms:
        for d in dimensions:
            datasource_with_dim = [datasource for datasource in datasources if str(d) + 'd' in datasource or 'Independent' in datasource or 'Random' in datasource]
            avg_pvalues = np.stack([algorithm_pvalues[(algorithm, ds)] for ds in datasource_with_dim]).mean(1)
            power95 = []
            for i in range(num_iterations):
                predictions = [1 if x >= 0.05 else 0 for x in avg_pvalues[:,i]]

                cnf_matrix = confusion_matrix(get_ground_truth(datasource_with_dim), predictions)

                FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
                TP = np.diag(cnf_matrix)

                FN = FN.astype(float)
                TP = TP.astype(float)

                # Sensitivity, hit rate, recall, or true positive rate
                TPR = TP/(TP+FN)
                power95.append(TPR[0])  
            algorithm_power95_dim[(algorithm,d)] = power95


def get_ground_truth(y):
    return np.asarray([1 if 'Independent' in x or 'Random' in x else calc_ground_truth(y) if 'Interpolating' in x else 0 for x in y])

def calc_ground_truth(y):
    return np.asarray([])

def plot_F1():
    x = np.arange(num_iterations)
    p_thresholds = [0.01,0.05,0.1]
    for p in range(len(p_thresholds)):
        for algorithm in algorithms:
            y = [algorithm_F1_90[algorithm,i][p] for i in range(num_iterations)]
            plot.plot(x,y,  algo_marker_dict[algorithm] + '-', label=algorithm)
        plot.ylabel("F1 score")
        plot.xlabel("iteration")
        plot.savefig(f'{result_folder}/F1/{class_string}_F1_{p_thresholds[p]}.png',dpi=500)
        plot.clf()

def plot_F1_noise():
    x = np.arange(num_iterations)
    noise_level = [0.0,0.5,2.0]
    for noise in noise_level:
        for algorithm in algorithms:
            y = algorithm_F1_90[algorithm,noise]
            plot.plot(x,y,algo_marker_dict[algorithm] + '-', label=algorithm)
        plot.figlegend()
        plot.ylim(top=1)
        plot.ylabel("F1 score")
        plot.xlabel("iteration")
        plot.title(f'F1 90',fontsize=30)
        plot.savefig(f'{result_folder}/F1/noise_F1_90_{noise}_{class_string}.png',dpi=500)
        plot.clf()

        for algorithm in algorithms:
            y = algorithm_F1_95[algorithm,noise]
            plot.plot(x,y,algo_marker_dict[algorithm] + '-', label=algorithm)
        plot.figlegend()
        plot.ylim(top=1)
        plot.ylabel("F1 score")
        plot.xlabel("iteration")
        plot.title(f'F1 95',fontsize=30)
        plot.savefig(f'{result_folder}/F1/noise_F1_95_{noise}_{class_string}.png',dpi=500)
        plot.clf()

        for algorithm in algorithms:
            y = algorithm_F1_99[algorithm,noise]
            plot.plot(x,y,algo_marker_dict[algorithm] + '-', label=algorithm)
        plot.figlegend()
        plot.ylim(top=0.7)
        plot.ylabel("F1 score")
        plot.xlabel("iteration")
        plot.title(f'F1 99',fontsize=30)
        plot.savefig(f'{result_folder}/F1/noise_F1_99_{noise}_{class_string}.png',dpi=500)
        plot.clf()

def plot_AUC():
    x = np.arange(num_iterations)
    p_thresholds = [0.01,0.05,0.1]
    noise_level = [0.0,0.5,2.0]
    for noise in noise_level:
        for algorithm in algorithms:
            y = algorithm_ROC_AUC90[(algorithm,noise)]
            plot.plot(x,y,  algo_marker_dict[algorithm] + '-', label=algorithm)
        plot.ylabel("ROC_AUC score")
        plot.figlegend()
        plot.xlabel("iteration")
        plot.title(f'ROC AUC 90',fontsize=30)
        plot.figlegend()
        plot.savefig(f'{result_folder}/ROC_AUC/noise_ROC_AUC_90_{noise}_{class_string}.png',dpi=500)
        plot.clf()

        for algorithm in algorithms:
            y = algorithm_ROC_AUC95[(algorithm,noise)]
            plot.plot(x,y,  algo_marker_dict[algorithm] + '-', label=algorithm)
        plot.ylabel("ROC_AUC score")
        plot.figlegend()
        plot.xlabel("iteration")
        plot.title(f'ROC AUC 95',fontsize=30)
        plot.figlegend()
        plot.savefig(f'{result_folder}/ROC_AUC/noise_ROC_AUC_95_{noise}_{class_string}.png',dpi=500)
        plot.clf()

        for algorithm in algorithms:
            y = algorithm_ROC_AUC99[(algorithm,noise)]
            plot.plot(x,y,  algo_marker_dict[algorithm] + '-', label=algorithm)
        plot.ylabel("ROC_AUC score")
        plot.figlegend()
        plot.xlabel("iteration")
        plot.title(f'ROC AUC 99',fontsize=30)
        plot.figlegend()
        plot.savefig(f'{result_folder}/ROC_AUC/noise_ROC_AUC_99_{noise}_{class_string}.png',dpi=500)
        plot.clf()

def plot_PR_AUC():
    x = np.arange(num_iterations)
    p_thresholds = [0.01,0.05,0.1]
    noise_level = [0.0,0.5,2.0]
    for noise in noise_level:
        for algorithm in algorithms:
            y = algorithm_PR_AUC90[(algorithm,noise)]
            plot.plot(x,y,  algo_marker_dict[algorithm] + '-', label=algorithm)
        plot.ylabel("Precision_Recall_AUC score")
        plot.figlegend()
        plot.xlabel("iteration")
        plot.title(f'PR_AUC 90',fontsize=30)
        plot.savefig(f'{result_folder}/Precision_Recall_AUC/noise_Precision_Recall_AUC_90_{noise}_{class_string}.png',dpi=500)
        plot.clf()

        for algorithm in algorithms:
            y = algorithm_PR_AUC95[(algorithm,noise)]
            plot.plot(x,y,  algo_marker_dict[algorithm] + '-', label=algorithm)
        plot.ylabel("Precision_Recall_AUC score")
        plot.figlegend()
        plot.xlabel("iteration")
        plot.title(f'PR_AUC 95',fontsize=30)
        plot.savefig(f'{result_folder}/Precision_Recall_AUC/noise_Precision_Recall_AUC_95_{noise}_{class_string}.png',dpi=500)
        plot.clf()

        for algorithm in algorithms:
            y = algorithm_PR_AUC99[(algorithm,noise)]
            plot.plot(x,y,  algo_marker_dict[algorithm] + '-', label=algorithm)
        plot.ylabel("Precision_Recall_AUC score")
        plot.figlegend()
        plot.xlabel("iteration")
        plot.title(f'PR_AUC 99',fontsize=30)
        plot.savefig(f'{result_folder}/Precision_Recall_AUC/noise_Precision_Recall_AUC_99_{noise}_{class_string}.png',dpi=500)
        plot.clf()


def plot_power():
    x = np.arange(num_iterations)
    for algorithm in algorithms:
        p = algorithm_power90[(algorithm,0.0)]
        y = np.nan_to_num(p)
        plot.plot(x,y,  algo_marker_dict[algorithm] + '-', label=algorithm)
    plot.ylim(top=1)
    plot.ylabel("power")
    plot.xlabel("iteration")
    plot.title(f'Power 90',fontsize=30)
    plot.legend()
    plot.savefig(f'{result_folder}/Power/{class_string}_Power90.png',dpi=500)
    plot.clf()

    for algorithm in algorithms:
        p = algorithm_power95[(algorithm,0.0)]
        y = np.nan_to_num(p)
        plot.plot(x,y,  algo_marker_dict[algorithm] + '-', label=algorithm)
    plot.ylim(top=1)
    plot.ylabel("power")
    plot.xlabel("iteration")
    plot.title(f'Power 95',fontsize=30)
    plot.legend()
    plot.savefig(f'{result_folder}/Power/{class_string}_Power95.png',dpi=500)
    plot.clf()

    for algorithm in algorithms:
        p = algorithm_power99[(algorithm,0.0)]
        y = np.nan_to_num(p)
        plot.plot(x,y,  algo_marker_dict[algorithm] + '-', label=algorithm)
    plot.ylim(top=1)
    plot.ylabel("power")
    plot.xlabel("iteration")
    plot.title(f'Power 99',fontsize=30)
    plot.legend()
    plot.savefig(f'{result_folder}/Power/{class_string}_Power99.png',dpi=500)
    plot.clf()

def plot_power_noise():
    noise_level = [0.0,0.5,2.0]
    x = list(range(num_iterations))
    for noise in noise_level:
        for algorithm in algorithms:
            y = algorithm_power90[(algorithm,noise)]
            plot.plot(x,y,algo_marker_dict[algorithm] + '-', label=algorithm)
        plot.ylim(top=1)
        plot.ylabel("power")
        plot.xlabel("iteration")
        plot.figlegend()
        plot.title(f'Power 90',fontsize=30)
        plot.savefig(f'{result_folder}/Power/noise_Power90_{noise}_{class_string}.png',dpi=500)
        plot.clf()

        for algorithm in algorithms:
            y = algorithm_power95[(algorithm,noise)]    
            plot.plot(x,y,  algo_marker_dict[algorithm] + '-', label=algorithm)
        plot.ylim(top=1)
        plot.ylabel("power")
        plot.xlabel("iteration")
        plot.figlegend()
        plot.title(f'Power 95',fontsize=30)
        plot.savefig(f'{result_folder}/Power/noise_Power95_{noise}_{class_string}.png',dpi=500)
        plot.clf()

        for algorithm in algorithms:
            y = algorithm_power99[(algorithm,noise)]
            plot.plot(x,y,  algo_marker_dict[algorithm] + '-', label=algorithm)
        plot.ylim(top=1)
        plot.ylabel("power")
        plot.xlabel("iteration")
        plot.figlegend()
        plot.title(f'Power 99',fontsize=30)
        plot.savefig(f'{result_folder}/Power/noise_Power99_{noise}_{class_string}.png',dpi=500)
        plot.clf()

def plot_power_dim():
    dim = [1,2,3,4,5]
    x = list(range(num_iterations))
    for d in dim:
        for algorithm in algorithms:
            y = algorithm_power95_dim[(algorithm,d)]
            plot.plot(x,y,algo_marker_dict[algorithm] + '-', label=algorithm)
        plot.ylim(top=1)
        plot.ylabel("power")
        plot.xlabel("iteration")
        plot.figlegend()
        plot.title(f'Power 95 d = {d}',fontsize=30)
        plot.savefig(f'{result_folder}/Power/Power95_{d}_{class_string}.png',dpi=500)
        plot.clf()

def plot_cd_F1():
    avranks = []
    ranks = []
    for i in range(num_experiments):
        scores = np.nan_to_num([algorithm_F1_95[algorithm,num_iterations-1][1] for algorithm in algorithms])
        ranks.append(rankdata(scores))
    ranks = np.asarray(ranks)
    for i in range(len(algorithms)): 
        avranks.append(np.mean(ranks[:,i]))
    cd = Orange.evaluation.compute_CD(avranks, len(datasources))
    Orange.evaluation.graph_ranks(avranks, algorithms, cd=cd, width = 6, textspace=1.5,filename=f'{result_folder}/CD/{class_string}_cd_f1.png')

def plot_cd_power():
    avranks = []
    ranks = []
    for i in range(num_experiments):
        scores = np.nan_to_num([algorithm_power95[(algorithm,0.0)][99] for algorithm in algorithms])
        ranks.append(rankdata(scores))
    ranks = np.asarray(ranks)
    for i in range(len(algorithms)): 
        avranks.append(np.mean(ranks[:,i]))
    cd = Orange.evaluation.compute_CD(avranks, len(datasources))
    Orange.evaluation.graph_ranks(avranks, algorithms, cd=cd, width = 6, textspace=1.5,filename=f'{result_folder}/CD/{class_string}_cd_power.png')

def plot_cd_AUC():
    avranks = []
    for algorithm in algorithms:
        avranks.append(algorithm_ROC_AUC95[algorithm,num_iterations-1]) 
    cd = Orange.evaluation.compute_CD(avranks, len(datasources))
    Orange.evaluation.graph_ranks(avranks, algorithms, cd=cd, width = 6, textspace=1.5,filename=f'{result_folder}/CD/{class_string}_cd_ROC_AUC.png')

def plot_cd_PR_AUC():
    avranks = []
    for algorithm in algorithms:
        avranks.append(algorithm_PR_AUC95[algorithm,num_iterations-1]) 
    cd = Orange.evaluation.compute_CD(avranks, len(datasources))
    Orange.evaluation.graph_ranks(avranks, algorithms, cd=cd, width = 6, textspace=1.5,filename=f'{result_folder}/CD/{class_string}_cd_PR _AUC.png')


def plot_data_efficiency(fig_name = "data_efficiency"):
    plot_F1()
    plot_AUC()
    plot_power()

def plot_cd():
    plot_cd_F1()
    plot_cd_power()
    plot_cd_AUC()

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

keys = list(algorithm_pvalues.keys())
datasources = set([x[1] for x in keys])
algorithms = sorted(set([x[0] for x in keys]))
markers = ['.','v','^','<','>','s','P','*','+','x','D','d','|']
algo_marker_dict = dict(zip(algorithms,markers))

#calculate_power_noise()
#calculate_F1_noise()
#calculate_ROC_AUC()
#calculate_PR_AUC()
#calculate_prediction_recall_AUC()
#calculate_power_dim()
#calculate_power_dim()

#plot_F1_noise()
#plot_power()
#plot_power_noise()
#plot_data_efficiency()
plot_cd_power()
#plot_AUC()
#plot_PR_AUC()
#plot_power_dim()