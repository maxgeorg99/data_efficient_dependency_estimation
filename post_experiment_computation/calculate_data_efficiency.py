from random import random
from typing import Dict, List
from matplotlib import pyplot as plot # type: ignore
import numpy as np
import os
from requests import get
from statsmodels.stats.power import TTestIndPower
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

result_folder = "./experiment_results/data_efficiency"
log_folder = "./log"
log_prefix = "DataEfficiency_"
algorithms = ["NaivDependencyTest"]
#algorithms = ["Pearson","Kendalltau","Spearmanr","XiCor"]
my_file = open("C:/Users/maxig/ThesisActiveLearningFramework/data_efficient_dependency_estimation/log/DataSources.txt", 'r')
datasources = my_file.read().splitlines()
datasources = list(set(datasources))
#datasources = ["LineDataSource","SquareDataSource","HyperSphereDataSource"]
num_experiments = 1
ignore_nans_count = -1
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
        TTestIndPower
        results = item[1]['score']
        t90 = percentile_scala_breeze(results, 0.90)
        t95 = percentile_scala_breeze(results, 0.95)
        t99 = percentile_scala_breeze(results, 0.99)
        algorithm_power90[item[0]] = sum([r > t90 for r in results]) / len(results)
        algorithm_power95[item[0]] = sum([r > t95 for r in results]) / len(results)
        algorithm_power99[item[0]] = sum([r > t99 for r in results]) / len(results)
        

algorithm_F1: Dict = {}
def calculate_F1():
    for algorithm in algorithms:
        f1_iterations = []
        for i in range(num_iterations):
            predictions = []
            for datasource in datasources:
                prediction = algorithm_data.get((algorithm,datasource))['predictions'][i]
                predictions.append(prediction)
            f1_iterations.append(f1_score(get_ground_truth(datasources), predictions))
        algorithm_F1[algorithm] = f1_iterations

algorithm_ROC_AUC: Dict = {}
def calculate_ROC_AUC():
    for algorithm in algorithms:
        predictions = []
        for datasource in datasources:
            prediction = np.median(algorithm_data.get((algorithm,datasource))['predictions'])
            predictions.append(prediction)
        auc = roc_auc_score(get_ground_truth(datasources), predictions)
        algorithm_ROC_AUC[algorithm] = auc

algorithm_prediction_recall_AUC: Dict = {}
def calculate_prediction_recall_AUC():
    for algorithm in algorithms:
        predictions = []
        for datasource in datasources:
            prediction = np.median(algorithm_data.get((algorithm,datasource))['predictions'])
            predictions.append(prediction)        
        auc = average_precision_score(get_ground_truth(datasources), predictions)
        algorithm_prediction_recall_AUC[algorithm] = auc

def get_ground_truth(y):
    return np.asarray( [1 if x.startswith('Independent') else calc_ground_truth() if x.startswith('Real') else 0 for x in y]
 )

def calc_ground_truth():
    return [algorithm_data.get(('MCDE',datasource))['predictions'][-1:] for datasource in datasources]

def plot_data_efficiency(fig_name = "data_efficiency"):
    y_pos = list(algorithms)

    for algorithm in algorithms:
        plot.bar(np.asarray([i for i in range(num_iterations)]), algorithm_F1[algorithm])
        plot.ylabel("F1")
        plot.savefig(f'{result_folder}/F1/{algorithm}_{fig_name}_F1.png',dpi=500)
        plot.clf()

    for datasource in datasources:

        negative_data = []
        positive_data = []
        for algorithm in algorithms:
            value = algorithm_ROC_AUC[algorithm]-0.5
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
calculate_ROC_AUC()
calculate_prediction_recall_AUC()

plot_data_efficiency()