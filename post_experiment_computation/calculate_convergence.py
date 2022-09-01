#From https://scicomp.stackexchange.com/questions/26059/computing-rate-and-order-of-convergence
#Function to calculate order of convergence  
import os
import statistics as stat
from typing import Dict
from matplotlib import pyplot as plt
import numpy as np

result_folder = "./experiment_results/convergence"
fig_name = 'convergence'
log_folder = "./log"
algorithms = ["Pearson","Kendalltau","Spearmanr","MCDE","HiCS","CMI"]
#datasources = ["LineDataSource","SquareDataSource","CrossDataSource","DoubleLinearDataSource","HourglassDataSource","HypercubeDataSource"]
datasources = ["LineDataSource","SquareDataSource","CrossDataSource","DoubleLinearDataSource","HourglassDataSource","HypercubeDataSource","SineDataSource","StarDataSource","ZDataSource","InvZDataSource"]

algorithms_color_dict = {
    'Pearson':"green",
    'Kendalltau': "purple",
    'Spearmanr': "orange",
    'CMI': 'blue',
    'MCDE': 'red',
    'HiCS': 'pink',
}
num_experiments = 5

def walk_algorithms():
    for algorithm in algorithms:
        for datasource in datasources:
            walk_files(algorithm, datasource)

def walk_files( algorithm, datasource):
        for i in range(num_experiments):
            f = f"{log_folder}/{algorithm}_{datasource}_{i}.npz"
            data = np.load(f)
            prepare_data(data, algorithm, datasource)

algorithm_data: Dict = {}
def prepare_data(data, algorithm, datasource):
    runs_data = algorithm_data.get((algorithm, datasource), [])
    runs_data.append(data)
    algorithm_data[(algorithm, datasource)] = runs_data

def convergence_order(x,e):
    p = np.log(e[2:]/e[1:-1]) / np.log(e[1:-1]/e[:-2])
    #mw = p.sum()/len(p)
    mw = stat.median(p)/len(p)
    return mw

#Function to calculate rate of convergence (for linear convergence)
def convergence_rate(x,e):
    n = len(e)
    k = np.arange(0,n)
    fit = np.polyfit(k,np.log(e),1)
    L = np.exp(fit[0])
    return L

def plot_convergence(): 
    #for each datasource plot variance of every algorithm
    #legend with all algorithms colors
    start = 0
    end = 100
    step = 1
    num_experiments = 5
    it = np.arange(start, end, step)

    for datasource in datasources:
        for algorithm in algorithms:
            variances = []
            for curr in range(start, end):
                #extend to numer of runs
                subvariances = []
                for i in range(num_experiments):
                    subvariances.append(np.var(algorithm_data[(algorithm, datasource)][i]['pValues'][0:curr]))
                variances.append(stat.mean(subvariances))
            plt.plot(it, variances,'-', color = algorithms_color_dict[algorithm],linewidth=1,label = algorithm)
        plt.ylabel("Var")
        plt.grid()
        plt.title(datasource)
        plt.xlim(start, end)
        plt.legend(loc = "upper right")
        plt.savefig(f'{result_folder}/{datasource}_{fig_name}.png',dpi=500)
        plt.clf()

walk_algorithms()

plot_convergence()