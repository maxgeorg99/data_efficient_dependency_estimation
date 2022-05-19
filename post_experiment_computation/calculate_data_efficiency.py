from typing import Dict
from matplotlib import pyplot as plot # type: ignore
import numpy as np
import os

folder = "./experiment_results"
log_prefix = "log"
algorithms = ["optimal", "de", "ide"]
ignore_nans_count = -1

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
    
    plot.savefig(f'{folder}/{fig_name}.png',dpi=500)
    plot.clf()

calculate_means()
calculate_median()
calculate_vars()

plot_p_value()
