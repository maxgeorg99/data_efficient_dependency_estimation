from typing import Dict
from matplotlib import pyplot as plot # type: ignore
import numpy as np
import os

folder = "./experiment_results/gain"
log_folder = "./log"
log_prefix = "log"

baseline = "Pearson"
num_iterations = 100
ignore_nans_count = -1
num_experiments = 1

class_string = os.path.dirname(log_folder).split('/')[-1]

def walk_files(path):
    for dirpath, dnames, fnames in os.walk(path):
        f: str
        for f in fnames:
            if f.endswith(".npz"):
                data = np.load(os.path.join(dirpath, f))
                c = f.split("_")
                algorithm = c[0]
                datasource = c[3]
                compute_data(data,algorithm,datasource)

algorithm_data: Dict = {}
def compute_data(data, algorithm, datasource):
    runs_data = algorithm_data.get((algorithm,datasource), [])
    runs_data.append(data)
    algorithm_data[(algorithm, datasource)] = runs_data


algorithm_means: Dict = {}
def calculate_means():
    for item in algorithm_data.items():
        mean = []
        for i in range(num_iterations):
            mean.append(np.nanmean([item[1][j]['pValues'][i] for j in range(num_experiments)], axis=0))
        algorithm_means[item[0]] = np.asarray(mean)
algorithm_medians: Dict = {}
def calculate_median():
    for item in algorithm_data.items():
        median = []
        for i in range(num_iterations):
            median.append(np.nanmedian([item[1][j]['pValues'][i] for j in range(num_experiments)], axis=0))
        algorithm_medians[item[0]] = np.asarray(median)
algorithm_vars: Dict = {}
def calculate_vars():
    for item in algorithm_data.items():
        vars = []
        for i in range(num_iterations):     
            vars.append(np.nanvar([item[1][j]['pValues'][i] for j in range(num_experiments)], axis=0))
        algorithm_vars[item[0]] = np.asarray(vars)
def plot_p_value(fig_name = "p-value"):

    for datasource in datasources:
        for algorithm in algorithms:
            key = (algorithm,datasource)

            y = algorithm_means[key]
            x = np.asarray([i for i in range(len(y))])
            y_err = algorithm_vars[key]
            plot.ylim(0,1)
            plot.errorbar(x, y, y_err, alpha=0.5, fmt=' ', label=f'{key[0]}_{key[1]}_var')
        
            y = algorithm_means[key]
            x = np.asarray([i for i in range(len(y))])
            plot.ylim(0,1)
            plot.plot(x, y, alpha=1, label=f'{key[0]}_{key[1]}_mean')

            y = algorithm_medians[key]
            x = np.asarray([i for i in range(len(y))])
            plot.ylim(0,1)
            plot.plot(x, y, alpha=1, label=f'{key[0]}_{key[1]}_median')

        plot.title(datasource,fontsize=20)
        plot.xlabel("learning iteration")
        plot.ylabel("p-value")
        plot.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

            
        plot.savefig(f'{folder}/{class_string}_{fig_name}_{datasource}.png',dpi=500, bbox_inches='tight')
        plot.clf()

gain_mean_p: Dict = {}
def calculate_p_mean_gain():
    for key in algorithm_data.keys():
        mean = algorithm_means[key]
        base_mean = algorithm_means[(baseline,key[1])]
        gain = (base_mean - mean) / base_mean
        gain_mean_p[key] = gain

gain_median_p: Dict = {}
def calculate_p_median_gain():
    for key in algorithm_data.keys():
        median = algorithm_medians[key]
        base_median = algorithm_medians[(baseline,key[1])]
        gain = (base_median - median) / base_median
        gain_median_p[key] = gain


def plot_p_gain(p_gain, fig_name = "p-gain"):
    for datasource in datasources:
        for algorithm in algorithms:
            key = (algorithm,datasource)
            y = p_gain[key]
            x = np.asarray([i for i in range(len(y))])
            plot.xlim(0,len(y))
            plot.ylim(-1,1)
            plot.plot(x, y, label=f'{key[0]}')

        plot.title(datasource,fontsize=20)
        plot.xlabel("learning iteration")
        plot.ylabel("p-value gain against baseline")
        plot.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

        
        plot.savefig(f'{folder}/{class_string}_{fig_name}_{datasource}.png',dpi=500, bbox_inches='tight')
        plot.clf()

max_p = 0.25
min_p = 0.75
mean_data_gain: Dict = {}
def calculate_mean_data_gain():
    for key in algorithm_data.keys():
        mean = algorithm_means[key]
        base_mean = algorithm_means[(baseline,key[1])]

        mean_data_gain[key] = calculate_data_gain(mean, base_mean, key)

median_data_gain: Dict = {}
def calculate_median_data_gain():
    for key in algorithm_data.keys():
        median = algorithm_medians[key]
        base_median = algorithm_medians[(baseline,key[1])]

        median_data_gain[key] = calculate_data_gain(median, base_median, key)

def calculate_data_gain(mean, base_mean, key):
        if 'Independent' in key[1] or 'Random' in key[1]:
            ps = np.linspace(max_p,1)
        else:
            ps = np.linspace(max_p, 0)
        itters = []
        base_itters = []
        gains_p = []
        for i in range(ps.shape[0]):
            if 'Independent' in key[1] or 'Random' in key[1]:
                itter = np.where(mean >= ps[i])
                base_itter = np.where(base_mean >= ps[i])
            else:
                itter = np.where(mean <= ps[i])
                base_itter = np.where(base_mean <= ps[i])

            try:
                base_itter = base_itter[0][0]
            except IndexError:
                base_itter = base_mean.shape[0]

            try:
                itter = itter[0][0]

                itters.append(itter)
                base_itters.append(base_itter)
                gains_p.append(ps[i])

            except IndexError:
                ...

        itters = np.asarray(itters)
        base_itters = np.asarray(base_itters)
        gains_p = np.asarray(gains_p)

        gain = (base_itters - itters) / base_itters
        return (gains_p, gain)


def plot_data_gain(data_gain, fig_name = "data-gain"):
    for datasource in datasources:
        for algorithm in algorithms:
            key = (algorithm,datasource)
            x, y = data_gain[key]
            if 'Independent' in datasource or 'Random' in datasource:
                plot.xlim(min_p,1)
            else:
                plot.xlim(max_p,0)
            plot.ylim(-1,1)
            plot.plot(x, y, label=f'{key[0]}')

        plot.title(datasource[:-1],fontsize=20)
        plot.xlabel("p-value")
        plot.ylabel("data gain against baseline")
        plot.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    
        plot.savefig(f'{folder}/{class_string}_{fig_name}_{datasource}.png',dpi=500, bbox_inches='tight')
        plot.clf()




walk_files(log_folder)

keys = list(algorithm_data.keys())
datasources = set([x[1] for x in keys])
algorithms = set([x[0] for x in keys])

calculate_means()
calculate_median()
calculate_vars()

plot_p_value()

calculate_p_mean_gain()
calculate_p_median_gain()

plot_p_gain(gain_mean_p, "p-mean-gain")
plot_p_gain(gain_median_p, "p-median-gain")

calculate_mean_data_gain()
calculate_median_data_gain()

plot_data_gain(mean_data_gain, "mean-data-gain")
plot_data_gain(median_data_gain, "median-data-gain")