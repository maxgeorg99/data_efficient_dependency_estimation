#From https://scicomp.stackexchange.com/questions/26059/computing-rate-and-order-of-convergence
#Function to calculate order of convergence  
import statistics as stat
import numpy as np

result_folder = "./experiment_results/convergence"
fig_name = 'convergence'
log_folder = "./log"

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

def load_r_values():
    pass

rs = load_r_values()
convergence_rate(1,rs)

#plot?! 