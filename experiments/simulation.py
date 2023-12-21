"""
This file aims to conduct experiments on synthetic data which have the similar distribution of real datasets.
"""
import sys

sys.path.append("./methods")
sys.path.append("./experiments")

from synthetic import (generator, save_data, load_data)
from util import user_accuracy
from hits import (hits, truthfinder, investment, pooledinvestment)
from spectral import ABH
from hitsndiffs import  HITSnDIFFS
from grm import grm_estimator
from baseline import baseline

import numpy as np
import matplotlib.pyplot as plt

def simulate(M, N, name, loading):
    """
    Simulate data based on the given shape and run experiments on the data

    Parameters
    -------
    M : int
        user number
    N : int
        question number
    name : string
        The experiment name
    loading : bool
        An indicator of whether we have already generated the data
    
    Returns
    -------
    result : array
        The accuracy result

    """

    if loading:
        Labels, theta, gold, discrimination = load_data(name)
    else:
        model = "3PL"
        if "halfmoon" in name:
            model = "3PL-Halfmoon"
        Labels, theta, gold, discrimination = generator(M, N, 2, 0, 0, model)
        save_data(Labels, theta, gold, name, discrimination)
    
    s_HITSnDIFFS = HITSnDIFFS(Labels)[0]
    s_ABH = ABH(Labels)[0]
    s_hits = hits(Labels)[0]
    s_truthfinder = truthfinder(Labels)[0]
    s_investment = investment(Labels)[0]
    s_pooledinvestment = pooledinvestment(Labels)[0]
    s_baseline = baseline(Labels, gold)
    s_grm = grm_estimator(Labels, discrimination, "Samejima")[0]
    
    result = np.zeros(8)
    result[0] = user_accuracy(theta, s_HITSnDIFFS)
    result[1] = user_accuracy(theta, s_ABH)
    result[2] = user_accuracy(theta, s_hits)
    result[3] = user_accuracy(theta, s_truthfinder)
    result[4] = user_accuracy(theta, s_investment)
    result[5] = user_accuracy(theta, s_pooledinvestment)
    result[6] = user_accuracy(theta, s_grm)
    result[7] = user_accuracy(theta, s_baseline)

    return result

def main():
    dataset = "simulation_halfmoon" # simulation_large, simulation_100 or simulation_halfmoon

    loading = True
    if loading:
        reading = np.load("experiments/result/" + dataset + ".npz")
        result = reading["result"]
        min = reading["min"]
        max = reading["max"]
        std = reading["std"]
    else:
        result = np.zeros((8, 10))
        max = np.zeros(8)
        min = np.zeros(8)
        average = np.zeros(8)
        std = np.zeros(8)
        for i in range(1,11):
            print(i)
            if dataset == "simulation_large":
                result[:,i - 1] = simulate(2692,40,dataset + '_' + str(i), True)
            elif dataset == "simulation_100":
                result[:,i - 1] = simulate(100,40,dataset + '_' + str(i), True)
            elif dataset == "simulation_halfmoon":
                result[:,i - 1] = simulate(100,100,dataset + '_' + str(i), True)
        for i in range(0,8):
            max[i] = np.max(result[i])
            min[i] = np.min(result[i])
            average[i] = np.sum(result[i]) / 10
            std[i] = np.std(result[i] * 100)
        result = average
        np.savez("experiments/result/" + dataset, result = result, min = min, max = max, std = std)
    
    if dataset == "simulation_large":
        result = np.delete(result, 3)
        std = np.delete(std, 3)

        result = [i * 100 for i in result]
        labels = ["HnD", "ABH", "HITS", "Inv", "PooledInv", "GRM-estimator", "True-answer"]
        plt.barh(labels, result, color=['r','b','g','g','g','y','y'], xerr = std)
        for x,y in zip(result, range(0,len(labels))):
            plt.text(x+1, y + 0.2, '%.2f'%x, va = "center", fontweight="bold", fontsize = "xx-large")
        plt.title('Accuracy of user ranking', fontdict={'fontsize':'xx-large'})
        plt.xlim(80,100)
        plt.xticks(fontsize = "x-large")
        plt.yticks(fontsize = "x-large")
        plt.savefig("experiments/figures/Figure_" + dataset + ".pdf", bbox_inches='tight')
        plt.show()

    elif dataset == "simulation_100":
        result = [i * 100 for i in result]
        labels = ["HnD", "ABH", "HITS", "TF", "Inv", "PooledInv", "GRM-estimator", "True-answer"]
        plt.barh(labels, result, color=['r','b','g','g','g','g','y','y'], xerr = std)
        for x,y in zip(result, range(0,len(labels))):
            plt.text(x+1, y + 0.2, '%.2f'%x, va = "center", fontweight="bold", fontsize = "xx-large")
        plt.title('Accuracy of user ranking', fontdict={'fontsize':'xx-large'})
        plt.xlim(50,100)
        plt.xticks(fontsize = "x-large")
        plt.yticks(fontsize = "x-large")
        plt.savefig("experiments/figures/Figure_" + dataset + ".pdf", bbox_inches='tight')
        plt.show()
    
    elif dataset == "simulation_halfmoon":
        result = [i * 100 for i in result]
        labels = ["HnD", "ABH", "HITS", "TF", "Inv", "PooledInv", "GRM-estimator", "True-answer"]
        plt.barh(labels, result, color=['r','b','g','g','g','g','y','y'], xerr = std)
        for x,y in zip(result, range(0,len(labels))):
            plt.text(x+1, y + 0.2, '%.2f'%x, va = "center", fontweight="bold", fontsize = "xx-large")
        plt.title('Accuracy of user ranking', fontdict={'fontsize':'xx-large'})
        plt.xlim(0,100)
        plt.xticks(fontsize = "x-large")
        plt.yticks(fontsize = "x-large")
        plt.savefig("experiments/figures/Figure_" + dataset + ".pdf", bbox_inches='tight')
        plt.show()
    
if __name__ == "__main__":
    main()