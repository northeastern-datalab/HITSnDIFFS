"""
This file aims to conduct experiments to compare different methods on real-world datasets.
"""
import sys

sys.path.append("./methods")

from util import user_accuracy
from hits import (hits, truthfinder, investment, pooledinvestment)
from spectral import ABH
from hitsndiffs import  HITSnDIFFS
from baseline import baseline

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

def read_csv(folderename):
    """
    Read the answer and truth csv files in a given folder

    Parameters
    ----------
    filename : string
        The folder which contains the csv files

    Returns
    -------
    ans : 2-dimensional matrix
        The answer matrix
    truth : array
        The gold standard
    """
    answer = folderename + "/answer.csv"
    truth = folderename + "/truth.csv"

    answer = np.loadtxt(answer, dtype=int, delimiter=',').T
    truth = np.loadtxt(truth, dtype=int, delimiter=',')

    return answer, truth

def experiment(dataset):
    """
    Multi-choice experiment on user accuracy

    Parameters
    ----------
    dataset : string
        The folder which contains the data files

    Returns
    -------
    result : array
        The user accuracy list
    """
    Labels, gold = read_csv(dataset)

    s_HITSnDIFFS = HITSnDIFFS(Labels)[0]
    s_ABH = ABH(Labels)[0]
    s_hits = hits(Labels)[0]
    s_truthfinder = truthfinder(Labels)[0]
    s_investment = investment(Labels)[0]
    s_pooledinvestment = pooledinvestment(Labels)[0]
    s_baseline = baseline(Labels, gold)
    
    result = np.zeros(6)
    result[0] = user_accuracy(rankdata(s_baseline), rankdata(s_HITSnDIFFS))
    result[1] = user_accuracy(rankdata(s_baseline), rankdata(s_ABH))
    result[2] = user_accuracy(rankdata(s_baseline), rankdata(s_hits))
    result[3] = user_accuracy(rankdata(s_baseline), rankdata(s_truthfinder))
    result[4] = user_accuracy(rankdata(s_baseline), rankdata(s_investment))
    result[5] = user_accuracy(rankdata(s_baseline), rankdata(s_pooledinvestment))

    return result


def multi_choice(loading = False):
    """
    Accuracy experiments

    Parameters
    ----------
    loading : bool
        the indicator of whether to load the existing experimental result
    """
    result = []

    if loading:
        result = np.load("experiments/result/multichoice" + ".npy")
    else:
        result.append(experiment("datasets/Chinese"))
        result.append(experiment("datasets/English"))
        result.append(experiment("datasets/IT"))
        result.append(experiment("datasets/Medicine"))
        result.append(experiment("datasets/Pokemon"))
        result.append(experiment("datasets/Science"))
        result = np.asarray(result)
        np.save("experiments/result/multichoice", result)
    
    result = np.mean(result, axis=0)
    result = [i * 100 for i in result]
    labels = ["HnD", "ABH", "HITS", "TF", "Inv", "PooledInv"]
    plt.barh(labels, result, color=['r','b','g','g','g','g'])
    for x,y in zip(result, range(0,len(labels))):
        plt.text(x+1, y, '%.2f'%x, va = "center", fontweight="bold")
    plt.title('Accuracy of user ranking', fontdict={'fontsize':'xx-large'})
    plt.xlim(0,100)
    plt.xticks(fontsize = "x-large")
    plt.yticks(fontsize = "x-large")
    plt.savefig("experiments/figures/Figure_multichoice.pdf", bbox_inches='tight')
    plt.show()

def main():
    multi_choice(True)
    
if __name__ == "__main__":
    main()