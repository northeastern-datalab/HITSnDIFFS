"""
This file provides the comparison of HnD vs ABH
"""
import sys

sys.path.append("./methods")
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (spearmanr, rankdata)

from spectral import ABH, getCMatrix
from hitsndiffs import HITSnDIFFS
from synthetic import generator
from util import user_accuracy, user_difference

def largest_eig(M):
    """
    Return the largest eigenvalue and corresponding eigenvector of a matrix

    Parameters
    ----------
    M : matrix (2-dimensional array)
        The input matrix

    Returns
    -------
    val : double
        the largest eigenvalue
    vec : vector
        the corresponding eigenvector of the largest eigenvalue

    """

    eigvals, eigvecs = np.linalg.eig(M)
    eigvals_sort = sorted(eigvals)
    val = eigvals_sort[-1]
    eig_idx = min([i for i,j in enumerate(eigvals) if j==eigvals_sort[-1]])
    vec = eigvecs[:, eig_idx]

    return val, vec

def compare_HnDvsABH(a, iter=100):
    """
    The comparison experiments to compare HnD and ABH in terms of their accuracy and stability for a given discrimination

    Parameters
    ----------
    a : double
        discrimination
    iter : int
        iteration number

    Returns
    -------
    accuracy_HnD : double
        The accuracy of the HnD method
    variance_HnD : double
        The average variance of each student's ranking given by the HnD method
    difference_HnD : double
        The average difference between each student's ranking given by the HnD method compared to the correct ranking, normalized by user number
    accuracy_ABH : double
        The accuracy of the ABH method
    variance_ABH : double
        The average variance of each student's ranking given by the ABH method 
    difference_ABH : double
        The average difference between each student's ranking given by the ABH method compared to the correct ranking, normalized by user number
    """
    M = 100
    N = 100
    K = 3
    b = 0.5

    HnD_set = np.zeros((iter,M))
    ABH_set = np.zeros((iter,M))

    accuracy_HnD = 0
    accuracy_ABH = 0

    for i in range(0,iter):
        Labels, theta, gold, discrimination = generator(M, N, K, b, a, "Bock", "equispaced")
        s_HITSnDIFFS = HITSnDIFFS(Labels)[0]
        s_ABH = ABH(Labels)[0]

        accuracy_HnD += user_accuracy(s_HITSnDIFFS, theta)
        accuracy_ABH += user_accuracy(s_ABH, theta)
        HnD_set[i] = rankdata(s_HITSnDIFFS)
        ABH_set[i] = rankdata(s_ABH)

    accuracy_HnD /= iter
    accuracy_ABH /= iter

    difference_HnD = 0
    difference_ABH = 0

    for i in range(0,M):
        difference_HnD += np.sum(abs(HnD_set[:,i] - i)) / iter
        difference_ABH += np.sum(abs(ABH_set[:,i] - i)) / iter

    difference_HnD /= M
    difference_ABH /= M

    return accuracy_HnD, difference_HnD / M, accuracy_ABH, difference_ABH / M

def comparison_HnDvsABH(loading=False):
    """
    The comparison experiments to compare HnD and ABH in terms of their accuracy and stability

    Parameters
    ----------
    loading : bool
        the indicator of whether to load the existing experimental result
    """
    aset = np.arange(21)

    result = []

    if loading:
        result = np.load("experiments/result/comparison_HnDvsABH" + ".npy")
    else:
        for a in aset:
            print(a)
            result.append(compare_HnDvsABH(a))
        result = np.asarray(result)
        np.save("experiments/result/comparison_HnDvsABH", result)

    accuracy_HnD = result[:,0]
    difference_HnD = result[:,1]
    accuracy_ABH = result[:,2]
    difference_ABH = result[:,3]

    plt.xlabel("Question discrimination", fontsize="xx-large")
    plt.ylabel("Accuracy of user ranking", fontsize="x-large")
    plt.xticks(fontsize = "x-large")
    plt.yticks(fontsize = "x-large")
    plt.xscale("log", base=2)
    plt.ylim(0,1)
    plt.plot(aset, accuracy_ABH, marker = 'o', label = "ABH")
    plt.plot(aset, accuracy_HnD, marker = 's', label = "HnD")
    plt.legend(fontsize="xx-large", labelspacing = 0.1)
    plt.savefig("experiments/figures/Figure_comparison_accuracy.pdf", bbox_inches='tight')
    plt.show()
    plt.clf()

    plt.xlabel("Question discrimination", fontsize="xx-large")
    plt.ylabel("Normalized user displacement", fontsize="x-large")
    plt.xticks(fontsize = "x-large")
    plt.yticks(fontsize = "x-large")
    plt.xscale("log", base=2)
    plt.plot(aset, difference_ABH, marker = 'o', label = "ABH")
    plt.plot(aset, difference_HnD, marker = 's', label = "HnD")
    plt.legend(fontsize="xx-large", labelspacing = 0.1)
    plt.savefig("experiments/figures/Figure_comparison_difference.pdf", bbox_inches='tight')
    plt.show()
    plt.clf()

def compare_variance(a, iter=100):
    """
    The comparison experiments to compare the variance of Udiff and βI-M for a given discrimination

    Parameters
    ----------
    a : double
        discrimination
    iter : int
        iteration number

    Returns
    -------
    variance_HnD : double
        The variance of the largest eigenvector of Udiff
    variance_ABH : double
        The variance of the largest eigenvector of βI-M
    """
    M = 100
    N = 100
    K = 3
    b = 0.5

    variance_HnD = 0
    variance_ABH = 0

    for i in range(0,iter):
        Labels, theta, gold, discrimination = generator(M, N, K, b, a, "Bock", "equispaced")
        
        C = getCMatrix(Labels)

        S = np.zeros((M-1, M))
        for i in range(0, M-1):
            S[i][i] = -1
            S[i][i+1] = 1
        T = np.zeros((M, M-1))
        for i in range(0, M):
            for j in range(0, i):
                T[i][j] = 1

        C_l0 = C.sum(0)
        C_l1 = C.sum(1)
        C_l0[C_l0 == 0] = 1
        C_col = np.true_divide(C, C_l0)
        C_row = np.true_divide(C.T, C_l1).T
        U = C_row.dot(C_col.T)
        Udiff = S.dot(U).dot(T)

        A = C.dot(C.T)
        L = np.diag(np.sum(A,1)) - A
        Matrix = S.dot(L).dot(T)
        betaIminusM = np.max(Matrix) * np.identity(M-1) - Matrix

        sdiff_hnd = largest_eig(Udiff)[1]
        sdiff_abh = largest_eig(betaIminusM)[1]

        variance_HnD += np.var(abs(sdiff_hnd))
        variance_ABH += np.var(abs(sdiff_abh))

    variance_HnD /= iter
    variance_ABH /= iter

    return variance_HnD, variance_ABH

def comparison_variance(loading=False):
    """
    The comparison experiments to compare the largest eigenvector of Udiff and βI-M

    Parameters
    ----------
    loading : bool
        the indicator of whether to load the existing experimental result
    """
    aset = np.arange(21)

    result = []

    if loading:
        result = np.load("experiments/result/comparison_variance" + ".npy")
    else:
        for a in aset:
            print(a)
            result.append(compare_variance(a))
        result = np.asarray(result)
        np.save("experiments/result/comparison_variance", result)

    variance_HnD = result[:,0]
    variance_ABH = result[:,1]

    plt.xlabel("Question discrimination", fontsize="xx-large")
    plt.ylabel("Variance of the largest eigenvector", fontsize="x-large")
    plt.xticks(fontsize = "x-large")
    plt.yticks(fontsize = "x-large")
    plt.xscale("log", base=2)
    plt.plot(aset, variance_ABH, marker = 'o', label = "βI-M of ABH")
    plt.plot(aset, variance_HnD, marker = 's', label = "Udiff of HnD")
    plt.legend(fontsize="xx-large", labelspacing = 0.1)
    plt.savefig("experiments/figures/Figure_comparison_variance.pdf", bbox_inches='tight')
    plt.show()
    plt.clf()

def main():
    comparison_HnDvsABH(True)
    comparison_variance(True)

if __name__ == "__main__":
    main()

