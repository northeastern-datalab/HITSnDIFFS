"""
This file provides some functions used in experiments
"""
import sys
sys.path.append("./methods")

from hits import (hits, truthfinder, investment, pooledinvestment)
from spectral import (ABH, ABH_power_iteration, getCMatrix)
from grm import grm_estimator
from hitsndiffs import (HITSnDIFFS, AvgHITS_2nd_eigenvector_deflation, AvgHITS_2nd_eigenvector)
from baseline import baseline
from synthetic import (generator, save_data, load_data)

import numpy as np
from scipy.stats import (spearmanr, rankdata)
import matplotlib.pyplot as plt
from func_timeout import func_timeout, FunctionTimedOut

def accuracy(gold, a):
    """
    Compare the correct answers and answers given by methods to return the accuracy of each method

    Parameters
    ----------
    gold : array
        An array to store the correct answer for each question
    a : matrix (2-dimensional array) / array
        A matrix to store the option correctness
        For most methods, a is a matrix.
        Among all the options of one question i, if a[i][k] is the largest in a[i], k is considered to be the answer.
        However, for some methods like GKM, a is an array because these methods directly give the answer of each question instead of option correctness.
    
    Returns
    -------
    accuracy : float
        The accuracy of this method
    
    """
    correct = 0.0
    if len(np.shape(a)) == 2:
        N, K = np.shape(a)
        for i in range(0, N):
            ans = np.argmax(a[i])
            if ans == gold[i]:
                correct += 1
    else:
        N = len(a)
        for i in range(0, N):
            if (a[i] == 1 and gold[i] == 0) or (a[i] == -1 and gold[i] == 1):
                correct += 1
    accuracy = correct / N
    return accuracy

def user_accuracy(theta, s):
    """
    Compare the generated user abilities and estimated user abilities given by methods to return the user accuracy of each method

    Parameters
    ----------
    theta : array
        An array to store the generated student abilities by our synthetic data generator
    s : array
        An array to store the estimated student abilities by one method

    Returns
    -------
    accuracy : float
        The user accuracy of this method
    
    """
    accuracy = spearmanr(s,theta)[0]
    return accuracy

def user_difference(theta, s, start, end):
    """
    Compare the generated user abilities and estimated user abilities given by methods to return the average ranking difference of each student

    Parameters
    ----------
    theta : array
        An array to store the generated student abilities by our synthetic data generator
    s : array
        An array to store the estimated student abilities by one method
    start : int
        The first student we want to compare
    end : int
        The last student we want to compare

    Returns
    -------
    difference : float
        The average ranking difference of each student
    
    """
    gold_ranking = rankdata(theta)[start:end]
    r = rankdata(s)[start:end]
    # return np.mean(abs(gold_ranking-r)
    r1 = rankdata(-s)[start:end]
    return min(np.mean(abs(gold_ranking-r)), np.mean(abs(gold_ranking-r1)))

def experiment(M, N, K, b, a, model, loading, name, p = 1.0, ability = False):
    """
    experiment

    Parameters
    ----------
    M : int
        user number
    N : int
        question number
    K : int
        option number for each question
    b : double
        [b-1, b] is the range of parameter b of the IRT models (indicating how difficult this test is)
    a : double
        [0, a] is the range of parameter a of the IRT models
    model : string
        the model according to which we generate the data
    loading : bool
        An indicator of whether we have already generated the data
    name : string
        The experiment name
    p : double
        The probability of whether one question is answered by one student.
        By default, it is 1.0, meaning all students answer all questions.
        The sign of p means two different patterns, both meaning with probability |p|.
        When it is in (0,1), it means all students answer the same number of questions N * p.
        When it is in (-1,0),  it means each student has the probability |p| to answer each question, which means every student may answer different number of questions.
    ability : bool
        An indicator of whether we use the existing abilities
    -------

    Returns
    -------
    result : array
        An array of the result of this experiment.
        In the sequence of HnD, ABH, HITS, TF, Investment, PInvestment, Baseline
    
    """
    
    if loading:
        Labels, theta, gold, discrimination = load_data(name)
    else:
        Labels, theta, gold, discrimination = generator(M, N, K, b, a, model)
        save_data(Labels, theta, gold, name, discrimination)

    if p != 1:
        if p > 0:
            for i in Labels:
                non_answers = np.random.choice(N, int(N * (1 - p)), False)
                i[non_answers] = -1
        elif p < 0:
            p = -p
            for i in Labels:
                for j in range(0,N):
                    if np.random.random_sample() > p:
                        i[j] = -1

    Labels = Labels.astype(int)

    correct = np.zeros(M)
    for k in range(0, N):
        for j in range(0, int(M)):
            if Labels[j][k] == gold[k]:
                correct[j] += 1
    avgcorrect = correct.sum() / M / N * 100
    print ("The average accuracy of students: " + str(avgcorrect))

    if name.find("accuracy") != -1:
        result = np.zeros(8)

        if ability:
            print ("We have abilities")
            abilities = np.load("experiments/ability/" + name +".npy")
            for i in range(0,8):
                result[i] = user_accuracy(abilities[0], abilities[i + 1])
        else:
            s_HITSnDIFFS = HITSnDIFFS(Labels)[0]

            s_ABH = ABH(Labels)[0]
            s_hits = hits(Labels)[0]
            s_truthfinder = truthfinder(Labels)[0]
            s_investment = investment(Labels)[0]
            s_pooledinvestment = pooledinvestment(Labels)[0]
            s_baseline = baseline(Labels, gold)
            s_grm = grm_estimator(Labels, discrimination, model)[0]

            result[0] = user_accuracy(theta, s_HITSnDIFFS)
            result[1] = user_accuracy(theta, s_ABH)
            result[2] = user_accuracy(theta, s_hits)
            result[3] = user_accuracy(theta, s_truthfinder)
            result[4] = user_accuracy(theta, s_investment)
            result[5] = user_accuracy(theta, s_pooledinvestment)
            result[6] = user_accuracy(theta, s_baseline)
            result[7] = user_accuracy(theta, s_grm)

            abilities = np.reshape(np.concatenate((theta, s_HITSnDIFFS, s_ABH, s_hits, 
                s_truthfinder, s_investment, s_pooledinvestment, s_baseline, s_grm)), (9, M))
            np.save("experiments/ability/"+name, abilities)
        
        return result, avgcorrect
    else:
        result = np.zeros(6)
        iter = np.zeros(3)
        try:
            result[0], iter[0] = func_timeout(1000, HITSnDIFFS, (Labels,))[1:]
        except FunctionTimedOut:
            result[0] = 1000
        try:
            result[1] = func_timeout(1000, AvgHITS_2nd_eigenvector, (Labels,))[1]
        except FunctionTimedOut:
            result[1] = 1000
        try:
            result[2], iter[1] = func_timeout(1000, AvgHITS_2nd_eigenvector_deflation, (Labels,))[1:]
        except FunctionTimedOut:
            result[2] = 1000
        try:
            result[3] = func_timeout(1000, ABH, (Labels,))[1]
        except FunctionTimedOut:
            result[3] = 1000
        try:
            result[4], iter[2] = func_timeout(1000, ABH_power_iteration, (Labels,))[1:]
        except FunctionTimedOut:
            result[4] = 1000
        try:
            result[5] = func_timeout(1000, grm_estimator, (Labels, discrimination, model))[1]
        except FunctionTimedOut:
            result[5] = 1000
    
        return result, iter

def plot(x, y, xlabel, ylabel, name, loglog=0, show=False):
    """
    Plot the result figure

    Parameters
    ----------
    x : array
        An array of the x axis value
    y : 2 dimensional array
        An array of the y axis values of all approaches.
        In the sequence of HnD, ABH, HITS, TF, Investment, PInvestment, Baseline
    xlabel : string
        The label of the x axis
    ylabel : string
        The label of the y axis
    name : string
        The experiment name
    log : int
        An indicator of whether the axes needs to be logarithmic
        0 for linear, 1 for logarithmic x axis, 2 for logarithmic both axes
    -------
    
    """
    if loglog == 2:
        y[y == 1000] = np.nan
        plt.grid(True)
        if name.find("Q") != -1:
            plt.loglog(x, y[5], marker = 'p', color = "gray", label = "GRM-estimator")
            plt.loglog(x, y[4], marker = '.', color = "blue", label = "ABH-Power", alpha = 0.5)
            plt.loglog(x, y[2], marker = 'D', color = "orange", label = "HnD-Deflation", alpha = 0.5)
            plt.loglog(x, y[0], marker = 's', color = "orange", label = "HnD-Power")
            plt.loglog(x, y[1], marker = '^', color = "orange", label = "HnD-Direct", alpha = 0.5)
            plt.loglog(x, y[3], marker = 'o', color = "blue", label = "ABH-Direct")
            plt.loglog(x, np.array(x)/10000, marker = '', color = "black", linestyle="--")
            quadratic = (np.array(x) * np.array(x))/100000
            quadratic[quadratic > 1000] = np.nan
            plt.loglog(x, quadratic, marker = '', color = "black", linestyle="-.")
        elif name.find("U") != -1:
            plt.loglog(x, y[5], marker = 'p', color = "gray", label = "GRM-estimator")
            plt.loglog(x, y[4], marker = '.', color = "blue", label = "ABH-Power", alpha = 0.5)
            plt.loglog(x, y[3], marker = 'o', color = "blue", label = "ABH-Direct")
            plt.loglog(x, y[1], marker = '^', color = "orange", label = "HnD-Direct", alpha = 0.5)
            plt.loglog(x, y[2], marker = 'D', color = "orange", label = "HnD-Deflation", alpha = 0.5)
            plt.loglog(x, y[0], marker = 's', color = "orange", label = "HnD-Power")
            plt.loglog(x, np.array(x)/10000, marker = '', color = "black", linestyle="--")
            quadratic = (np.array(x) * np.array(x))/100000
            quadratic[quadratic > 1000] = np.nan
            plt.loglog(x, quadratic, marker = '', color = "black", linestyle="-.")
        elif name.find("B") != -1:
            plt.loglog(x, y[5], marker = 'p', color = "gray", label = "GRM-estimator")
            plt.loglog(x, y[4], marker = '.', color = "blue", label = "ABH-Power", alpha = 0.5)
            plt.loglog(x, y[3], marker = 'o', color = "blue", label = "ABH-Direct")
            plt.loglog(x, y[1], marker = '^', color = "orange", label = "HnD-Direct", alpha = 0.5)
            plt.loglog(x, y[2], marker = 'D', color = "orange", label = "HnD-Deflation", alpha = 0.5)
            plt.loglog(x, y[0], marker = 's', color = "orange", label = "HnD-Power")
            quadratic = (np.array(x) * np.array(x))/100000
            quadratic[quadratic > 1000] = np.nan
            plt.loglog(x, quadratic, marker = '', color = "black", linestyle="-.")
            cubic = np.array(x) / 1000000 * np.array(x) * np.array(x)
            cubic[cubic > 1000] = np.nan
            plt.loglog(x, cubic, marker = '', color = "black", linestyle="--")
    elif loglog == 1:
        plt.xscale("log", base=2)
        plt.xticks(x, x)
        if name.find("C1P") != -1:
            plt.ylim(0,1.02)
        else:
            plt.ylim(0,1)
        plt.plot(x, y[1], marker = 'o', label = "ABH")
        plt.plot(x, y[0], marker = 's', label = "HnD", linewidth = 2, zorder = 100)
        plt.plot(x, y[2], marker = 'X', label = "HITS")
        plt.plot(x, y[3], marker = 'D', label = "TruthFinder")
        plt.plot(x, y[4], marker = '+', label = "Invest")
        plt.plot(x, y[5], marker = 'P', label = "PooledInv")
        plt.plot(x, y[6], marker = '*', label = "True-Answer", linestyle="--")
        plt.plot(x, y[7], marker = 'p', label = "GRM-estimator", linestyle="--")
    else:
        if name.find("_O") != -1:
            plt.ylim(0,1)
            plt.xticks(x, x)
        elif name.find("_D") != -1:
            plt.ylim(0,1)
        plt.plot(x, y[1], marker = 'o', label = "ABH",)
        plt.plot(x, y[0], marker = 's', label = "HnD", linewidth = 2, zorder = 100)
        plt.plot(x, y[2], marker = 'X', label = "HITS")
        plt.plot(x, y[3], marker = 'D', label = "TruthFinder")
        plt.plot(x, y[4], marker = '+', label = "Invest")
        plt.plot(x, y[5], marker = 'P', label = "PooledInv")
        plt.plot(x, y[6], marker = '*', label = "True-Answer", linestyle="--")
        plt.plot(x, y[7], marker = 'p', label = "GRM-estimator", linestyle="--")

    plt.xlabel(xlabel, fontsize="xx-large")
    plt.ylabel(ylabel, fontsize="xx-large")
    plt.xticks(fontsize = "x-large")
    plt.yticks(fontsize = "x-large")
    plt.text(15000, 0.8, "Linear", va = "center", fontweight="bold")
    plt.text(3800, 100, "Quadratic", va = "center", fontweight="bold")
    # plt.legend(fontsize = "x-large", labelspacing = 0.25, ncol = 1, columnspacing = 0.5, framealpha = 0.5)

    plt.savefig("experiments/figures/Figure_" + name + ".pdf", bbox_inches='tight')
    if show:
        plt.show()
    plt.clf()