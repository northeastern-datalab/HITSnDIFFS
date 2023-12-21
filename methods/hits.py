"""
Provide the implementations of hits-style methods including hits, Investment, PooledInvestment, TruthFinder and pre-processing functions.
All these methods iterate between student ability and option quality.
"""

import numpy as np
import math

def initiate(Labels):
    """
    Get the initial inputs from the label matrix which stores all the annotations for all questions from all students

    Parameters
    ----------
    Labels : matrix (2-dimensional array)
        A matrix to store all the annotations for all questions from all students
        Labels[j][i] = -1 means student j didn't answer question i
        Labels[j][i] = k, k = 0, 1, ..., means student j chose option k for question i
    
    Returns
    -------
    M : int
        student number
    N : int
        question number
    K : int
        option number for each question
    s : array
        an array to store all student abilities
    a : 2-dimensional array
        an array to store all option qualities with priors set to be uniform
    """
    M, N = np.shape(Labels)
    K = int(Labels.max(1).max() + 1)
    s = np.random.random((M,))
    a = np.zeros((N, K))
    for i in range(N):
        for j in range(K):
            a[i][j] = 1 / K
    return M, N, K, s, a

def getLists(Labels, M, N, K):
    """
    Get the lists for iterations. These two lists store the options chosen by each student and the students who chose each option

    Parameters
    ----------
    Labels : matrix (2-dimensional array)
        A matrix to store all the annotations for all questions from all students
        Labels[j][i] = -1 means student j didn't answer question i
        Labels[j][i] = k, k = 0, 1, ..., means student j chose option k for question i
    M : int, student number
    N : int, question number
    K : int, option number for each question

    Returns
    -------
    s_list : list
        A list to store the options chosen by each student
    a_list : list
        A list to store the students who chose each option
    """
    s_list = []
    a_list = []
    for i in range(0, N * K):
        l = []
        a_list.append(l)
    for j in range(0, M):
        l = []
        for i in range(0, N):
            if Labels[j][i] != -1:
                l.append(int(i * K + Labels[j][i]))
                a_list[int(i * K + Labels[j][i])].append(j)
        s_list.append(l)
    return s_list, a_list

def hits(Labels, tol=1e-5):
    """
    The implementation of HITS Algorithm
    Jon M Kleinberg. 1999. Authoritative sources in a hyperlinked environment. JACM 46, 5 (1999), 604-632.

    Parameters
    ----------
    Labels : matrix (2-dimensional array)
        A matrix to store all the annotations for all questions from all students
        Labels[j][i] = -1 means student j didn't answer question i
        Labels[j][i] = k, k = 0, 1, ..., means student j chose option k for question i

    Returns
    -------
    s : array
        an array to store the student abilities after ieratrions
    a : 2-dimensional array
        an array to store the option qualities after iterations
    """
    M, N, K, s, a = initiate(Labels)
    
    s_list, a_list = getLists(Labels, M, N, K)

    previous_s = np.ones((M,))

    while np.sum(abs((previous_s - s)/previous_s)) > tol:
        previous_s = np.copy(s)
        for j in range(0, M):
            s[j] = 0
            for i in s_list[j]:
                s[j] += a[int(i / K)][i % K]
        s /= s.max()
        for i in range(0, N):
            for k in range(0, K):
                a[i][k] = 0
                for j in a_list[i * K + k]:
                    a[i][k] += s[j]
            a[i] /= a[i].sum()

    return s, a

def truthfinder(Labels, tol=1e-5):
    """
    The implementation of TruthFinder Algorithm
    X. Yin, J. Han, and P. S. Yu. Truth discovery with multiple conflicting information providers on the web. In Proc. of the ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD'07), pages 1048-1052, 2007.

    Parameters
    ----------
    Labels : matrix (2-dimensional array)
        A matrix to store all the annotations for all questions from all students
        Labels[j][i] = -1 means student j didn't answer question i
        Labels[j][i] = k, k = 0, 1, ..., means student j chose option k for question i

    Returns
    -------
    s : array
        an array to store the student abilities after ieratrions
    a : 2-dimensional array
        an array to store the option qualities after iterations
    """
    M, N, K, s, a = initiate(Labels)
    
    s_list, a_list = getLists(Labels, M, N, K)

    previous_s = np.ones((M,))

    while np.sum(abs((previous_s - s)/previous_s)) > tol:
        previous_s = np.copy(s)
        for j in range(0, M):
            s[j] = 0
            for i in s_list[j]:
                s[j] += a[int(i / K)][i % K]
            s[j] /= len(s_list[j])
        for i in range(0, N):
            for k in range(0, K):
                t = 1
                for j in a_list[i * K + k]:
                    t *= (1 - s[j])
                a[i][k] = 1 - t
            a[i] /= a[i].sum()

    return s, a

def investment(Labels, iterations = 10):
    """
    The implementation of Investment Algorithm
    J. Pasternack and D. Roth. Knowing what to believe (when you already know something). In Proc. of the International Conference on Computational Linguistics (COLING'10), pages 877-885, 2010.

    Parameters
    ----------
    Labels : matrix (2-dimensional array)
        A matrix to store all the annotations for all questions from all students
        Labels[j][i] = -1 means student j didn't answer question i
        Labels[j][i] = k, k = 0, 1, ..., means student j chose option k for question i

    Returns
    -------
    s : array
        an array to store the student abilities after ieratrions
    a : 2-dimensional array
        an array to store the option qualities after iterations
    """
    M, N, K, s, a = initiate(Labels)
    
    s_list, a_list = getLists(Labels, M, N, K)

    for i in range(0, N):
        for k in range(0, K):
            a[i][k] = np.size(a_list[i * K + k]) / M

    previous_s = np.ones((M,))

    for i in range(0,iterations):
        for j in range(0, M):
            s[j] = 0
            for i in s_list[j]:
                investment_sum = 0
                for investment_j in a_list[i]:
                    investment_sum += previous_s[investment_j] / len(s_list[investment_j])
                investment_one = previous_s[j] / len(s_list[j])
                s[j] += a[int(i / K)][i % K] *  investment_one / investment_sum
        previous_s = np.copy(s)
        for i in range(0, N):
            for k in range(0, K):
                a[i][k] = 0
                for j in a_list[i * K + k]:
                    a[i][k] += s[j] / len(s_list[j])
                a[i][k] = a[i][k] ** 1.2

    return s, a

def pooledinvestment(Labels, iterations = 10):
    """
    The implementation of Pooled Investment Algorithm
    J. Pasternack and D. Roth. Knowing what to believe (when you already know something). In Proc. of the International Conference on Computational Linguistics (COLING'10), pages 877-885, 2010.

    Parameters
    ----------
    Labels : matrix (2-dimensional array)
        A matrix to store all the annotations for all questions from all students
        Labels[j][i] = -1 means student j didn't answer question i
        Labels[j][i] = k, k = 0, 1, ..., means student j chose option k for question i

    Returns
    -------
    s : array
        an array to store the student abilities after ieratrions
    a : 2-dimensional array
        an array to store the option qualities after iterations
    """
    M, N, K, s, a = initiate(Labels)
    
    s_list, a_list = getLists(Labels, M, N, K)

    def H(i, k):
        h = 0
        for j in a_list[i * K + k]:
            h += s[j] / len(s_list[j])
        return h

    previous_s = np.ones((M,))

    for i in range(0,iterations):
        for j in range(0, M):
            s[j] = 0
            for i in s_list[j]:
                investment_sum = 0
                for investment_j in a_list[i]:
                    investment_sum += previous_s[investment_j] / len(s_list[investment_j])
                investment_one = previous_s[j] / len(s_list[j])
                s[j] += a[int(i / K)][i % K] *  investment_one / investment_sum
        previous_s = np.copy(s)
        for i in range(0, N):
            for k in range(0, K):
                a[i][k] = 0
                mutual_sum = 0
                for mutual_k in range(0, K):
                    mutual_sum += (H(i, mutual_k) ** 1.4)
                a[i][k] = H(i, k) * (H(i, k) ** 1.4) / mutual_sum

    return s, a