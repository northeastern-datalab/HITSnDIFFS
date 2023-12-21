"""
Provide the implementations of ABH, pre-processing functions and an entropy-based symmetry-breaking approach for both HnD and ABH
"""

import numpy as np
from scipy.sparse.linalg import eigsh
import time
from scipy.stats import entropy

def order_vs_reverse(s, C, Labels):
    """
    To decide to use one ranking or its reverse

    Parameters
    ----------
    s : array
        An array to store the estimated student abilities by one method
    Labels : matrix (2-dimensional array)
        A matrix to store all the annotations for all questions from all students
        Labels[j][i] = -1 means student j didn't answer question i
        Labels[j][i] = k, k = 0, 1, ..., means student j chose option k for question i
    C : matrix (2-dimensional array)
        A matrix to store all the annotations for all options from all students
        C[j][k] = 1 means student j chose option k, otherwise 0
    -------

    Returns
    -------
    s : array
        An array to store the estimated student abilities by one method
    
    """
    m, n = np.shape(Labels)
    k = int(Labels.max(1).max() + 1)

    epsilon = 1e-6
    
    r = int(m / 10)
    C_tens = np.reshape(C.T,[n,k,m])
    C_tens = C_tens.transpose(0,2,1)
    r_best_fwd_idx = np.argpartition(s, -r)[-r:]
    r_best_rev_idx = np.argpartition(-s, -r)[-r:]
    total_fwd_entropy = 0
    total_rev_entropy = 0
    for i in range(n):
        raw_val_fwd = np.sum(C_tens[i][r_best_fwd_idx],0) + epsilon
        raw_val_rev = np.sum(C_tens[i][r_best_rev_idx],0) + epsilon
        prob_fwd = np.divide(raw_val_fwd,np.sum(raw_val_fwd))
        prob_rev = np.divide(raw_val_rev,np.sum(raw_val_rev))
        total_fwd_entropy += entropy(prob_fwd)
        total_rev_entropy += entropy(prob_rev)
    if total_rev_entropy < total_fwd_entropy:
        s = -s
    return s

def getCMatrix(Labels):
    """
    Get the C matrix (m,nk) (student-option matrix) from the student-question matrix.
    What is a C matrix can be found in our paper.

    Parameters
    ----------
    Labels : matrix (2-dimensional array)
        A matrix to store all the annotations for all questions from all students
        Labels[j][i] = -1 means student j didn't answer question i
        Labels[j][i] = k, k = 0, 1, ..., means student j chose option k for question i
    
    Returns
    -------
    C : matrix (2-dimensional array)
        A matrix to store all the annotations for all options from all students
        C[j][k] = 1 means student j chose option k, otherwise 0
    """
    K = int(np.amax(Labels) + 1) 
    C_tensor = np.eye(K + 1)[(Labels + 1).T]
    return np.concatenate(np.delete(C_tensor, 0, axis = 2), axis=1)

def ABH(Labels):
    """
    The implementation of ABH algorithm by using scipy
    Jonathan E Atkins, Erik G Boman, and Bruce Hendrickson. 1998. A spectral algorithm for seriation and the consecutive ones problem. SIAM J. Comput. 28, 1 (1998), 297-310.
    This approach gets the second smallest eigenvector of matrix D - C(C.T) as the student abilities and is able to recover a C1P matrix.
    For eigenvector computation, 
    we use scipy.sparse package which utilizes the Lanczos algorithm 

    Parameters
    ----------
    Labels : matrix (2-dimensional array)
        A matrix to store all the annotations for all questions from all students
        Labels[j][i] = -1 means student j didn't answer question i
        Labels[j][i] = k, k = 0, 1, ..., means student j chose option k for question i
    
    Returns
    -------
    s : array
        An array to store the student abilities
    a : array
        An array to store the option correctness
    """
    C = getCMatrix(Labels)

    start = time.perf_counter()
    A = C.dot(C.T)
    L = np.diag(np.sum(A,1)) - A

    e = eigsh(L, k = 2, which = 'SM')
    s = e[1][:,np.argmax(e[0])]

    end = time.perf_counter()
    s = order_vs_reverse(s, C, Labels)

    return np.round(s, 8), end - start


def ABH_power_iteration(Labels, tol=1e-5, beta_coefficient = 1):
    """
    The implementation of ABH algorithm by computing the largest eigenvector of βI - S(D - C(C.T))T using the power method 
    when β is larger than all the entries and eigenvalues of S(D - C(C.T))T
    In this way, ABH can also use matrix-vector multiplication instead of matrix-matrix multiplication

    Parameters
    ----------
    Labels : matrix (2-dimensional array)
        A matrix to store all the annotations for all questions from all students
        Labels[j][i] = -1 means student j didn't answer question i
        Labels[j][i] = k, k = 0, 1, ..., means student j chose option k for question i

    Returns
    -------
    s : array
        An array to store the student abilities
    a : array
        An array to store the option correctness
    """
    C = getCMatrix(Labels)
    M, NK = np.shape(C)

    start = time.perf_counter()

    sdiff = np.random.uniform(0, 1, M-1)
    sdiff_new = np.random.uniform(0, 1, M-1)

    D = np.diag(np.dot(C, np.sum(C.T, axis=1)))
    # beta = np.max(D) * M
    beta = np.max(D) * beta_coefficient
    betaI = beta * np.identity(M-1)

    iter = 0
    while np.linalg.norm(sdiff_new - sdiff) > tol:
        iter += 1
        sdiff = np.copy(sdiff_new)
        s = np.zeros(M)
        s[1:] = np.cumsum(sdiff)
        w = C.T.dot(s)
        s = D.dot(s) - C.dot(w)

        sdiff_new = betaI.dot(sdiff) - np.diff(s)
        sdiff_new = sdiff_new / np.linalg.norm(sdiff_new)

    s = np.zeros(M)
    s[1:] = np.cumsum(sdiff)

    end = time.perf_counter()
    s = order_vs_reverse(s, C, Labels)

    return np.round(s, 8), end - start, iter