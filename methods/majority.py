"""
Provide the implementations of majority vote
"""
import numpy as np

def majority(Labels):
    """
    The implementation of majority vote

    Parameters
    ----------
    Labels : matrix (2-dimensional array)
        A matrix to store all the annotations for all questions from all students
        Labels[j][i] = -1 means student j didn't answer question i
        Labels[j][i] = k, k = 0, 1, ..., means student j chose option k for question i

    Returns
    -------
    a : 2-dimensional array
        an array to store the option qualities obtained by majority vote
    """
    M, N = np.shape(Labels)
    K = int(Labels.max(1).max() + 1)
    a = np.zeros([N,K])
    for i in range(0, N):
        ans_count = 0
        for j in range(0, M):
            if Labels[j][i] != -1:
                a[i][int(Labels[j][i])] += 1
                ans_count += 1
        for k in range(0, K):
            a[i][k] /= ans_count

    return a