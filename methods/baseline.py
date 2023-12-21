"""
Provide the implementations of the baseline method with correct answers to rank students
"""
import numpy as np

def baseline(Labels, gold):
    """
    The implementation of the baseline method with correct answers to rank students

    Parameters
    ----------
    Labels : matrix (2-dimensional array)
        A matrix to store all the annotations for all questions from all students
        Labels[j][i] = -1 means student j didn't answer question i
        Labels[j][i] = k, k = 0, 1, ..., means student j chose option k for question i
    gold : array
        An array to store the correct answer for each question

    Returns
    -------
    s : array
        an array to store the student abilities
    """
    M, N = np.shape(Labels)
    s = np.zeros(M)
    for j in range(0, M):
        for i in range(0, N):
            if Labels[j][i] == gold[i]:
                s[j] += 1
    return s