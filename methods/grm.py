"""
The GRM estimator from the girth package
"""

import numpy as np
import time
from girth import grm_mml

def grm_estimator(Labels, discrimination, model):
    """
    Estimate the student abilities based on the Graded Response Model (GRM)

    Parameters
    ----------
    Labels : matrix (2-dimensional array)
        A matrix to store all the annotations for all questions from all students
        Labels[j][i] = -1 means student j didn't answer question i
        Labels[j][i] = k, k = 0, 1, ..., means student j chose option k for question i
    discrimination : matrix (2-dimensional array)
        A matrix to store all the discrimination parameters of the Bock/Samejima model.
        In Bock/Samejima model, this parameter indicates the correctness of the options.
        Here, we use the parameters to sort the Labels by correctness.
    model : string
        the model according to which we generate the data
    
    Returns
    -------
    s : array
        an array to store the student abilities
    """
    input = Labels.T
    if model == "Bock" or model == "Samejima":
        N, K = np.shape(discrimination)
        order = np.argsort(discrimination, axis=1)
        input += K
        for i in range(0,N):
            for k in range(0,K):
                input[i][input[i] == (k + K)] = order[i][k]
        input += 1 

    start = time.perf_counter()
    estimates = grm_mml(input)
    end = time.perf_counter()

    return estimates['Ability'], end - start