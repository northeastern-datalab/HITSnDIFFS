"""
This file aims to conduct experiments to compare different methods on synthetic datasets.
We're generating the synthetic datasets based on different models.
"""
import numpy as np
from girth.synthetic import create_synthetic_irt_polytomous

def save_data(Labels, theta, gold, name, discrimination):
    """
    Save the generated dataset

    Parameters
    -------
    Labels : matrix (2-dimensional array)
        A matrix to store all the annotations for all questions from all students
        Labels[j][i] = -1 means student j didn't answer question i
        Labels[j][i] = k, k = 0, 1, ..., means student j chose option k for question i
    theta : array
        An array to store the generated student abilities by our synthetic data generator
    gold : array
        An array to store the correct answer for each question
    name : string
        the data file name
    discrimination : matrix (2-dimensional array)
        A matrix to store all the discrimination parameters of the Bock/Samejima model
    
    """

    np.savez("experiments/synthetic/" + name, Labels = Labels, theta = theta, 
            gold = gold, discrimination = discrimination)

def load_data(name):
    """
    Load a generated dataset

    Parameters
    -------
    name : string
        The file name

    Returns
    -------
    Labels : matrix (2-dimensional array)
        A matrix to store all the annotations for all questions from all students
        Labels[j][i] = -1 means student j didn't answer question i
        Labels[j][i] = k, k = 0, 1, ..., means student j chose option k for question i
    theta : array
        An array to store the generated student abilities by our synthetic data generator
    gold : array
        An array to store the correct answer for each question
    discrimination : matrix (2-dimensional array)
        A matrix to store all the discrimination parameters of the Bock/Samejima model
    
    """

    dataset = np.load("experiments/synthetic/" + name + ".npz")
    return dataset["Labels"], dataset["theta"], dataset["gold"], dataset["discrimination"]

def generator(M, N, K, difficulty, discrimination, model, theta_distribution="uniform"):
    """
    Synthetic data generator

    Parameters
    ----------
    M : int
        student number
    N : int
        question number
    K : int
        option number for each question
    difficulty : double
        [difficulty-1, difficulty] is the range of parameter b of the IRT models 
        indicating how difficult this test is
    discrimination : double
        [0, discrimination] is the range of parameter a of the IRT models
    model : string
        the model according to which we generate the data
    theta_distribution : string
        the distribution of student abilities
        
    -------
    Labels : matrix (2-dimensional array)
        A matrix to store all the annotations for all questions from all students
        Labels[j][i] = -1 means student j didn't answer question i
        Labels[j][i] = k, k = 0, 1, ..., means student j chose option k for question i
    theta : array
        An array to store the generated student abilities by our synthetic data generator
    gold : array
        An array to store the correct answer for each question
    a : matrix (2-dimensional array)
        A matrix to store all the discrimination parameters of the Bock/Samejima model.
        If the generative model is not Bock/Samejima, return 0
    """
    Labels = np.zeros((M, N))
    gold = np.zeros(N)
    a = 0

    if model == "Bock":
        # Multi-2PL/Bock model
        if theta_distribution == "uniform":
            theta = np.random.uniform(0,1,M)
        elif theta_distribution == "normal":
            theta = np.random.normal(0,1/3,M)
        elif theta_distribution == "equispaced":
            theta = np.linspace(1.0/M,1.0,M)

        if theta_distribution == "equispaced":
            b = np.linspace(difficulty/M-0.5,1.0-difficulty/M-0.5,N)
            b = b.reshape(M,1).dot(np.ones(K).reshape(1,K))
            a = (np.ones(N) * discrimination).reshape(N,1).dot(np.linspace(1.0/K,1.0,K).reshape(1,K))
        else:
            b = np.random.uniform(difficulty - 1,difficulty,(N,K))
            a = np.random.uniform(0,discrimination,(N,K))
        gold = np.argmax(a, axis=1)

        p = np.exp(a[:, :, None] * (theta[None, :] - b[:, :, None])) # N * K * M
        sum = np.sum(p, axis = 1) # N * M
        p = (p.swapaxes(0,1) / sum) # K * N * M

        option = np.cumsum(p, axis=0) - np.random.uniform(0,1,(N,M)) # K * N * M
        option[option < 0] = 1
        Labels = np.argmin(option, axis=0).T # M * N
    
    elif model == "Samejima":
        # Samejima model with the similar idea as 3PL
        if theta_distribution == "uniform":
            theta = np.random.uniform(0,1,M)
        elif theta_distribution == "normal":
            theta = np.random.normal(0,1/3,M)

        b = np.random.uniform(difficulty - 1,difficulty,(N,K + 1))
        a = np.random.uniform(0,discrimination,(N,K + 1))
        a[:,K] = -(np.sum(a, axis=1) - a[:,K])
        b[:,K] = -(np.sum(a * b, axis=1) - a[:,K] * b[:,K]) / a[:,K]
        gold = np.argmax(a, axis=1)

        p = np.exp(a[:, :, None] * (theta[None, :] - b[:, :, None])) # N * (K + 1) * M
        sum = np.sum(p, axis = 1) # N * M
        p = (p.swapaxes(0,1) + p[:,K,:] / K)[:-1,:,:] / sum # K * N * M

        option = np.cumsum(p, axis=0) - np.random.uniform(0,1,(N,M)) # K * N * M
        option[option < 0] = 1
        Labels = np.argmin(option, axis=0).T # M * N

    elif model == "C1P":
        # ideal C1P model
        if theta_distribution == "uniform":
            theta = np.random.uniform(0,1/2,int(M / 10))
            theta2 = np.random.uniform(1/2,1,M - int(M / 10))
            theta = np.append(theta, theta2)
        elif theta_distribution == "normal":
            theta = np.random.normal(0,1/3,M)
        
        # print (theta)
        theta = np.sort(theta)
        
        d = np.random.uniform(0,1,(N,K-1))
        d = np.sort(d)
        gold[:] = K - 1
        
        for j in range(0, M):
            for i in range(0, N):
                Labels[j][i] = 0
                for k in range(0, K - 1):
                    if theta[j] > d[i][k]:
                        Labels[j][i] = k + 1
                    else:
                        break

    elif model == "GRM":
        # Homogeneous graded response model
        if theta_distribution == "uniform":
            theta = np.random.uniform(0,1,M)
        elif theta_distribution == "normal":
            theta = np.random.normal(0,1/3,M)

        b = np.random.uniform(difficulty - 1,difficulty,(N,K-1))
        b = np.sort(b, axis = 1)
        a = np.random.uniform(0,discrimination * 2 /(K+1),N)
        gold[:] = K - 1
        
        Labels = create_synthetic_irt_polytomous(b, a, theta, "grm").T
        Labels -= 1
    
    elif model == "3PL-Halfmoon":
        # 3PL model used for simulating real data distribution of a half moon shape
        theta = np.random.normal(0,1,M)

        b = np.random.uniform(-2,3,N)
        loga = np.zeros(N)
        for i in range(0,N):
            loc = (abs(b[i] - 0.5) / 2.5) * (abs(b[i] - 0.5) / 2.5) * 0.4
            s = 0.2 - 0.1 * (abs(b[i] - 0.5) / 2.5)
            loga[i] = np.random.normal(loc,s)
        a = np.power(10,loga)
        c = np.random.uniform(0,0.5,N)

        p = c[:, None] + (1 - c[:, None]) / (1 + np.exp(-a[:, None] * (theta[None, :] - b[:,None]))) # N * M
        p = p - np.random.uniform(0,1,(N,M))
        p[p >= 0] = 1
        p[p < 0] = 0
        Labels = p.T
        gold = gold + 1
        a = np.zeros((N,2))
        a[:,1] += 1
    
    elif model == "3PL":
        # 3PL model used for simulating real data distribution
        theta = np.random.normal(0,1,M)

        a = np.array([0.82, 0.57, 0.81, 0.68, 0.69, 0.77, 1.14, 1.42, 0.79, 0.59, 0.79, 0.48, 0.89, 0.8, 0.82, 0.7, 1.01, 0.6, 0.83, 0.64, 0.82, 0.76, 0.89, 
        0.95, 0.69, 0.74, 0.75, 0.77, 1.07, 0.75, 0.56, 0.71, 0.68, 0.9, 0.82, 0.9, 0.62, 0.46, 0.54, 0.67]) * 1.7
        b = np.array([-1.47, 0.11, -0.75, 1.15, 0.18, -1.1, 0, 0.05, -1.53, -0.59, 0.69, -2.11, -0.55, -0.43, 0.15, -0.45, 0.47, -0.15, -2.11, -1.87, -0.01, 
        -1.12, -0.62, 0.01, 0.08, 0.05, -0.92, -0.03, -1.31, -0.04, 0.38, -1.58, 0.29, 0.07, 0.01, 0.99, 0.68, -2.06, 0.44, 0.68])
        c = np.array([0.15, 0.26, 0.26, 0.15, 0.26, 0.12, 0.32, 0.28, 0.37, 0.28, 0.14, 0.21, 0.29, 0.33, 0.22, 0.25, 0.25, 0.24, 
        0.14, 0.19, 0.21, 0.25, 0.2, 0.19, 0.23, 0.15, 0.23, 0.21, 0.17, 0.23, 0.15, 0.21, 0.23, 0.15, 0.12, 0.18, 0.24, 0.18, 0.17, 0.26])
        
        p = c[:, None] + (1 - c[:, None]) / (1 + np.exp(-a[:, None] * (theta[None, :] - b[:,None]))) # N * M
        p = p - np.random.uniform(0,1,(N,M))
        p[p >= 0] = 1
        p[p < 0] = 0
        Labels = p.T
        gold = gold + 1
        a = np.zeros((N,2))
        a[:,1] += 1

    return Labels.astype(int), theta, gold, a