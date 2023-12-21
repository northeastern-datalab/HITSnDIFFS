"""
This file contains all the major experiments.
"""
import sys
import numpy as np
import cpuinfo

from util import (experiment, plot)

def accuracy_question_numbers(model, loading=False):
    """
    The accuracy experiments with different question numbers
    Fixed student number 100
    Different question numbers 25, 50, 100, 200, 400, 800, 1600
    Each experiment set runs 10 times
    4 * 7 * 10 = 280 times of experiments

    Parameters
    ----------
    model : string
        the model according to which we generate the data
    loading : bool
        the indicator of whether to load the existing experimental result
    """
    M = 100
    N = [25, 50, 100, 200, 400, 800, 1600]
    K = 3
    b = 0.5
    a = 10

    name = "accuracy_" + model + "_Q"
    print (name)

    if loading:
        accuracy = np.load("experiments/result/" + name + ".npy")
        show = True
    else:
        times = 10
        models = 8
        show = False

        accuracy = np.zeros((models, len(N)))
        for i in range(0, len(N)):
            print (i)
            result = np.zeros(models)
            for j in range(0, times):
                returns = experiment(M, N[i], K, b, a, model, True, name + "_" + str(N[i]) + "_" + str(j + 1))
                result += returns[0]
            result /= times
            for j in range(0, models):
                accuracy[j][i] = result[j]

        np.save("experiments/result/" + name, accuracy)

    plot(N, accuracy, "Number of questions", "Accuracy of user ranking", name, 1, show)

def accuracy_user_numbers(model, loading=False):
    """
    The accuracy experiments with user numbers
    Fixed question numbers 100
    Different student numbers 25, 50, 100, 200, 400, 800, 1600
    Each experiment set runs 10 times
    3 * 7 * 10 = 210 times of experiments
    
    Parameters
    ----------
    model : string
        the model according to which we generate the data
    loading : bool
        the indicator of whether to load the existing experimental result
    """
    M = [25, 50, 100, 200, 400, 800, 1600]
    N = 100
    K = 3
    b = 0.5
    a = 10

    name = "accuracy_" + model + "_U"
    print(name)

    if loading:
        accuracy = np.load("experiments/result/" + name + ".npy")
        show = True
    else:
        times = 10
        models = 8
        show = False

        accuracy = np.zeros((models, len(M)))
        for i in range(0, len(M)):
            print (i)
            result = np.zeros(models)
            for j in range(0, times):
                returns = experiment(M[i], N, K, b, a, model, True, name + "_" + str(M[i]) + "_" + str(j + 1))
                result += returns[0]
            result /= times
            for j in range(0, models):
                accuracy[j][i] = result[j]

        np.save("experiments/result/" + name, accuracy)

    plot(M, accuracy, "Number of users", "Accuracy of user ranking", name, 1, show)

def accuracy_option_numbers(model, loading=False):
    """
    The accuracy experiments with different option numbers 2, 3, 4, 5, 6
    Fixed student number 100 and fixed question numbers 100
    Each experiment set runs 10 times
    3 * 5 * 10 = 150 times of experiments

    Parameters
    ----------
    model : string
        the model according to which we generate the data
    loading : bool
        the indicator of whether to load the existing experimental result
    """
    M = 100
    N = 100
    if model == "GRM":
        K = [3, 4, 5, 6, 7]
    else:
        K = [2, 3, 4, 5, 6]
    b = 0.5
    a = 10

    name = "accuracy_" + model + "_O"
    print (name)

    if loading:
        accuracy = np.load("experiments/result/" + name + ".npy")
        show = True
    else:
        times = 10
        models = 8
        show = False

        accuracy = np.zeros((models, len(K)))
        for i in range(0, len(K)):
            print (i)
            result = np.zeros(models)
            for j in range(0, times):
                returns = experiment(M, N, K[i], b, a, model, True, name + "_" + str(K[i]) + "_" + str(j + 1))
                result += returns[0]
            result /= times
            for j in range(0, models):
                accuracy[j][i] = result[j]

        np.save("experiments/result/" + name, accuracy)

    plot(K, accuracy, "Number of options", "Accuracy of user ranking", name, 0, show)

def accuracy_difficulty(model, loading=False):
    """
    The accuracy experiments with different difficulty ranges (parameter b of the IRT models)
    [0.5,1.5], [0.25,1.25], [0,1], [-0.25,0.75], [-0.5, 0.5], [-0.75, 0.25], [-1, 0]
    Fixed student number 100 and fixed question numbers 100
    Each experiment set runs 10 times
    3 * 7 * 10 = 210 times of experiments

    Parameters
    ----------
    model : string
        the model according to which we generate the data
    loading : bool
        the indicator of whether to load the existing experimental result
    """
    M = 100
    N = 100
    K = 3
    b = [1.5, 1.25, 1, 0.75, 0.5, 0.25, 0]
    a = 10

    name = "accuracy_" + model + "_D"
    print(name)

    if loading:
        reading = np.load("experiments/result/" + name + ".npz")
        accuracy = reading["accuracy"]
        student_avg = reading["stduent_avg"]
        show = True
    else:
        times = 10
        models = 8
        show = False

        accuracy = np.zeros((models, len(b)))
        student_avg = np.zeros(len(b))
        for i in range(0, len(b)):
            print (i)
            result = np.zeros(models)
            for j in range(0, times):
                returns = experiment(M, N, K, b[i], a, model, True, name + "_" + str(b[i]) + "_" + str(j + 1))
                result += returns[0]
                student_avg[i] += returns[1]
            result /= times
            student_avg[i] /= times
            for j in range(0, models):
                accuracy[j][i] = result[j]

        np.savez("experiments/result/" + name, accuracy = accuracy, stduent_avg = student_avg)

    plot(student_avg, accuracy, "Average user accuracy", "Accuracy of user ranking", name, 0, show)

def accuracy_discrimination(model, loading=False):
    """
    The accuracy experiments with different discrimination ranges (parameter a of the IRT models)
    [0,2.5], [0,5], [0,10], [0,20], [0,40]
    Fixed student number 100 and fixed question numbers 100
    Each experiment set runs 10 times
    3 * 5 * 10 = 150 times of experiments

    Parameters
    ----------
    model : string
        the model according to which we generate the data
    loading : bool
        the indicator of whether to load the existing experimental result
    """
    M = 100
    N = 100
    K = 3
    b = 0.5
    a = [2.5, 5, 10, 20, 40]

    name = "accuracy_" + model + "_A"
    print(name)

    if loading:
        accuracy = np.load("experiments/result/" + name + ".npy")
        show = True
    else:
        times = 10
        models = 8
        show = False

        accuracy = np.zeros((models, len(a)))
        for i in range(0, len(a)):
            print (i)
            result = np.zeros(models)
            for j in range(0, times):
                returns = experiment(M, N, K, b, a[i], model, True, name + "_" + str(a[i]) + "_" + str(j + 1))
                result += returns[0]
            result /= times
            for j in range(0, models):
                accuracy[j][i] = result[j]

        np.save("experiments/result/" + name, accuracy)

    plot(a, accuracy, "Question discrimination", "Accuracy of user ranking", name, 1, show)

def efficiency_question_numbers(model="GRM", loading=False):
    """
    The efficiency experiments with different question numbers
    Fixed student number 100 and 
    different question numbers 10, 32, 100, 320, 1000, 3200, 10000, 32000, 100000
    Each experiment set runs 5 time
    1 * 9 * 5 = 45 times of experiments

    Parameters
    ----------
    model : string
        the model according to which we generate the data
    loading : bool
        the indicator of whether to load the existing experimental result
    """
    M = 100
    N = [10, 32, 100, 320, 1000, 3200, 10000, 32000, 100000]
    K = 3
    b = 0.5
    a = 10

    name = "efficiency_" + model + "_Q"
    print (name)

    if loading:
        reading = np.load("experiments/result/" + name + ".npz")
        t = reading["t"]
        iter = reading["iter"]
        show = True
    else:
        times = 5
        models = 6
        show = False

        t = np.zeros((models, len(N)))
        iter = np.zeros((3, len(N)))
        for i in range(0, len(N)):
            print (i)
            result_t = np.zeros((models, times))
            result_iter = np.zeros((3, times))
            for j in range(0, times):
                print(j)
                returns = experiment(M, N[i], K, b, a, model, True, name + "_" + str(N[i]) + "_" + str(j + 1))
                result_t[:,j] = returns[0]
                result_iter[:,j] = returns[1]
            for j in range(0, models):
                t[j][i] = np.median(result_t[j])
            for j in range(0, 3):
                iter[j][i] = np.median(result_iter[j])

        np.savez("experiments/result/" + name, t = t, iter = iter)
        
    plot(N, t, "Number of questions", "Execution time (sec)", name, 2, show)

def efficiency_user_numbers(model="GRM", loading=False):
    """
    The efficiency experiments with different user numbers
    Fixed student number 100 and 
    different question numbers 10, 32, 100, 320, 1000, 3200, 10000, 32000, 100000
    Each experiment set runs 5 time
    1 * 9 * 5 = 45 times of experiments

    Parameters
    ----------
    model : string
        the model according to which we generate the data
    loading : bool
        the indicator of whether to load the existing experimental result
    """
    M = [10, 32, 100, 320, 1000, 3200, 10000, 32000, 100000]
    N = 100
    K = 3
    b = 0.5
    a = 10

    name = "efficiency_" + model + "_U"
    print (name)

    if loading:
        reading = np.load("experiments/result/" + name + ".npz")
        t = reading["t"]
        iter = reading["iter"]
        show = True
    else:
        times = 5
        models = 6
        show = False

        t = np.zeros((models, len(M)))
        iter = np.zeros((3, len(M)))
        for i in range(0, len(M)):
            print (i)
            result_t = np.zeros((models, times))
            result_iter = np.zeros((3, times))
            for j in range(0, times):
                print(j)
                returns = experiment(M[i], N, K, b, a, model, True, name + "_" + str(M[i]) + "_" + str(j + 1))
                result_t[:,j] = returns[0]
                result_iter[:,j] = returns[1]
                print(returns)
            for j in range(0, models):
                t[j][i] = np.median(result_t[j])
            for j in range(0, 3):
                iter[j][i] = np.median(result_iter[j])

        np.savez("experiments/result/" + name, t = t, iter = iter)
        
    plot(M, t, "Number of users", "Execution time (sec)", name, 2, show)

def accuracy_probability_different_number(model, loading=False):
    """
    The accuracy experiments with different probability for a student to answer a question when all students answer the different number of questions
    Different probabilities 0.6,0.7,0.8,0.9,1.0
    Fixed student number 100 and fixed question numbers 100
    Each experiment set runs 5 times
    1 * 5 * 10 = 50 times of experiments

    Parameters
    ----------
    model : string
        the model according to which we generate the data
    loading : bool
        the indicator of whether to load the existing experimental result
    """
    M = 100
    N = 100
    K = 3
    b = 0.5
    a = 10
    p = [0.6, 0.7, 0.8, 0.9, 1.0]

    name = "accuracy_" + model + "_PD"
    print(name)

    if loading:
        accuracy = np.load("experiments/result/" + name + ".npy")
        show = True
    else:
        times = 10
        models = 8
        show = False

        accuracy = np.zeros((models, len(p)))
        for i in range(0, len(p)):
            print (i)
            result = np.zeros(models)
            for j in range(0, times):
                returns = experiment(M, N, K, b, a, model, True, "accuracy_" + model + "_Q_100_" + str(j + 1), p[i])
                result += returns[0]
            result /= times
            for j in range(0, models):
                accuracy[j][i] = result[j]

        np.save("experiments/result/" + name, accuracy)

    plot(p, accuracy, "Probability", "Accuracy of user ranking", name, 1, show)

def main():
    info = cpuinfo.get_cpu_info()
    print(info)

    if sys.argv[1] == '1':
        accuracy_question_numbers("Bock")
    elif sys.argv[1] == '2':
        accuracy_question_numbers("Samejima")
    elif sys.argv[1] == '3':
        accuracy_question_numbers("GRM")
    elif sys.argv[1] == '4':
        accuracy_user_numbers("Bock")
    elif sys.argv[1] == '5':
        accuracy_user_numbers("Samejima")
    elif sys.argv[1] == '6':
        accuracy_user_numbers("GRM")
    elif sys.argv[1] == '7':
        accuracy_option_numbers("Bock")
    elif sys.argv[1] == '8':
        accuracy_option_numbers("Samejima")
    elif sys.argv[1] == '9':
        accuracy_option_numbers("GRM")
    elif sys.argv[1] == "10":
        accuracy_difficulty("Bock")
    elif sys.argv[1] == "11":
        accuracy_difficulty("Samejima")
    elif sys.argv[1] == "12":
        accuracy_difficulty("GRM")
    elif sys.argv[1] == "13":
        accuracy_discrimination("Bock")
    elif sys.argv[1] == "14":
        accuracy_discrimination("Samejima")
    elif sys.argv[1] == "15":
        accuracy_discrimination("GRM")
    elif sys.argv[1] == "16":
        accuracy_probability_different_number("Bock")
    elif sys.argv[1] == "17":
        accuracy_probability_different_number("Samejima")
    elif sys.argv[1] == "18":
        accuracy_probability_different_number("GRM")
    elif sys.argv[1] == "19":
        accuracy_question_numbers("C1P")
    elif sys.argv[1] == "20":
        efficiency_question_numbers()
    elif sys.argv[1] == "21":
        efficiency_user_numbers()
    

if __name__ == "__main__":
    main()

