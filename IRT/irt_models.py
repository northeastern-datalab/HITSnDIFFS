"""
This file defines various IRT models, their alternative parameterizations, functions to convert between them, or to normalize the parameters
Also includes a standardized way to show plots
First version: 12/21/2021
This version: 4/25/2024

Methods explained in detail in:
“HITSNDIFFS: From Truth Discovery to Ability Discovery by Recovering Matrices with the Consecutive Ones Property,”
Zixuan Chen, Subhodeep Mitra, R Ravi, Wolfgang Gatterbauer. ICDE 2024. (https://arxiv.org/abs/2401.00013)
"""

import numpy as np
from scipy.special import expit
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)


def IRT_parameterization(alpha, beta):
    """
    transforms numpy arrays for alpha and beta of equal length
    from slope-intercept parameteriation (logistic regression)
    to discrimination-difficulty parameterization (IRT)
    """
    assert type(alpha).__module__ == "numpy"
    assert type(beta).__module__ == "numpy"
    assert len(alpha) == len(beta)

    return alpha, -beta / alpha


def slope_intercept_parameterization(a, b):
    """
    transforms numpy arrays for a and b of equal length
    from discrimination-difficulty parameterization (IRT)
    to slope-intercept parameteriation (logistic regression)
    """
    assert type(a).__module__ == "numpy"
    assert type(b).__module__ == "numpy"
    assert len(a) == len(b)

    return a, -b * a


def normalize_Bock_parameterization(alpha, beta):
    """
    normalizes the slope and intercept for the Bock model.
    takes k-dimensional alpha, beta parameters and returns k-1 dimensional alpha', beta'
    Subtracts alpha[0] and beta[0]
    """
    assert type(alpha).__module__ == "numpy"
    assert type(beta).__module__ == "numpy"
    assert len(alpha) == len(beta)

    alpha2 = alpha - alpha[0]
    beta2 = beta - beta[0]
    return alpha2[1:], beta2[1:]


def normalize_Bock_IRT_parameterization(a, b):
    """
    normalizes the discrimination and difficulty for the Bock-IRT model.
    takes k-dimensional a, b parameters and returns k-1 dimensional a0, b0
    Uses: normalize_Bock_parameterization(alpha, beta)
    Thus ues alpha'[i] = alpha[i]-alpha[0] and beta'[i] = beta[i]-beta[0]
    """
    alpha1, beta1 = slope_intercept_parameterization(a, b)
    alpha2, beta2 = normalize_Bock_parameterization(alpha1, beta1)
    a3, b3 = IRT_parameterization(alpha2, beta2)
    return a3, b3


def Bock_IRT_response(a, b, theta):
    """
    calculates the response functions for the Bock model under IRT parameterization (discrimination-difficulty)
    Args:
        a: [k] discrimination parameters in 1D numpy array
        b: [k] difficulty parameters in 1D numpy array
        theta: [m] student abilities in 1D numpy array
    Returns:
        (k x m) np array of response probabilities
    """
    assert type(a).__module__ == "numpy"
    assert type(b).__module__ == "numpy"
    assert type(theta).__module__ == "numpy"
    assert len(a) == len(b)

    logit = theta[None, :] - b[:, None]
    logit *= a[:, None]
    y = np.exp(logit)
    return y / y.sum(axis=0)


def Bock_normalized_IRT_response(a, b, theta):
    """
    calculates the response functions for the normalized Bock model under IRT parameterization (discrimination-difficulty)
    Normalization assumes alpha'[i] = alpha[i]-alpha[0] and beta'[i] = beta[i]-beta[0]
    Args:
        a: [k-1] discrimination parameters in 1D numpy array
        b: [k-1] difficulty parameters in 1D numpy array
        theta: [m] student abilities in 1D numpy array
    Returns:
        (k x m) np array of response probabilities
    """
    assert type(a).__module__ == "numpy"
    assert type(b).__module__ == "numpy"
    assert type(theta).__module__ == "numpy"
    assert len(a) == len(b)

    logit = theta[None, :] - b[:, None]
    logit *= a[:, None]
    y = np.exp(logit)
    m = len(theta)
    y = np.vstack([np.ones(m), y])      # exp^0 = 1, thus add a row of all ones
    return y / y.sum(axis=0)


def Bock_normalized_response(alpha, beta, theta):
    """
    calculates the response functions for the normalized Bock model under original slope intercept parameterization
    Normalization assumes that alpha = 0 and beta = 0 for the left-out value
    Args:
        alpha: [k-1] discrimination parameters in 1D numpy array
        beta: [k-1] difficulty parameters in 1D numpy array
        theta: [m] student abilities in 1D numpy array

    Returns:
        (k x m) np array of response probabilities
    """
    assert type(alpha).__module__ == "numpy"
    assert type(beta).__module__ == "numpy"
    assert type(theta).__module__ == "numpy"
    assert len(alpha) == len(beta)

    logit = alpha[:, None] * theta[None, :] + beta[:, None]
    y = np.exp(logit)
    m = len(theta)
    y = np.vstack([np.ones(m), y])      # exp^0 = 1, thus add a row of all ones
    return y / y.sum(axis=0)


def TwoPL_response(a, b, theta):
    """
    calculates the 2PL response functions
    Args:
        a, b: discrimination and difficulty parameters (scalars)
        theta: [m] student abilities in 1D numpy array
    Returns:
        m np array of response probabilities
    """
    return 1 / (1 + np.exp(-a * (theta - b)))


def ThreePL_response(a, b, c, theta):
    """
    calculates the 3PL response functions
    Args:
        a, b: discrimination and difficulty parameters (scalars)
        c: random guessing parameter
        theta: [m] student abilities in 1D numpy array
    Returns:
        m np array of response probabilities
    """
    return (1-c) / (1 + np.exp(-a * (theta - b))) + c


def Heaviside_response(b, theta):
    """
    calculates the response function of a Heaviside distribution
    to be consistent with other response functions, still uses the x-axis values (theta vector) as input
    Args:
        b: difficulty parameters (scalar)
        theta: [m] student abilities in 1D numpy array (only to be consistent with other response functions)
    Returns:
        4 np array of theta values
        4 np array of response probabilities
    """
    x = (np.min(theta), b, b, np.max(theta))
    y = (0, 0, 1, 1)
    return (np.array(x),np.array(y))


def transform_Bock_IRT_To_2PL(a, b):
    """
    If the two options with 1st and 2nd highest discrimination values "a" are nicely enough separated from the other options in the Bock model,
    then this is a 2PL approximation of Bock IRT for top option.
    Works for both normalized and non-normalized a, b (as long as at least 2 entries)
    """
    assert type(a).__module__ == "numpy"
    assert type(b).__module__ == "numpy"
    assert len(a) == len(b)

    sort = np.argsort(a)    # find indices of top and second highest a values
    first = sort[-1]
    second = sort[-2]
    a0 = a[first] - a[second]
    b0 = (a[first]*b[first] - a[second]*b[second])/a0
    return a0, b0



def GRM_cum(a, b, theta):
    """
    calculates the cumulative response functions for the homogenous GRM (Graded Response Model)

    Args:
        a: discrimination number
        b: [k-1] difficulty parameters in 1D numpy array
        theta: [m] student abilities in 1D numpy array

    Returns:
        (k-1 x m) np array of cumulative response probabilities
    """
    assert type(a) in(int, float)
    assert type(b).__module__ == "numpy"
    assert type(theta).__module__ == "numpy"
    assert not isinstance(a, list)

    logit = theta[None, :] - b[:, None]
    logit *= a
    return expit(logit)


def GRM_response(a, b, theta):
    """
    calculates the actual response functions for the homogenous GRM (Graded Response Model)

    Args:
        a: discrimination number
        b: [k-1] difficulty parameters in 1D numpy array
        theta: [m] student abilities in 1D numpy array

    Returns:
        (k x m) np array of response probabilities
    """
    assert type(b).__module__ == "numpy"
    assert type(theta).__module__ == "numpy"
    assert not isinstance(a, list)

    y = GRM_cum(a, b, theta)
    # (h, w) = y.shape
    w = len(theta)
    y = np.vstack([np.ones(w), y, np.zeros(w)])
    return -np.diff(y, axis=0)




def transform_GRM_To_Bock_IRT(a, b):
    """
    Transforms the value from GRM to some approximate Bock_IRT.
    Only works well if the values are spread out enough
    """
    assert type(a) in (int, float)
    assert type(b).__module__ == "numpy"

    k = len(b)  # k-1
    a0 = np.array([i * a for i in range(1, k + 1)])
    for i, bi in enumerate(b):
        if i == 0:
            b0 = [b[0]]
        else:
            b0.append(((a0[i] - a0[i - 1]) * b[i] + a0[i - 1] * b0[i - 1]) / a0[i])  # basically inverse of Bock_IRT to 2PL
    b0 = np.array(b0)
    return a0, b0


def Samejima_response(alpha, beta, theta):
    """
    calculates the response functions for the Samejima model under original slope intercept parameterization

    Args:
        alpha: [k+1] discrimination parameters in 1D numpy array
        beta: [k+1] intersection parameters in 1D numpy array
        theta: [m] student abilities in 1D numpy array

    Returns:
        (k x m) np array of response probabilities
    """
    assert type(alpha).__module__ == "numpy"
    assert type(beta).__module__ == "numpy"
    assert type(theta).__module__ == "numpy"
    assert len(alpha) == len(beta)

    k = len(alpha) - 1
    logit = alpha[:, None] * theta[None, :] + beta[:, None]
    y0 = np.exp(logit)
    y = y0[1:, :] + y0[0, :]/k
    return y / y0.sum(axis=0)


def Samejima_IRT_response(a, b, theta):
    """
    calculates the response functions for the Samejima model under discrimination-difficulty parameterization

    Args:
        a: [k+1] discrimination parameters in 1D numpy array
        b: [k+1] difficulty parameters in 1D numpy array
        theta: [m] student abilities in 1D numpy array

    Returns:
        (k x m) np array of response probabilities
    """
    assert type(a).__module__ == "numpy"
    assert type(b).__module__ == "numpy"
    assert type(theta).__module__ == "numpy"
    assert len(a) == len(b)

    k = len(a) - 1
    logit = theta[None, :] - b[:, None]
    logit *= a[:, None]
    y0 = np.exp(logit)
    y = y0[1:, :] + y0[0, :] / k
    return y / y0.sum(axis=0)


def normalized_Samejima_IRT_response(a, b, theta):
    """
    calculates the response functions for the normalized Samejima model under discrimination-difficulty parameterization

    Args:
        a: [k] discrimination parameters in 1D numpy array
        b: [k] difficulty parameters in 1D numpy array
        theta: [m] student abilities in 1D numpy array

    Returns:
        (k x m) np array of response probabilities
    """
    assert type(a).__module__ == "numpy"
    assert type(b).__module__ == "numpy"
    assert type(theta).__module__ == "numpy"
    assert len(a) == len(b)

    k = len(a)
    logit = theta[None, :] - b[:, None]
    logit *= a[:, None]
    y0 = np.exp(logit)
    y = y0 + 1 / k
    return y / (y0.sum(axis=0) + 1)


def plot_figure(x, y, low, high,
                title=None, pdfname=None, linewidth=None, label=None,
                logscale=False, linestyle=None, marker=None, markevery=None,
                xmin=None, show_legend=True, fine_grid=False,
                xlabel=r'User Ability', ylabel=r'Probability', color=None, squaresize=False):
    """
    Unified plot function. Allows to input y as tuple of 1D and 2D numpy arrays
    Optional linewidth, label tuples

    Example usage:
    plot_figure(theta, (y_nb, y_bi), low, high, title =r"Samejima-IRT vs. Bock-IRT",
               label=("1", "2", "3", "4", "1", "2", "3", "4"), linewidth=(5, 5, 5, 5, 1, 1, 1, 1))
    """
    mpl.rc('font', **{'family': 'sans-serif', 'sans-serif': [u'Arial', u'Liberation Sans']})
    mpl.rcParams['axes.labelsize'] = 20
    mpl.rcParams['xtick.labelsize'] = 16
    mpl.rcParams['ytick.labelsize'] = 16
    mpl.rcParams['legend.fontsize'] = 14  # 6
    mpl.rcParams['grid.color'] = '777777'  # grid color
    mpl.rcParams['xtick.major.pad'] = 2  # padding of tick labels: default = 4
    mpl.rcParams['ytick.major.pad'] = 1  # padding of tick labels: default = 4
    mpl.rcParams['xtick.direction'] = 'out'  # default: 'in'
    mpl.rcParams['ytick.direction'] = 'out'  # default: 'in'
    mpl.rcParams['axes.titlesize'] = 16
    if squaresize:
        # mpl.rcParams['figure.figsize'] = [4, 4]
        fig = plt.figure(figsize=(4, 4))
    else:
        fig = plt.figure(figsize=(8, 4))
    ax = fig.add_axes([0.18, 0.17, 0.76, 0.75])   # [left, bottom, width, height]

    if type(y) == tuple:
        y = np.vstack(y)
    y = np.atleast_2d(y)
    for i, yi in enumerate(y):
        plt.plot(x, yi,
                 label='{}'.format(i) if label is None else label[i],
                 linewidth=2 if linewidth is None else linewidth[i],
                 linestyle="-" if linestyle is None else linestyle[i],
                 marker=None if marker is None else marker[i],
                 markevery=None if markevery is None else markevery[i],
                 color=color[i] if color is not None else plt.rcParams['axes.prop_cycle'].by_key()['color'][i]) # assign colors from default color cycle unless explicitly specified

    plt.xlim(low, high)
    if logscale:
        plt.yscale("log")
        plt.grid(True, which="both")
        if xmin:
            plt.ylim(xmin, 2)
    else:
        plt.ylim(-0.02, 1.02)
        plt.grid(True)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    if title:
        plt.title(title, fontsize=15)

    handles, labels = ax.get_legend_handles_labels()
    legend = plt.legend(handles, labels,
                        # loc=legend_location,  # 'upper right'
                        handlelength=2,
                        labelspacing=0,  # distance between label entries
                        handletextpad=0.3,  # distance between label and the line representation
                        # title='Variants',
                        borderaxespad=0.2,  # distance between legend and the outer axes
                        borderpad=0.3,  # padding inside legend box
                        numpoints=1,  # put the marker only once
                        )
    frame = legend.get_frame()
    frame.set_linewidth(0.0)
    frame.set_alpha(0.9)  # 0.8

    if not show_legend:
        legend.remove()

    if fine_grid:                                           # Change major and minor ticks
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        # ax.xaxis.set_minor_locator(AutoMinorLocator(1))     # multiplicator for number of minor gridlines
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))     # multiplicator for number of minor gridlines
        ax.grid(which='major', alpha=0.7)
        ax.grid(which='minor', alpha=0.2)



    if pdfname:
        plt.savefig("figures/" + pdfname + ".pdf",
                    format='pdf',
                    dpi=None,
                    edgecolor='w',
                    orientation='portrait',
                    transparent=False,
                    bbox_inches='tight',
                    pad_inches=0.05)
    plt.show()

