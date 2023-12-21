"""
Provide the implementations of HITSNDIFFS
"""
import numpy as np
import math
import time
from scipy.sparse.linalg import eigs
from spectral import getCMatrix, order_vs_reverse

def AvgHITS_2nd_eigenvector(Labels):
	"""
    Calculate the second largest eigenvector of U (defined in our paper) directly

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
	C_l0 = C.sum(0)
	C_l1 = C.sum(1)
	C_l0[C_l0 == 0] = 1
	C_col = np.true_divide(C, C_l0)
	C_row = np.true_divide(C.T, C_l1).T
	U = C_row.dot(C_col.T)

	e = eigs(U, k = 2, which = 'LM')
	s = e[1][:,np.argmin(e[0])]

	end = time.perf_counter()
	s = order_vs_reverse(s, C, Labels)

	return np.round(s, 8), end - start

def AvgHITS_2nd_eigenvector_deflation(Labels, tol=1e-5):
	"""
    Calculate the second largest eigenvector of U (defined in our paper) using the deflation method
	This is the implementation of the Hotelling's deflation method,
	which for a non-symmetric matrix, needs to compute both the left and right eigenvectors

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
	C_l0 = C.sum(0)
	C_l1 = C.sum(1)
	C_l0[C_l0 == 0] = 1
	C_col = np.true_divide(C, C_l0)
	C_row = np.true_divide(C.T, C_l1).T

	lambda1 = 1
	v1 = np.ones(M) * math.sqrt(1 / M)

	iter = 0
	u1 = np.random.uniform(0,1,M)
	u1_new = np.random.uniform(0,1,M)
	while np.linalg.norm(u1_new - u1) > tol:
		iter += 1
		u1 = np.copy(u1_new)                     
		w = (u1.dot(C_row)).dot(C_col.T)
		u1_new = w / np.linalg.norm(w)

	v2 = np.random.uniform(0,1,M)
	v2_new = np.random.uniform(0,1,M)
	while np.linalg.norm(v2_new - v2) > tol:
		iter += 1
		v2 = np.copy(v2_new)                                                   
		w = C_row.dot((C_col.T).dot(v2)) - lambda1 * v1.reshape(M,1).dot(u1.reshape(1,M).dot(v2))
		v2_new = w / np.linalg.norm(w)
		
	end = time.perf_counter()
	v2 = order_vs_reverse(v2, C, Labels)

	return np.round(v2, 8), end - start, iter

def HITSnDIFFS(Labels, tol=1e-5):
	"""
	The faster implementation of HITSnDIFFS by power iteration
	by using matrix-vector multiplication instead of matrix-matrix multiplication

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

	C_l0 = C.sum(0)
	C_l1 = C.sum(1)
	C_l0[C_l0 == 0] = 1
	C_col = np.true_divide(C, C_l0)
	C_row = np.true_divide(C.T, C_l1).T

	iter = 0
	while np.linalg.norm(sdiff_new - sdiff) > tol:
		iter += 1
		sdiff = np.copy(sdiff_new)
		s = np.zeros(M)
		s[1:] = np.cumsum(sdiff)
		w = C_col.T.dot(s)
		s = C_row.dot(w)

		sdiff_new = np.diff(s)
		sdiff_new = sdiff_new / np.linalg.norm(sdiff_new)

	s = np.zeros(M)
	s[1:] = np.cumsum(sdiff)
	
	end = time.perf_counter()
	s = order_vs_reverse(s, C, Labels)

	return np.round(s, 8), end - start, iter