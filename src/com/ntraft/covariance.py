'''
Various Gaussian Process kernel functions.

Created on Apr 5, 2014

@author: ntraft
'''
from __future__ import division
import numpy as np

################################################################################
# COVARIANCE FUNCTIONS
################################################################################

def sq_dist(a, b):
	''' Squared distance from every point in a to every point in b. '''
	return (a ** 2).reshape(-1, 1) + b ** 2 - 2 * np.outer(a, b)

def sq_exp(a, b, l, sigma2):
	''' Squared exponential kernel '''
	return sigma2 * np.exp(-.5 * sq_dist(a/l, b/l))

################################################################################
# KERNEL GENERATION FUNCTIONS
################################################################################

def sq_exp_kernel(l=1, sigma2=1):
	''' Squared exponential kernel '''
	def f(a, b):
		return sq_exp(a, b, l, sigma2)
	return f