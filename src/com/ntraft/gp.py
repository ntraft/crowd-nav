'''
Created on Apr 1, 2014

@author: ntraft
'''
from __future__ import division
import numpy as np
import com.ntraft.covariance as cov
from util import OBS_NOISE

class GaussianProcess:
	'''
	Represents a Gaussian process that can be sampled from. Samples are taken
	at each test point, given the supplied observations.
	'''

	def __init__(self, observations, timepoints, kernel=cov.sq_exp_kernel()):
		''' Creates a new Gaussian process from the given observations. '''
		zt = observations[:,0]
		zx = observations[:,1]
		zy = observations[:,2]
		self.timepoints = timepoints
		
		# covariance of observations
		K = kernel(zt, zt)
		L = np.linalg.cholesky(K + OBS_NOISE*np.eye(len(zt)))
		
		# compute the mean at our test points
		Lk = np.linalg.solve(L, kernel(zt, timepoints))
		self.xmu = np.dot(Lk.T, np.linalg.solve(L, zx))
		self.ymu = np.dot(Lk.T, np.linalg.solve(L, zy))
		
		# compute the variance at our test points
		K_ = kernel(timepoints, timepoints)
		self.L = np.linalg.cholesky(K_ + 1e-6*np.eye(K_.shape[0]) - np.dot(Lk.T, Lk))
	
	def sample(self, n=1):
		''' Draw n samples from the gaussian process. '''
		sz = (len(self.timepoints), n)
		x_post = self.xmu.reshape(-1,1) + np.dot(self.L, np.random.normal(size=sz))
		y_post = self.ymu.reshape(-1,1) + np.dot(self.L, np.random.normal(size=sz))
		return np.dstack((x_post, y_post))
