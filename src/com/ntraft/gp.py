'''
Created on Apr 1, 2014

@author: ntraft
'''
from __future__ import division
import numpy as np
import com.ntraft.covariance as cov

class GaussianProcess:
	'''
	Represents a Gaussian process that can be sampled from. Samples are taken
	at each test point, given the supplied observations.
	'''

	def __init__(self, zx, zy, testpoints, kernel=cov.sq_exp_kernel()):
		''' Creates a new Gaussian process from the given observations. '''
		self.timepoints = testpoints
		
		# covariance of observations
		K = kernel(zx, zx)
		K += 1e-9*np.eye(K.shape[0])
		L = np.linalg.cholesky(K)
		
		# compute the mean at our test points
		Lk = np.linalg.solve(L, kernel(zx, testpoints))
		self.mu = np.dot(Lk.T, np.linalg.solve(L, zy))
		
		# compute the variance at our test points
		K_ = kernel(testpoints, testpoints)
		K_ += 1e-9*np.eye(K_.shape[0])
		self.prior_L = np.linalg.cholesky(K_)
		self.L = np.linalg.cholesky(K_ - np.dot(Lk.T, Lk))
	
	def sample(self, n=1):
		'''
		Draw n samples from the gaussian process posterior.
		
		Returns a timepoints x n matrix, with each sample being a column.
		'''
		sz = (len(self.timepoints), n)
		return self.mu.reshape(-1,1) + np.dot(self.L, np.random.normal(size=sz))
	
	def sample_prior(self, n=1):
		'''
		Draw n samples from the gaussian process prior.
		
		Returns a timepoints x n matrix, with each sample being a column.
		'''
		sz = (len(self.timepoints), n)
		return np.dot(self.prior_L, np.random.normal(size=sz))


class ParametricGaussianProcess:
	'''
	Represents a Gaussian process of a parametric function. This is actually
	implemented as two separate GPs, one for x and one for y. The processes can
	be sampled from to predict x,y = f(t). Samples are taken at each test
	point, given the supplied observations.
	'''

	def __init__(self, observations, timepoints, xkernel=cov.sq_exp_kernel(), ykernel=cov.sq_exp_kernel()):
		zt = observations[:,0]
		zx = observations[:,1]
		zy = observations[:,2]

		self.xgp = GaussianProcess(zt, zx, timepoints, xkernel)
		self.ygp = GaussianProcess(zt, zy, timepoints, ykernel)
	
	def sample(self, n=1):
		'''
		Draw n samples from the gaussian process posterior.
		
		Returns a timepoints x n x 2 matrix. The first dimension is time, the
		second dimension is samples, and the third dimension is x,y.
		'''
		x_post = self.xgp.sample(n)
		y_post = self.ygp.sample(n)
		return np.dstack((x_post, y_post))
	
	def sample_prior(self, n=1):
		'''
		Draw n samples from the gaussian process prior.
		
		Returns a timepoints x n x 2 matrix. The first dimension is time, the
		second dimension is samples, and the third dimension is x,y.
		'''
		x_post = self.xgp.sample_prior(n)
		y_post = self.ygp.sample_prior(n)
		return np.dstack((x_post, y_post))
