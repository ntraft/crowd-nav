'''
Created on Apr 1, 2014

@author: ntraft
'''
from __future__ import division
import numpy as np
from util import OBS_NOISE

def sq_exp(a, b):
	""" GP squared exponential kernel """
	kernelParameter = 10 # l^2 in the formulas
	sqdist = (a**2).reshape(-1,1) + b**2 - 2*np.outer(a, b)
	return np.exp(-.5 * (1/kernelParameter) * sqdist)

class GaussianProcess:
	'''
	Represents a Gaussian process that can be sampled from. Samples are taken
	at each test point, given the supplied observations.
	'''

	def __init__(self, observations, timepoints, kernel=sq_exp):
		'''
		Creates a new Gaussian process from the given observations.
		'''
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
		'''
		Draw n samples from the gaussian process.
		'''
		sz = (len(self.timepoints), n)
		x_post = self.xmu.reshape(-1,1) + np.dot(self.L, np.random.normal(size=sz))
		y_post = self.ymu.reshape(-1,1) + np.dot(self.L, np.random.normal(size=sz))
		return np.dstack((x_post, y_post))


if __name__ == "__main__":
	import matplotlib
	# The 'MacOSX' backend appears to have some issues on Mavericks.
	import sys
	if sys.platform.startswith('darwin'):
		matplotlib.use('TkAgg')
	import matplotlib.pyplot as pl
	
	# This is the true unknown function we are trying to approximate
	x1 = lambda x: x.flatten()
	x2 = lambda x: np.sin(0.9*x).flatten()
	
	# Sample some input points and noisy versions of the function evaluated at
	# these points.
	N = 10		# number of training points
	n = 500		# number of test points
	T = np.random.uniform(-5, 5, size=(N,))
	x = x1(T) + OBS_NOISE*np.random.randn(N)
	y = x2(T) + OBS_NOISE*np.random.randn(N)
	z = np.column_stack((T, x, y))
	
	# points we're going to make predictions at.
	Ttest = np.linspace(-5, 5, n)
	
	# draw 10 samples from the posterior
	gp = GaussianProcess(z, Ttest)
	samples = gp.sample(10)
	pl.clf()
	pl.plot(samples[:,:,0], samples[:,:,1])
	pl.title('Ten samples from the GP posterior')
	pl.axis([-5, 5, -3, 3])
	
	pl.show()
