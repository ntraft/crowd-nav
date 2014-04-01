from __future__ import division
import numpy as np
import matplotlib
# The 'MacOSX' backend appears to have some issues on Mavericks.
import sys
if sys.platform.startswith('darwin'):
	matplotlib.use('TkAgg')
import matplotlib.pyplot as pl


# This is the true unknown function we are trying to approximate
x1 = lambda x: x.flatten()
x2 = lambda x: np.sin(0.9*x).flatten()
#f = lambda x: (0.25*(x**2)).flatten()


# Define the kernel
def kernel(a, b):
	""" GP squared exponential kernel """
	kernelParameter = 1 # l^2 in the formulas
	sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
	return np.exp(-.5 * (1/kernelParameter) * sqdist)

N = 10		# number of training points
n = 500		# number of test points
s = 0.00005	# noise variance

# We take the ultimate destination to be the mean, so that all the paths
# "gravitate" toward that location.
xdest = 4	# destination x coord
ydest = 2	# destination x coord

# Sample some input points and noisy versions of the function evaluated at
# these points. 
T = np.random.uniform(-5, 5, size=(N,1))
x = x1(T) + s*np.random.randn(N)
y = x2(T) + s*np.random.randn(N)

K = kernel(T, T)
L = np.linalg.cholesky(K + s*np.eye(N))

# points we're going to make predictions at.
Ttest = np.linspace(-5, 5, n).reshape(-1,1)

# compute the mean at our test points.
Lk = np.linalg.solve(L, kernel(T, Ttest))
xmu = np.dot(Lk.T, np.linalg.solve(L, x))
ymu = np.dot(Lk.T, np.linalg.solve(L, y))

# compute the variance at our test points.
K_ = kernel(Ttest, Ttest)
s2 = np.diag(K_) - np.sum(Lk**2, axis=0) # same as np.dot(Lk.T, Lk) ??
s = np.sqrt(s2)


# PLOTS:

# draw samples from the prior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
x_prior = np.dot(L, np.random.normal(size=(n,10)))
y_prior = np.dot(L, np.random.normal(size=(n,10)))
pl.figure(1)
pl.clf()
pl.plot(x_prior, y_prior)
pl.title('Ten samples from the GP prior')
pl.axis([-5, 5, -3, 3])

# draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
x_post = xmu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,10)))
y_post = ymu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,10)))
pl.figure(2)
pl.clf()
pl.plot(x_post, y_post)
pl.title('Ten samples from the GP posterior')
pl.axis([-5, 5, -3, 3])

# illustrate the possible paths.
pl.figure(3)
pl.clf()
pl.plot(x, y, 'r+', ms=20)
pl.plot(x1(Ttest), x2(Ttest), 'b-')
# pl.gca().fill_between(Ttest.flat, mu-3*s, mu+3*s, color="#dddddd") # how to draw this?
pl.plot(xmu, ymu, 'r--', lw=2)
pl.title('Mean predictions plus 3 st.deviations')
pl.axis([-5, 5, -3, 3])

pl.show()
