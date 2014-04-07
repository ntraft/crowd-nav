from __future__ import division
import numpy as np
import com.ntraft.covariance as cov
from com.ntraft.gp import GaussianProcess
import matplotlib
# The 'MacOSX' backend appears to have some issues on Mavericks.
import sys
if sys.platform.startswith('darwin'):
	matplotlib.use('TkAgg')
import matplotlib.pyplot as pl

# This is the true unknown function we are trying to approximate
x1 = lambda x: x.flatten() # y = x
x2 = lambda x: x.flatten() # y = x
# x2 = lambda x: 2*np.ones_like(x) # constant
# x2 = lambda x: np.sin(0.9*x).flatten() # sin

# Sample some input points and noisy versions of the function evaluated at
# these points.
N = 20		# number of training points
n = 30		# number of test points
s = 0.00000	# noise variance
# T = np.random.uniform(-5, 0, size=(N,))
T = np.linspace(-20, 0, N)
T[-1] = 39 # set a goal point
x = x1(T) + s*np.random.randn(N)
y = x2(T) + s*np.random.randn(N)
z = np.column_stack((T, x, y))

# points we're going to make predictions at.
Ttest = np.linspace(0, 40, n)

# Build our Gaussian process.
# kernel = cov.sq_exp_kernel(3.2, 1)
# kernel = cov.matern_kernel(2.28388, 2.52288)
# kernel = cov.linear_kernel(-2.87701)
# kernel = cov.summed_kernel(cov.sq_exp_kernel(3.2, 1), cov.noise_kernel(s))
xkernel = cov.summed_kernel(
	cov.matern_kernel(33.542, 47517),
	cov.linear_kernel(315.46),
	cov.noise_kernel(0.53043)
)
ykernel = cov.summed_kernel(
	cov.matern_kernel(9.8147, 155.36),
	cov.linear_kernel(17299),
	cov.noise_kernel(0.61790)
)
xgp = GaussianProcess(T, x, Ttest, xkernel)
ygp = GaussianProcess(T, y, Ttest, ykernel)

# PLOTS:

# draw samples from the prior at our test points.
xs = xgp.sample_prior(10)
ys = ygp.sample_prior(10)
pl.figure(1)
pl.plot(xs, ys)
pl.title('Ten samples from the GP prior')

# draw 10 samples from the posterior
xs = xgp.sample(5)
ys = ygp.sample(5)
pl.figure(2)
pl.subplots_adjust(0.05, 0.1, 0.95, 0.9)
pl.subplot(1,2,1)
pl.plot(x, y, 'yo', ms=8)
pl.plot(xs, ys)
pl.title('Five samples from the GP posterior')
pl.axis([-30, 70, -30, 70])

# illustrate the possible paths.
pl.subplot(1,2,2)
ns = 1000
pl.plot(x, y, 'yo', ms=8)
xmean = np.mean(xgp.sample(ns), 1)
ymean = np.mean(ygp.sample(ns), 1)
pl.plot(xmean, ymean, 'r--', lw=2)
pl.title('Mean of {} samples'.format(ns))
# pl.plot(x1(Ttest), x2(Ttest), 'b-')
# pl.plot(xgp.mu, ygp.mu, 'r--', lw=2)
# pl.title('Mean predictions')
pl.axis([-30, 70, -30, 70])

pl.show()
