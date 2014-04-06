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
x1 = lambda x: x.flatten()
x2 = lambda x: np.sin(0.9*x).flatten()
# x2 = lambda x: 2*np.ones_like(x)

# Sample some input points and noisy versions of the function evaluated at
# these points.
N = 10		# number of training points
n = 500		# number of test points
s = 0.00005	# noise variance
T = np.random.uniform(-5, 0, size=(N,))
T[-1] = 4.8 # set a goal point
x = x1(T) + s*np.random.randn(N)
y = x2(T) + s*np.random.randn(N)
z = np.column_stack((T, x, y))

# points we're going to make predictions at.
Ttest = np.linspace(-5, 5, n)

# Build our Gaussian process.
# kernel = cov.sq_exp_kernel(3.2, 1)
# kernel = cov.summed_kernel(cov.sq_exp_kernel(3.2, 1), cov.noise_kernel(s))
kernel = cov.summed_kernel(
	cov.matern_kernel(2.28388, 2.52288),
	cov.linear_kernel(2.87701),
	cov.noise_kernel(0.24071)
)
# kernel = cov.matern_kernel(2.28388, 2.52288)
# kernel = cov.linear_kernel(-2.87701)
gp = GaussianProcess(z, Ttest, kernel)

# PLOTS:

# draw samples from the prior at our test points.
samples = gp.sample_prior(10)
pl.figure(1)
pl.plot(samples[:,:,0], samples[:,:,1])
pl.title('Ten samples from the GP prior')
pl.axis([-5, 5, -3, 3])

# draw 10 samples from the posterior
samples = gp.sample(10)
pl.figure(2)
pl.plot(samples[:,:,0], samples[:,:,1])
pl.title('Ten samples from the GP posterior')
pl.axis([-5, 5, -3, 3])

# illustrate the possible paths.
pl.figure(3)
pl.plot(x, y, 'bx', ms=20)
pl.plot(x1(Ttest), x2(Ttest), 'b-')
# pl.gca().fill_between(Ttest.flat, mu-3*s, mu+3*s, color="#dddddd") # how to draw this?
pl.plot(gp.xmu, gp.ymu, 'r--', lw=2)
pl.title('Mean predictions plus 3 st.deviations')
pl.axis([-5, 5, -3, 3])

pl.show()
