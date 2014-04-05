from __future__ import division
import numpy as np
import com.ntraft.covariance as cov
from com.ntraft.gp import GaussianProcess
from util import OBS_NOISE
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

# Build our Gaussian process.
gp = GaussianProcess(z, Ttest)

# draw 10 samples from the posterior
samples = gp.sample(10)
pl.clf()
pl.plot(samples[:,:,0], samples[:,:,1])
pl.title('Ten samples from the GP posterior')
pl.axis([-5, 5, -3, 3])

pl.show()
