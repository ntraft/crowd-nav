'''
Various utility functions.

@author: ntraft
'''
from __future__ import division
import numpy as np
from numpy.core.numeric import inf
import time

from com.ntraft.gp import ParametricGaussianProcess
import com.ntraft.covariance as cov
from collections import namedtuple

NUM_SAMPLES = 100	# number of particles
OBS_NOISE = 0.00005	# noise variance
ALPHA = 0.8			# repelling force
H = 24				# safety distance

# The all-important kernels and their hyperparameters.
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

total_time = 0
total_runs = 0
def timeit(f):
	def timed(*args, **kw):
		global total_time, total_runs
		ts = time.time()
		result = f(*args, **kw)
		te = time.time()
		total_time += te-ts
		total_runs += 1
		return result
	return timed

def reset_timer():
	total_time = 0; total_runs = 0

def report_time():
	print 'IGP on average took {:.2f} seconds with {} particles.'.format(total_time/total_runs, NUM_SAMPLES)

def to_pixels(Hinv, loc):
	"""
	Given H^-1 and (x, y, z) in world coordinates, returns (c, r) in image
	pixel indices.
	"""
	loc = to_image_frame(Hinv, loc).astype(int)
	return (loc[1], loc[0])

def to_image_frame(Hinv, loc):
	"""
	Given H^-1 and (x, y, z) in world coordinates, returns (u, v, 1) in image
	frame coordinates.
	"""
	loc = np.dot(Hinv, loc) # to camera frame
	return loc/loc[2] # to pixels (from millimeters)

Predictions = namedtuple('Predictions', ['past', 'true_paths', 'prior', 'posterior', 'MAP'])
empty_predictions = Predictions([],[],[],[],[])

@timeit
def make_predictions(t, timesteps, agents, robot=-1, past_plan=None):
	peds = timesteps[t]
	past_paths = []
	true_paths = []
	prior = []
	for ped in peds:
		# Get the past and future paths of the agent.
		past_plus_dest, future = get_path_at_time(t, agents[ped])
		past_paths.append(past_plus_dest[:-1,1:4].copy())
		# Replace human's path with robot's path.
		if past_plan is not None and ped == robot:
			past_plus_dest[:-1,1:] = past_plan
		true_paths.append(future[:,1:4])
		
		# Predict possible paths for the agent.
		t_future = future[:,0]
		gp = ParametricGaussianProcess(past_plus_dest, t_future, xkernel, ykernel)
		samples = gp.sample(NUM_SAMPLES)
		prior.append(samples)
	
	# Perform importance sampling and get the maximum a-posteriori path.
	weights = interaction(prior)
	posterior = resample(prior, weights)
	MAP = [get_final_path(p) for p in posterior]
	return Predictions(past_paths, true_paths, prior, posterior, MAP)

def get_path_at_time(t, fullpath):
	path_end = next(i for i,v in enumerate(fullpath[:,0]) if v==t)
	points = list(range(0,path_end+1))
	if path_end < fullpath.shape[0]:
		points += [-1] # Add the destination point.
	past_plus_dest = fullpath[np.ix_(points)]
	future = fullpath[path_end:]
	return past_plus_dest, future

def dist(loc1, loc2):
	return np.linalg.norm(loc2 - loc1)

def rbf(loc1, loc2):
	d = dist(loc1, loc2)
	d *= 100 # Hack for now, to get distances into a sensible range.
	return 1 - ALPHA*np.exp(-(d**2) / (2*H**2))

def interaction(allpriors):
	"""
	The Interaction Potential, denoted as 'psi' in Trautman & Krause, 2010.
	"""
	# Input has shape: (agent, time, samples, x/y)
	weights = np.ones(NUM_SAMPLES)
	for i in range(NUM_SAMPLES):
		num_agents = len(allpriors)
		for j in range(num_agents):
			agent_j = allpriors[j]
			for k in range(j+1, num_agents):
				agent_k = allpriors[k]
				for t in range(min(len(agent_j), len(agent_k))):
					weights[i] *= rbf(agent_j[t,i], agent_k[t,i])
	# Renormalize
	# TODO deal with the case when all paths are weighted to 0
	weights /= np.sum(weights)
	return weights

def resample(allpriors, weights):
	"""
	Performs importance sampling over the prior distribution in order to
	approximate the posterior, using the given weights. Implemented using a
	resampling wheel.
	"""
	allposteriors = [np.empty_like(p) for p in allpriors]
	beta = 0
	N = len(weights)
	wmax = max(weights)
	windex = np.random.choice(range(N))
	for i in range(N):
		beta += np.random.random() * 2*wmax
		while weights[windex] < beta:
			beta -= weights[windex]
			windex = (windex+1) % N
		# We've selected a sample. Now copy that sample for all agents.
		for agent in range(len(allpriors)):
			allposteriors[agent][:,i,:] = allpriors[agent][:,windex,:]
	return allposteriors

def get_final_path(samples):
	return np.column_stack((np.mean(samples, 1), np.ones(samples.shape[0])))

def calc_score(path, other_paths):
	length = 0
	safety = inf
	prev_loc = None
	for t in range(len(path)):
		loc = path[t]
		if prev_loc is not None:
			length += dist(prev_loc, loc)
		prev_loc = loc
		for o in other_paths:
			if t < len(o):
				d = dist(o[t], loc)
				if d < safety:
					safety = d
	return (length, safety)

def calc_scores(true_paths, MAP):
	robot_scores = np.array([calc_score(path, true_paths[:i]+true_paths[i+1:]) for i, path in enumerate(MAP)])
	ped_scores = np.array([calc_score(path, true_paths[:i]+true_paths[i+1:]) for i, path in enumerate(true_paths)])
	return ped_scores, robot_scores
