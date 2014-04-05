'''
Various utility functions.

@author: ntraft
'''
from __future__ import division
import numpy as np

NUM_SAMPLES = 10	# number of particles
OBS_NOISE = 0.00005	# noise variance
ALPHA = 0.8			# repelling force
H = 24				# safety distance

def to_pixels(Hinv, loc):
	"""
	Given H^-1 and (x, y, z) in world coordinates, returns (c, r) in image
	pixel indices.
	"""
	loc = to_image_frame(Hinv, loc)
	return (loc[1], loc[0])

def to_image_frame(Hinv, loc):
	"""
	Given H^-1 and (x, y, z) in world coordinates, returns (u, v, 1) in image
	pixel indices.
	"""
	loc = np.dot(Hinv, loc) # to camera frame
	return (loc/loc[2]).astype(int) # to pixels (from millimeters)

def rbf(loc1, loc2):
	dist = np.linalg.norm(loc2 - loc1)
	dist *= 100 # Hack for now, to get distances into a sensible range.
	return 1 - ALPHA*np.exp(-(dist**2) / (2*H**2))

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

