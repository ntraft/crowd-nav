'''
Utility functions for dealing with the BIWI Walking Pedestrians dataset (or
EWAP for "ETH Walking Pedestrians"). From Pellegrini et al., ICCV 2009.

Created on Mar 1, 2014

@author: ntraft
'''
import numpy as np
from PIL import Image

def create_obstacle_map(H, map_png):
	rawmap = np.array(Image.open(map_png))
	return rawmap # TODO transform into world coords

def parse_annotations(obsmat_txt):
	mat = np.loadtxt(obsmat_txt)
	num_frames = mat[-1,0] + 1
	num_times = np.unique(mat[:,0]).size
	num_peds = int(np.max(mat[:,1])) + 1
	frames = [-1] * num_frames # maps frame -> timestep
	timesteps = [[] for _ in range(num_times)] # maps timestep -> ped IDs
	peds = [np.array([]).reshape(0,4) for _ in range(num_peds)] # maps timestep -> ped IDs
	frame = 0
	time = -1
	for row in mat:
		if row[0] != frame:
			frame = int(row[0])
			time += 1
			frames[frame] = time
		ped = int(row[1])
		timesteps[time].append(ped)
		loc = [time, row[2], row[4], 1]
		peds[ped] = np.vstack((peds[ped], loc))
	return (frames, timesteps, peds)
