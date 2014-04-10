'''
Utility functions for dealing with the BIWI Walking Pedestrians dataset (or
EWAP for "ETH Walking Pedestrians"). From Pellegrini et al., ICCV 2009.

Created on Mar 1, 2014

@author: ntraft
'''
import numpy as np
from PIL import Image
import com.ntraft.util as util

def create_obstacle_map(map_png):
	rawmap = np.array(Image.open(map_png))
	return rawmap

def parse_annotations(Hinv, obsmat_txt):
	mat = np.loadtxt(obsmat_txt)
	num_frames = mat[-1,0] + 1
	num_times = np.unique(mat[:,0]).size
	num_peds = int(np.max(mat[:,1])) + 1
	frames = [-1] * num_frames # maps frame -> timestep
	timeframes = [-1] * num_times # maps timestep -> (first) frame
	timesteps = [[] for _ in range(num_times)] # maps timestep -> ped IDs
	peds = [np.array([]).reshape(0,4) for _ in range(num_peds)] # maps ped ID -> (t,x,y,z) path
	frame = 0
	time = -1
	for row in mat:
		if row[0] != frame:
			frame = int(row[0])
			time += 1
			frames[frame] = time
			timeframes[time] = frame
		ped = int(row[1])
		timesteps[time].append(ped)
		loc = np.array([row[2], row[4], 1])
		loc = util.to_image_frame(Hinv, loc)
		loc = [time, loc[0], loc[1], loc[2]]
		peds[ped] = np.vstack((peds[ped], loc))
	return (frames, timeframes, timesteps, peds)
