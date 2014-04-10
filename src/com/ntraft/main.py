'''
Created on Mar 1, 2014

@author: ntraft
'''
from __future__ import division
import sys
import os
import argparse
import cv2
import numpy as np
import matplotlib
# The 'MacOSX' backend appears to have some issues on Mavericks.
import sys
if sys.platform.startswith('darwin'):
	matplotlib.use('TkAgg')
import matplotlib.pyplot as pl

import com.ntraft.ewap as ewap
from com.ntraft.gp import ParametricGaussianProcess
import com.ntraft.covariance as cov
import com.ntraft.util as util
from numpy.core.numeric import inf

POS_MSEC = cv2.cv.CV_CAP_PROP_POS_MSEC
POS_FRAMES = cv2.cv.CV_CAP_PROP_POS_FRAMES
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ESC = 27

DRAW_FINAL_PATH = True

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

def main():
	# Parse command-line arguments.
	args = parse_args()
	
	Hfile = os.path.join(args.datadir, "H.txt")
	mapfile = os.path.join(args.datadir, "map.png")
	obsfile = os.path.join(args.datadir, "obsmat.txt")
	destfile = os.path.join(args.datadir, "destinations.txt")

	# Parse homography matrix.
	H = np.loadtxt(Hfile)
	Hinv = np.linalg.inv(H)
	# Parse obstacle map.
	obs_map = ewap.create_obstacle_map(H, mapfile)
	# Parse pedestrian annotations.
	frames, timesteps, agents = ewap.parse_annotations(obsfile)
	# Parse destinations.
	destinations = np.loadtxt(destfile)
	
	# Play the video
	seqname = os.path.basename(args.datadir)
	cap = cv2.VideoCapture(os.path.join(args.datadir, seqname+".avi"))
	
# 	seekpos = 7.5 * 60 * 1000 # About 7 mins 30 secs
# 	endpos = 8.7 * 60 * 1000 # About 8 mins 40 secs
# 	cap.set(POS_MSEC, seekpos)
	cap.set(POS_FRAMES, 11300)
	pl.ion()
	pl.subplot(1,2,1)
	pl.title('Path Length')
	pl.subplot(1,2,2)
	pl.title('Minimum Safety')
	pl.draw()
	
	paths = []
	agent_num = 0
	last_t = -1
	while cap.isOpened():
		frame_num = int(cap.get(POS_FRAMES))
		now = int(cap.get(POS_MSEC) / 1000)
		_, frame = cap.read()
		
		# Draw the obstacles.
		frame = np.maximum(frame, cv2.cvtColor(obs_map, cv2.COLOR_GRAY2BGR))
		
		# Draw destinations.
		for d in destinations:
			d = np.append(d, 1)
			cv2.circle(frame, util.to_pixels(Hinv, d), 5, (0,255,0), -1)

		frame_txt = "{:0>2}:{:0>2}".format(now//60, now%60)

		# Check for end of annotations.
		if frame_num >= len(frames):
			frame_txt += ' (eof)'
		else:
			frame_txt += ' (' + str(frame_num) + ')'
			# If we've reached a new timestep, recompute the observations.
			t = frames[frame_num]
			if t >= 0 and t != last_t:
				last_t = t
				paths, true_paths, predictions, MAP = make_predictions(t, timesteps, agents)
				ped_scores, IGP_scores = calc_scores(true_paths, MAP)
				pl.subplot(1,2,1)
				pl.scatter(IGP_scores[:,0], ped_scores[:,0])
				pl.subplot(1,2,2)
				pl.scatter(IGP_scores[:,1], ped_scores[:,1])
				pl.draw()
# 				for i in range(ped_scores.shape[0]):
# 					print 'Agent', i, ': Pedestrian:', ped_scores[i], 'IGP:', IGP_scores[i]

		# Inform of the frame number.
		font = cv2.FONT_HERSHEY_SIMPLEX
		pt = (5, frame.shape[0]-10)
		scale = 0.6
		thickness = 1
		width, baseline = cv2.getTextSize(frame_txt, font, scale, thickness)
		baseline += thickness
		cv2.rectangle(frame, (pt[0], pt[1]+baseline), (pt[0]+width[0], pt[1]-width[1]-2), (0,0,0), -1)
		cv2.putText(frame, frame_txt, pt, font, scale, (0,255,0), thickness)
		
		# Draw in the pedestrians, if we have them.
		if paths:
			# The paths they've already taken.
			for path in paths:
				draw_path(frame, path, Hinv, (255,0,0))
			
			# The ground truth for a single agent.
			draw_path(frame, true_paths[agent_num%len(true_paths)], Hinv, (192,0,192))
			
			# The predictions for a single agent.
			if DRAW_FINAL_PATH:
				path = MAP[agent_num%len(MAP)]
				draw_path(frame, path, Hinv, (0,192,192))
			else: # draw individual samples
				for i in range(util.NUM_SAMPLES):
					path = predictions[agent_num%len(predictions)][:,i,:]
					path = np.column_stack((path, np.ones(path.shape[0])))
					draw_path(frame, path, Hinv, (0,192,192))
		
		cv2.imshow('frame', frame)
		key = cv2.waitKey(0) & 0xFF
		if key == ord('q') or key == ESC:
			break
		elif key == LEFT:
			cap.set(POS_FRAMES, frame_num-1)
		elif key == UP:
			agent_num += 1
			cap.set(POS_FRAMES, frame_num)
		elif key == DOWN:
			agent_num -= 1
			cap.set(POS_FRAMES, frame_num)
	
	cap.release()
	cv2.destroyAllWindows()

def make_predictions(t, timesteps, agents):
	peds = timesteps[t]
	past_paths = []
	true_paths = []
	predictions = []
	for ped in peds:
		# Get the full and past paths of the agent.
		fullpath = agents[ped]
		path_end = next(i for i,v in enumerate(fullpath[:,0]) if v==t)
		points = list(range(0,path_end+1))
		if path_end < fullpath.shape[0]:
			points += [-1] # Add the destination point.
		past_plus_dest = fullpath[np.ix_(points)]
		past_paths.append(past_plus_dest[:,1:4])
		true_paths.append(fullpath[path_end:,1:4])
		
		# Predict possible paths for the agent.
		t_future = fullpath[path_end:,0]
		gp = ParametricGaussianProcess(past_plus_dest, t_future, xkernel, ykernel)
		samples = gp.sample(util.NUM_SAMPLES)
		predictions.append(samples)
	
	weights = util.interaction(predictions)
	predictions = util.resample(predictions, weights)
	MAP = [get_final_path(p) for p in predictions]
	return (past_paths, true_paths, predictions, MAP)

def get_final_path(samples):
	return np.column_stack((np.mean(samples, 1), np.ones(samples.shape[0])))

def calc_score(path, other_paths):
	length = 0
	safety = inf
	prev_loc = None
	for t in range(len(path)):
		loc = path[t]
		if prev_loc is not None:
			length += util.dist(prev_loc, loc)
		prev_loc = loc
		for o in other_paths:
			if t < len(o):
				dist = util.dist(o[t], loc)
				if dist < safety:
					safety = dist
	return (length, safety)

def calc_scores(true_paths, MAP):
	robot_scores = np.array([calc_score(path, true_paths[:i]+true_paths[i+1:]) for i, path in enumerate(MAP)])
	ped_scores = np.array([calc_score(path, true_paths[:i]+true_paths[i+1:]) for i, path in enumerate(true_paths)])
	return ped_scores, robot_scores

def draw_path(frame, path, Hinv, color):
	prev = None
	for loc in path:
		loc = util.to_pixels(Hinv, loc)
		cv2.circle(frame, loc, 3, color, -1)
		if prev:
			cv2.line(frame, prev, loc, color, 1)
		prev = loc

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("datadir", help="The parent directory for the dataset to be used.")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	sys.exit(main())
