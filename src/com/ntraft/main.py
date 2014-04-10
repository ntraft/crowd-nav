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
if sys.platform.startswith('darwin'):
	matplotlib.use('TkAgg')
import matplotlib.pyplot as pl

import com.ntraft.ewap as ewap
import com.ntraft.display as display
import com.ntraft.util as util

POS_MSEC = cv2.cv.CV_CAP_PROP_POS_MSEC
POS_FRAMES = cv2.cv.CV_CAP_PROP_POS_FRAMES
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ESC = 27

DRAW_FINAL_PATH = True

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
	obs_map = ewap.create_obstacle_map(mapfile)
	# Parse pedestrian annotations.
	frames, timeframes, timesteps, agents = ewap.parse_annotations(Hinv, obsfile)
	# Parse destinations.
	destinations = np.loadtxt(destfile)
	
	# Play the video
	seqname = os.path.basename(args.datadir)
	cap = cv2.VideoCapture(os.path.join(args.datadir, seqname+".avi"))
	disp = display.Display(cap, Hinv, obs_map, frames, timesteps, agents, destinations)
	disp.set_frame(11300)
# 	seekpos = 7.5 * 60 * 1000 # About 7 mins 30 secs
# 	endpos = 8.7 * 60 * 1000 # About 8 mins 40 secs
# 	cap.set(POS_MSEC, seekpos)
		
	pl.ion()
	display.update_plot([], [])
	
	while cap.isOpened():
		
		disp.do_frame()
		
		key = cv2.waitKey(0) & 0xFF
		if key == ord('r'):
			run_experiment(cap, disp, timeframes, timesteps, agents)
		elif key == ord('q') or key == ESC:
			break
		elif key == LEFT:
			disp.back_one_frame()
		elif key == UP:
			disp.agent_num += 1
			disp.reset_frame()
		elif key == DOWN:
			disp.agent_num -= 1
			disp.reset_frame()
	
	cap.release()
	cv2.destroyAllWindows()
	pl.close('all')

def run_experiment(cap, disp, timeframes, timesteps, agents):
	# We're going to compare ourselves to the agents with the following IDs.
	agents_to_test = range(319, 331)
	IGP_scores = []
	ped_scores = []
	display.update_plot(ped_scores, IGP_scores)
	for agent in agents_to_test:
		ped_path = agents[agent]
		path_length = len(ped_path)
		final_path = np.zeros((path_length, 3))
		# Run IGP through the whole path sequence.
		for i in range(1, path_length):
			frame_num = timeframes[ped_path[i,0]]
			disp.set_frame(frame_num)
			final_path[i] = disp.do_frame(False)
			if (cv2.waitKey(1) & 0xFF) != -1:
				return
		# Compute the final score for both IGP and pedestrian ground truth.
		start_time = ped_path[0,0]
		other_peds = [agents[a] for a in timesteps[start_time] if a != agent]
		other_paths = [util.get_path_at_time(start_time, fullpath)[1] for fullpath in other_peds]
		IGP_scores.append(util.calc_score(final_path, other_paths))
		ped_scores.append(util.calc_score(ped_path, other_paths))
		display.update_plot(ped_scores, IGP_scores)
	results = np.column_stack((agents_to_test, ped_scores, IGP_scores))
	np.savetxt('experiment.txt', results)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("datadir", help="The parent directory for the dataset to be used.")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	sys.exit(main())
