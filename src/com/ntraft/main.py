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
	disp.set_frame(11301)
# 	disp.set_frame(8289)
# 	disp.set_frame(9261)
# 	seekpos = 7.5 * 60 * 1000 # About 7 mins 30 secs
# 	endpos = 8.7 * 60 * 1000 # About 8 mins 40 secs
# 	cap.set(POS_MSEC, seekpos)
	
	pl.ion()
	display.plot_prediction_metrics([], [], [])
	
	while cap.isOpened():
		
		disp.do_frame()
		
		key = cv2.waitKey(0) & 0xFF
		if key == ord('e'):
			run_experiment(cap, disp, timeframes, timesteps, agents)
		elif key == ord('q') or key == ESC:
			break
		elif key == ord('r'):
			disp.redo_prediction()
		elif key == LEFT:
			disp.back_one_frame()
		elif key == UP:
			if disp.draw_all_agents: disp.sample_num += 1
			else: disp.agent_num += 1
			disp.reset_frame()
		elif key == DOWN:
			if disp.draw_all_agents: disp.sample_num -= 1
			else: disp.agent_num -= 1
			disp.reset_frame()
		elif key == ord('a'):
			disp.draw_all_agents = not disp.draw_all_agents
			disp.draw_all_samples = not disp.draw_all_samples
			disp.reset_frame()
		elif key == ord('t'):
			disp.draw_truth = not disp.draw_truth
			disp.reset_frame()
		elif key == ord('f'):
			disp.draw_plan = not disp.draw_plan
			disp.reset_frame()
		elif key == ord('p'):
			disp.draw_past = not disp.draw_past
			disp.reset_frame()
		elif key == ord('s'):
			disp.draw_samples = (disp.draw_samples + 1) % display.SAMPLE_CHOICES
			disp.reset_frame()
	
	cap.release()
	cv2.destroyAllWindows()
	pl.close('all')

def run_experiment(cap, disp, timeframes, timesteps, agents):
	# We're going to compare ourselves to the agents with the following IDs.
	print 'Running experiment...'
	util.reset_timer()
	agents_to_test = range(319, 331)
	true_paths = []
	planned_paths = []
# 	IGP_scores = np.zeros((len(agents_to_test), 2))
# 	ped_scores = np.zeros((len(agents_to_test), 2))
	display.plot_prediction_metrics([], [], [])
	for i, agent in enumerate(agents_to_test):
		ped_path = agents[agent]
		path_length = ped_path.shape[0]
		start = 3 # Initializes path with start+1 points.
		final_path = np.zeros((path_length, 3))
		final_path[0:start+1, :] = ped_path[0:start+1,1:4]
		# Run IGP through the path sequence.
		for t in range(start, path_length-1):
			frame_num = timeframes[int(ped_path[t,0])]
			print 'doing frame', frame_num
			disp.set_frame(frame_num)
			final_path[t+1] = disp.do_frame(agent, final_path[:t+1], False)
			if cv2.waitKey(1) != -1:
				print 'Canceled!'
				return
		# Compute the final score for both IGP and pedestrian ground truth.
		print 'Agent', agent, 'done.'
# 		start_time = int(ped_path[0,0])
# 		other_peds = [agents[a] for a in timesteps[start_time] if a != agent]
# 		other_paths = [util.get_path_at_time(start_time, fullpath)[1][:,1:4] for fullpath in other_peds]
# 		IGP_scores[i] = util.length_and_safety(final_path, other_paths)
# 		ped_scores[i] = util.length_and_safety(ped_path[:,1:4], other_paths)
		true_paths.append(ped_path[:,1:4])
		planned_paths.append(final_path)
		pred_errs = util.calc_pred_scores(true_paths, planned_paths, util.prediction_errors)
		path_errs = util.calc_pred_scores(true_paths, planned_paths, util.path_errors)
		display.plot_prediction_metrics(pred_errs, path_errs, agents_to_test[0:i+1])
# 	results = np.column_stack((agents_to_test, ped_scores, IGP_scores))
	pred_results = np.column_stack((agents_to_test, pred_errs.T))
	path_results = np.column_stack((agents_to_test, path_errs.T))
# 	np.savetxt('experiment.txt', results)
	np.savetxt('prediction_errors.txt', pred_results)
	np.savetxt('path_errors.txt', path_results)
	print 'EXPERIMENT COMPLETE.'
	util.report_time()

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("datadir", help="The parent directory for the dataset to be used.")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	sys.exit(main())
