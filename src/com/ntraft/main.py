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
	frames, timesteps, agents = ewap.parse_annotations(Hinv, obsfile)
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
	update_plot([], [])
	
	paths = []
	agent_num = 0
	agent_txt = ''
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
			agent_txt = ''
		else:
			frame_txt += ' (' + str(frame_num) + ')'
			# If we've reached a new timestep, recompute the observations.
			t = frames[frame_num]
			if t >= 0:
				curr_agent = timesteps[t][agent_num%len(timesteps[t])]
				agent_txt = 'Agent: {}'.format(curr_agent)
				if t >= 0 and t != last_t:
					last_t = t
					paths, true_paths, predictions, MAP = util.make_predictions(t, timesteps, agents)
					ped_scores, IGP_scores = util.calc_scores(true_paths, MAP)
					update_plot(ped_scores, IGP_scores)

		# Inform of the frame number.
		pt = (3, frame.shape[0]-3)
		ll, ur = draw_text(frame, pt, frame_txt)
		if agent_txt:
			pt = (ll[0], ur[1])
			draw_text(frame, pt, agent_txt)
		
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

def update_plot(ped_scores, IGP_scores):
	pl.clf()
	if len(ped_scores) > 0:
		pl.subplot(1,2,1)
		pl.title('Path Length')
		pl.xlabel('IGP'); pl.ylabel('Pedestrian')
		pl.scatter(IGP_scores[:,0], ped_scores[:,0])
		plot_diag()
		pl.subplot(1,2,2)
		pl.title('Minimum Safety')
		pl.xlabel('IGP'); pl.ylabel('Pedestrian')
		pl.scatter(IGP_scores[:,1], ped_scores[:,1])
		plot_diag()
		pl.draw()

def plot_diag():
	xmin, xmax = pl.xlim()
	ymin, ymax = pl.ylim()
	lim = (min(0, min(xmin, ymin)), max(xmax, ymax))
	pl.plot((0, 1000), (0, 1000), 'k')
	pl.xlim(lim); pl.ylim(lim)

def draw_text(frame, pt, frame_txt):
	font = cv2.FONT_HERSHEY_SIMPLEX
	scale = 0.6
	thickness = 1
	sz, baseline = cv2.getTextSize(frame_txt, font, scale, thickness)
	baseline += thickness
	lower_left = (pt[0], pt[1])
	pt = (pt[0], pt[1]-baseline)
	upper_right = (pt[0]+sz[0], pt[1]-sz[1]-2)
	cv2.rectangle(frame, lower_left, upper_right, (0,0,0), -1)
	cv2.putText(frame, frame_txt, pt, font, scale, (0,255,0), thickness)
	return lower_left, upper_right

def draw_path(frame, path, Hinv, color):
	prev = None
	for loc in ((int(y), int(x)) for x,y,z in path):
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
