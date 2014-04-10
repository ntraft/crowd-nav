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
	frames, timeframes, timesteps, agents = ewap.parse_annotations(Hinv, obsfile)
	# Parse destinations.
	destinations = np.loadtxt(destfile)
	
	# Play the video
	seqname = os.path.basename(args.datadir)
	cap = cv2.VideoCapture(os.path.join(args.datadir, seqname+".avi"))
	disp = Display(cap, obs_map, frames, timesteps, agents, destinations)
	disp.set_frame(11300)
# 	seekpos = 7.5 * 60 * 1000 # About 7 mins 30 secs
# 	endpos = 8.7 * 60 * 1000 # About 8 mins 40 secs
# 	cap.set(POS_MSEC, seekpos)
		
	pl.ion()
	update_plot([], [])
	
	while cap.isOpened():
		disp.do_frame()
		key = cv2.waitKey(0) & 0xFF
		if key == ord('p'):
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

class Display:
	
	def __init__(self, cap, Hinv, obs_map, frames, timesteps, agents, destinations):
		self.cap = cap
		self.Hinv = Hinv
		self.obs_map = obs_map
		self.frames = frames
		self.timesteps = timesteps
		self.agents = agents
		self.destinations = destinations
		self.paths = []
		self.agent_num = 0
		self.agent_txt = ''
		self.last_t = -1
		self.draw_samples = False
		self.frame_num = 0
	
	def set_frame(self, frame):
		self.cap.set(POS_FRAMES, frame)

	def reset_frame(self):
		frame_num = int(self.cap.get(POS_FRAMES))
		self.set_frame(frame_num-1)

	def back_one_frame(self):
		frame_num = int(self.cap.get(POS_FRAMES))
		self.set_frame(frame_num-2)

	def do_frame(self, with_scores=True):
		if not self.cap.isOpen():
			raise Exception('Video stream closed.')
		
		t_plus_one = None
		frame_num = int(self.cap.get(POS_FRAMES))
		now = int(self.cap.get(POS_MSEC) / 1000)
		_, frame = self.cap.read()
		
		# Draw the obstacles.
		frame = np.maximum(frame, cv2.cvtColor(self.obs_map, cv2.COLOR_GRAY2BGR))
		
		# Draw destinations.
		for d in self.destinations:
			d = np.append(d, 1)
			cv2.circle(frame, util.to_pixels(self.Hinv, d), 5, (0,255,0), -1)
	
		frame_txt = "{:0>2}:{:0>2}".format(now//60, now%60)
	
		# Check for end of annotations.
		if frame_num >= len(self.frames):
			frame_txt += ' (eof)'
			agent_txt = ''
		else:
			frame_txt += ' (' + str(frame_num) + ')'
			# If we've reached a new timestep, recompute the observations.
			t = self.frames[frame_num]
			if t >= 0:
				displayed_agent = self.timesteps[t][self.agent_num%len(self.timesteps[t])]
				agent_txt = 'Agent: {}'.format(displayed_agent)
				if t >= 0 and t != self.last_t:
					self.last_t = t
					paths, true_paths, predictions, MAP = util.make_predictions(t, self.timesteps, self.agents)
					t_plus_one = MAP[1]
					if with_scores:
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
				draw_path(frame, path, (255,0,0))
			
			# The ground truth for a single agent.
			draw_path(frame, true_paths[self.agent_num%len(true_paths)], (192,0,192))
			
			# The predictions for a single agent.
			if self.draw_samples:
				for i in range(util.NUM_SAMPLES):
					path = predictions[self.agent_num%len(predictions)][:,i,:]
					path = np.column_stack((path, np.ones(path.shape[0])))
					draw_path(frame, path, (0,192,192))
			else: # just the planned path
				path = MAP[self.agent_num%len(MAP)]
				draw_path(frame, path, (0,192,192))
		
		cv2.imshow('frame', frame)
		return t_plus_one

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

def draw_path(frame, path, color):
	prev = None
	for loc in ((int(y), int(x)) for x,y,z in path):
		cv2.circle(frame, loc, 3, color, -1)
		if prev:
			cv2.line(frame, prev, loc, color, 1)
		prev = loc

def run_experiment(cap, disp, timeframes, timesteps, agents):
	agents_to_test = range(319, 331)
	IGP_scores = []
	ped_scores = []
	update_plot(ped_scores, IGP_scores)
	for agent in agents_to_test:
		ped_path = agents[agent]
		path_length = len(ped_path)
		final_path = np.zeros((path_length, 3))
		# Run IGP through the whole path sequence.
		for i in range(1, path_length):
			frame_num = timeframes[ped_path[i,0]]
			cap.set(POS_FRAMES, frame_num)
			final_path[i] = disp.do_frame(False)
			if (cv2.waitKey(1) & 0xFF) != -1:
				return
		# Compute the final score for both IGP and pedestrian ground truth.
		start_time = ped_path[0,0]
		other_peds = [agents[a] for a in timesteps[start_time] if a != agent]
		other_paths = [util.get_path_at_time(start_time, fullpath)[1] for fullpath in other_peds]
		IGP_scores.append(util.calc_score(final_path, other_paths))
		ped_scores.append(util.calc_score(ped_path, other_paths))
		update_plot(ped_scores, IGP_scores)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("datadir", help="The parent directory for the dataset to be used.")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	sys.exit(main())
