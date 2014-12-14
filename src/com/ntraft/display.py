'''
Created on Apr 10, 2014

@author: ntraft
'''
from __future__ import division
import cv2
import numpy as np
import matplotlib.pyplot as pl

import com.ntraft.util as util

POS_MSEC = cv2.cv.CV_CAP_PROP_POS_MSEC
POS_FRAMES = cv2.cv.CV_CAP_PROP_POS_FRAMES

NO_SAMPLES = 0
PRIOR_SAMPLES = 1
POSTERIOR_SAMPLES = 2
SAMPLE_CHOICES = 3

'''
TODO

- New problem: cannot assume the robot will take the same time to arrive at
	the goal! Can we make sure the robot travels at the correct velocity?
	- Probably not, this is a problem that arises from having the GP be
		time-dependent rather than previous-state-dependent.
- Re-run experiment after fixing the above bugs.
- May still need to play with the interaction potential.
	- Check files Pete sent to see if his parameters are in there.
- Can maybe think about drawing other things like future paths or goals.
'''

class Display:
	
	def __init__(self, cap, Hinv, obs_map, frames, timesteps, agents, destinations):
		self.cap = cap
		self.Hinv = Hinv
		self.obs_map = obs_map
		self.frames = frames
		self.timesteps = timesteps
		self.agents = agents
		self.destinations = destinations
		self.predictions = util.empty_predictions
		self.agent_num = 0
		self.sample_num = 0
		self.agent_txt = ''
		self.last_t = -1
		self.draw_all_agents = False
		self.draw_all_samples = True
		self.draw_samples = NO_SAMPLES
		self.draw_truth = True
		self.draw_past = True
		self.draw_plan = True
	
	def set_frame(self, frame):
		self.cap.set(POS_FRAMES, frame)

	def back_one_frame(self):
		frame_num = int(self.cap.get(POS_FRAMES))
		self.set_frame(frame_num-2)

	def reset_frame(self):
		frame_num = int(self.cap.get(POS_FRAMES))
		self.set_frame(frame_num-1)

	def redo_prediction(self):
		self.last_t = -1
		self.reset_frame()

	def do_frame(self, agent=-1, past_plan=None, with_scores=True):
		if not self.cap.isOpened():
			raise Exception('Video stream closed.')
		
		t_plus_one = None
		frame_num = int(self.cap.get(POS_FRAMES))
		now = int(self.cap.get(POS_MSEC) / 1000)
		_, frame = self.cap.read()
	
		frame_txt = "{:0>2}:{:0>2}".format(now//60, now%60)
		if self.predictions.past:
			adex = self.agent_num % len(self.predictions.past)
		
		# Check for end of annotations.
		if frame_num >= len(self.frames):
			frame_txt += ' (eof)'
			self.agent_txt = ''
		else:
			frame_txt += ' (' + str(frame_num) + ')'
			# If we've reached a new timestep, recompute the observations.
			t = self.frames[frame_num]
			if t >= 0:
				if agent > -1:
					adex = next(i for i,v in enumerate(self.timesteps[t]) if v==agent)
				else:
					adex = self.agent_num % len(self.timesteps[t])
				displayed_agent = self.timesteps[t][adex]
				self.agent_txt = 'Agent: {}'.format(displayed_agent)
				if t != self.last_t:
					self.last_t = t
					self.predictions = util.make_predictions(t, self.timesteps, self.agents, agent, past_plan)
					if self.predictions.plan[adex].shape[0] > 1:
						t_plus_one = self.predictions.plan[adex][1]
					if with_scores:
# 						ped_scores, IGP_scores = util.calc_nav_scores(self.predictions.true_paths, self.predictions.plan)
# 						plot_nav_metrics(ped_scores, IGP_scores)
						pred_errs = util.calc_pred_scores(self.predictions.true_paths, self.predictions.plan, util.prediction_errors)
						path_errs = util.calc_pred_scores(self.predictions.true_paths, self.predictions.plan, util.path_errors)
						plot_prediction_metrics(pred_errs, path_errs, self.timesteps[t])
		
		# Draw the obstacles.
		frame = np.maximum(frame, cv2.cvtColor(self.obs_map, cv2.COLOR_GRAY2BGR))
		
		# Draw destinations.
		for d in self.destinations:
			d = np.append(d, 1)
			cv2.circle(frame, util.to_pixels(self.Hinv, d), 5, (0,255,0), -1)
	
		# Inform of frame and agent number.
		pt = (3, frame.shape[0]-3)
		ll, ur = draw_text(frame, pt, frame_txt)
		if self.agent_txt:
			pt = (ll[0], ur[1])
			ll, ur = draw_text(frame, pt, self.agent_txt)
		
		# Draw in the pedestrians, if we have them.
		if self.predictions.past:
			
			# The paths they've already taken.
			if self.draw_past:
				for path in self.predictions.past:
					draw_path(frame, path, (192,192,192))
			
			# For each agent, draw...
			peds_to_draw = range(len(self.predictions.plan)) if self.draw_all_agents else [adex]
			
			# The GP samples.
			if self.draw_samples != NO_SAMPLES:
				# What sample are we showing?
				if self.draw_all_samples:
					samples_to_draw = range(util.NUM_SAMPLES)
				else:
					sdex = self.sample_num % util.NUM_SAMPLES
					if sdex < 0: sdex = util.NUM_SAMPLES + sdex
					samples_to_draw = [sdex]
					pt = (ll[0], ur[1])
					ll, ur = draw_text(frame, pt, 'Sample: {}'.format(sdex+1))
					pt = (ll[0], ur[1])
					draw_text(frame, pt, 'Weight: {:.1e}'.format(self.predictions.weights[sdex]))
				# What kind of sample are we showing?
				preds = self.predictions.prior if self.draw_samples == PRIOR_SAMPLES else self.predictions.posterior
				# Now actually draw the sample.
				for ddex in peds_to_draw:
					for i in samples_to_draw:
						path = preds[ddex][:,i,:]
						path = np.column_stack((path, np.ones(path.shape[0])))
						draw_path(frame, path, (255,0,0))
						
			# The ground truth.
			if self.draw_truth:
				for ddex in peds_to_draw:
					draw_path(frame, self.predictions.true_paths[ddex], (0,255,0))
				
			# The final prediction.
			if self.draw_plan:
				for ddex in peds_to_draw:
					draw_path(frame, self.predictions.plan[ddex], (0,192,192))
					if not self.draw_all_agents: # past for all agents already drawn
						if past_plan is not None:
							draw_waypoints(frame, past_plan, (0,192,192))
						else:
							draw_waypoints(frame, self.predictions.past[ddex], (255,211,176))
		
		cv2.imshow('frame', frame)
		return t_plus_one

def plot_prediction_metrics(prediction_errors, path_errors, agents):
	pl.figure(1, (10,10))
	pl.clf()
	if len(prediction_errors) > 0:
		pl.subplot(2,1,1)
		pl.title('Prediction Error')
		pl.xlabel('Time (frames)'); pl.ylabel('Error (px)')
		m = np.nanmean(prediction_errors, 1)
		lines = pl.plot(prediction_errors)
		meanline = pl.plot(m, 'k--', lw=4)
		pl.legend(lines + meanline, ['{}'.format(a) for a in agents] + ['mean'])
		
		pl.subplot(2,1,2)
		pl.title('Path Error')
		pl.xlabel('Time (frames)'); pl.ylabel('Error (px)')
		m = np.nanmean(path_errors, 1)
		lines = pl.plot(path_errors)
		meanline = pl.plot(m, 'k--', lw=4)
		pl.legend(lines + meanline, ['{}'.format(a) for a in agents] + ['mean'])
		
		pl.draw()

def plot_nav_metrics(ped_scores, IGP_scores):
	pl.clf()
	if len(ped_scores) > 0:
		pl.subplot(1,2,1)
		pl.title('Path Length (px)')
		pl.xlabel('IGP'); pl.ylabel('Pedestrian')
		pl.scatter(IGP_scores[:,0], ped_scores[:,0])
		plot_diag()
		
		pl.subplot(1,2,2)
		pl.title('Minimum Safety (px)')
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
	cv2.rectangle(frame, lower_left, upper_right, (0,0,0), -1, cv2.CV_AA)
	cv2.putText(frame, frame_txt, pt, font, scale, (0,255,0), thickness, cv2.CV_AA)
	return lower_left, upper_right

def crossline(curr, prev, length):
	diff = curr - prev
	if diff[1] == 0:
		p1 = (int(curr[1]), int(curr[0]-length/2))
		p2 = (int(curr[1]), int(curr[0]+length/2))
	else:
		slope = -diff[0]/diff[1]
		x = np.cos(np.arctan(slope)) * length / 2
		y = slope * x
		p1 = (int(curr[1]-y), int(curr[0]-x))
		p2 = (int(curr[1]+y), int(curr[0]+x))
	return p1, p2

def draw_path(frame, path, color):
	if path.shape[0] > 0:
		prev = path[0]
		for curr in path[1:]:
			loc1 = (int(prev[1]), int(prev[0])) # (y, x)
			loc2 = (int(curr[1]), int(curr[0])) # (y, x)
			p1, p2 = crossline(curr, prev, 3)
			cv2.line(frame, p1, p2, color, 1, cv2.CV_AA)
			cv2.line(frame, loc1, loc2, color, 1, cv2.CV_AA)
			prev = curr

def draw_waypoints(frame, points, color):
	for loc in ((int(y), int(x)) for x,y,z in points):
		cv2.circle(frame, loc, 3, color, -1, cv2.CV_AA)
