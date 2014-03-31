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
import com.ntraft.ewap as ewap
import com.ntraft.util as util

POS_MSEC = cv2.cv.CV_CAP_PROP_POS_MSEC
POS_FRAMES = cv2.cv.CV_CAP_PROP_POS_FRAMES

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
	paths = []
	while cap.isOpened():
		_, frame = cap.read()
		now = int(cap.get(POS_MSEC) / 1000)
		frame_num = int(cap.get(POS_FRAMES))
		
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
			# If we've reached a new timestep, recompute the observations.
			frame_txt += ' (' + str(frame_num) + ')'
			t = frames[frame_num]
			if t >= 0:
				peds = timesteps[t]
				paths = []
				for ped in peds:
					fullpath = agents[ped]
					path_end = next(i for i,v in enumerate(fullpath[:,0]) if v==t)
					path = fullpath[0:path_end+1, 1:4]
					paths.append(path)

		# Inform of the frame number.
		font = cv2.FONT_HERSHEY_SIMPLEX
		pt = (5, frame.shape[0]-10)
		scale = 0.6
		thickness = 1
		width, baseline = cv2.getTextSize(frame_txt, font, scale, thickness)
		baseline += thickness
		cv2.rectangle(frame, (pt[0], pt[1]+baseline), (pt[0]+width[0], pt[1]-width[1]-2), (0,0,0), -1)
		cv2.putText(frame, frame_txt, pt, font, scale, (0,255,0), thickness)
		
		# Draw in the pedestrians.
		for path in paths:
			prev = None
			for loc in path:
				loc = util.to_pixels(Hinv, loc)
				cv2.circle(frame, loc, 3, (255,0,0), -1)
				if prev:
					cv2.line(frame, prev, loc, (255,0,0), 1)
				prev = loc
		
		cv2.imshow('frame', frame)
		key = cv2.waitKey(0)
		if key & 0xFF == ord('q'):
			break
	
	cap.release()
	cv2.destroyAllWindows()

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("datadir", help="The parent directory for the dataset to be used.")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	sys.exit(main())
