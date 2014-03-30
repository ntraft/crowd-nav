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

POS_MSEC = cv2.cv.CV_CAP_PROP_POS_MSEC
POS_FRAMES = cv2.cv.CV_CAP_PROP_POS_FRAMES
START_TIME = 7.5 # About 7 mins 30 secs
END_TIME = 8.7 # About 8 mins 40 secs

def main():
	# Parse command-line arguments.
	args = parse_args()
	
	Hfile = os.path.join(args.datadir, "H.txt")
	mapfile = os.path.join(args.datadir, "map.png")
	obsfile = os.path.join(args.datadir, "obsmat.txt")
	destfile = os.path.join(args.datadir, "destinations.txt")

	# Parse homography matrix.
	H = ewap.parse_homography_matrix(Hfile)
# 	H = np.column_stack((H, np.array([0,0,0])))
# 	H = np.row_stack((H, [0,0,0,1]))
	Hinv = np.linalg.inv(H)
	print H
	print Hinv
	print np.allclose(np.dot(H, Hinv), np.eye(3))
	print np.allclose(np.dot(Hinv, H), np.eye(3))
	L = 1
	sx = 1
	sy = 1
	# Parse obstacle map.
	obs_map = ewap.create_obstacle_map(H, mapfile)
	# Parse pedestrian annotations.
	annotations = ewap.parse_annotations(obsfile)
	destinations = ewap.parse_annotations(destfile)
	
	# Play the video
	seqname = os.path.basename(args.datadir)
	cap = cv2.VideoCapture(os.path.join(args.datadir, seqname+".avi"))
	
	seekpos = START_TIME * 60 * 1000
	endpos = END_TIME * 60 * 1000
# 	cap.set(POS_MSEC, seekpos)
	cap.set(POS_FRAMES, 11300)
	now = cap.get(POS_MSEC)
	peds = np.array([])
	while cap.isOpened() and now < endpos:
		_, frame = cap.read()
		now = cap.get(POS_MSEC)
		frame_num = cap.get(POS_FRAMES)
		print frame_num
		centerX = frame.shape[0]/2
		centerY = frame.shape[1]/2
		intrinsic = np.array([	[L/sx, 0, -centerX],
								[L/sy, 0, -centerY],
								[   0, 0,        1]])
		
		# Draw the obstacles.
		frame = np.maximum(frame, cv2.cvtColor(obs_map, cv2.COLOR_GRAY2BGR))
		
		# Draw destinations.
		for d in destinations:
			d = np.append(d, 1)
# 			d = np.append(d, 1)
			d = np.dot(Hinv, d).astype(int)
			d /= d[2]
			cv2.circle(frame, (d[1], d[0]), 5, (0,255,0), -1)
		
		# Draw in the pedestrians.
		# TODO inform/halt if reached the end of annotation file.
		# TODO annotation coords are still off
		# TODO annotation timing seems off as well; seems ahead
		newpeds = annotations[annotations[:,0]==frame_num]
		if newpeds.size > 0:
			peds = newpeds
		for ped in peds:
			loc = ped[np.ix_([2,4,3])].transpose()
			loc[2] = 1
			loc = np.dot(Hinv, loc) # to camera frame
			print 'before', loc
# 			loc = np.dot(intrinsic, loc) # to image plane
			loc /= loc[2] # to pixels (from millimeters)
			print 'after', loc
			loc = loc.astype(int) # discretize
			cv2.circle(frame, (loc[1], loc[0]), 5, (255,0,0), -1)
		
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
