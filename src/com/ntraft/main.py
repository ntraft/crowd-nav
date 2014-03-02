'''
Created on Mar 1, 2014

@author: ntraft
'''
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

	# Parse homography matrix.
	H = ewap.parse_homography_matrix(Hfile)
	Hinv = np.linalg.inv(H)
	print Hinv
	# Parse obstacle map.
	obs_map = ewap.create_obstacle_map(H, mapfile)
	# Parse pedestrian annotations.
	annotations = ewap.parse_annotations(obsfile)
	
	# Play the video
	seqname = os.path.basename(args.datadir)
	cap = cv2.VideoCapture(os.path.join(args.datadir, seqname+".avi"))
	
	seekpos = START_TIME * 60 * 1000
	endpos = END_TIME * 60 * 1000
	cap.set(POS_MSEC, seekpos)
	now = cap.get(POS_MSEC)
	peds = np.array([])
	while cap.isOpened() and now < endpos:
		_, frame = cap.read()
		now = cap.get(POS_MSEC)
		frame_num = cap.get(POS_FRAMES)
		
		# Draw in the pedestrians.
		# TODO inform/halt if reached the end of annotation file.
		# TODO annotation coords are still off
		# TODO annotation timing seems off as well; seems ahead
		newpeds = annotations[annotations[:,0]==frame_num]
		if newpeds.size > 0:
			peds = newpeds
		for ped in peds:
			loc = ped[np.ix_([2,4,3])].transpose()
			loc = np.dot(Hinv, loc).astype(int)
			cv2.circle(frame, (loc[1], loc[0]), 5, (255,0,0), -1)
		
		cv2.imshow('frame', frame)
		if cv2.waitKey(40) & 0xFF == ord('q'):
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
