'''
Created on Dec 10, 2014

@author: ntraft
'''
from __future__ import division
import sys
import os
import argparse
import numpy as np

import com.ntraft.ewap as ewap
import com.ntraft.util as util

def main():
	# Parse command-line arguments.
	args = parse_args()
	
	Hfile = os.path.join(args.datadir, "H.txt")
	obsfile = os.path.join(args.datadir, "obsmat.txt")

	# Parse homography matrix.
	H = np.loadtxt(Hfile)
	Hinv = np.linalg.inv(H)
	# Parse pedestrian annotations.
	frames, timeframes, timesteps, agents = ewap.parse_annotations(Hinv, obsfile)
	
	# Agents 319-338
	timeline = range(11205, 11554)
	
	print 'Running experiment...'
	util.reset_timer()
	
	num_samples = 100
	total_samples = 0
	M = np.zeros((2,2))
	for frame in timeline:
		t = frames[frame]
		if t == -1: continue
		
		print '{:.1%} complete'.format((frame-timeline[0])/len(timeline))
		
		for _ in range(num_samples):
# 			predictions = util.make_predictions(t, timesteps, agents)
			predictions = util.fake_predictions(t, timesteps, agents, 100.0)
			for a,plan in enumerate(predictions.plan):
				if plan.shape[0] > 1:
					error = predictions.true_paths[a][1,0:2] - plan[1,0:2]
					M += np.outer(error, error)
					total_samples += 1
		# for each agent...
			# generate GP samples
			# for each sample...
				# calc difference from ground truth
				# calc variance (outer product)
				# add to M accumulator
	M /= total_samples
	entropy = 0.5*np.log((2*np.pi*np.e)**2 * np.linalg.det(M))
	
	print 'EXPERIMENT COMPLETE.'
	util.report_time()
	print 'entropy is', entropy

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("datadir", help="The parent directory for the dataset to be used.")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	sys.exit(main())
