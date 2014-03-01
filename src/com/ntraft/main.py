'''
Created on Mar 1, 2014

@author: ntraft
'''
from __future__ import division
import sys
import os
import argparse
import numpy as np
import matplotlib
# The 'MacOSX' backend appears to have some issues on Mavericks.
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import com.ntraft.ewap as ewap

def main():
	# Parse command-line arguments.
	args = parse_args()
	
	Hfile = os.path.join(args.datadir, "H.txt")
	mapfile = os.path.join(args.datadir, "map.png")
	obsfile = os.path.join(args.datadir, "obsmat.txt")

	# Parse homography matrix.
	H = ewap.parse_homography_matrix(Hfile)	
	# Parse obstacle map.
	obs_map = ewap.create_obstacle_map(H, mapfile)
	# Parse pedestrian annotations.
	annotations = ewap.parse_annotations(obsfile)

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("datadir", help="The parent directory for the dataset to be used.")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	sys.exit(main())