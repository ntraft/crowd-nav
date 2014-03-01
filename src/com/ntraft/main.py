'''
Created on Mar 1, 2014

@author: ntraft
'''

from __future__ import division
import sys
import argparse
import numpy as np
import matplotlib
# The 'MacOSX' backend appears to have some issues on Mavericks.
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
import com.ntraft.ewap as ewap

def main():
	# Parse command-line arguments.
	args = parse_args()
	
	# Parse obstacle map.
	obs_map = ewap.create_obstacle_map(H, mapfile)
	# Parse pedestrian annotations.

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("left", help="The left-hand stereo image.")
	parser.add_argument("right", help="The right-hand stereo image.")
	parser.add_argument("truth", help="The ground truth depth field (from the perspective of the left image).")
	parser.add_argument("patchwidth", type=int, help="The width of the patch to use for normalized cross-correlation. Must be odd and at least 3.")
	args = parser.parse_args()
	if args.patchwidth < 3 or args.patchwidth % 2 == 0:
		parser.error("Invalid patch width.")
	return args

if __name__ == "__main__":
	sys.exit(main())