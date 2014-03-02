'''
Utility functions for dealing with the BIWI Walking Pedestrians dataset (or
EWAP for "ETH Walking Pedestrians"). From Pellegrini et al., ICCV 2009.

Created on Mar 1, 2014

@author: ntraft
'''
import numpy as np
from PIL import Image

def parse_homography_matrix(H_txt):
	return np.loadtxt(H_txt)

def create_obstacle_map(H, map_png):
	rawmap = np.array(Image.open(map_png))
	return rawmap # TODO transform into world coords

def parse_annotations(obsmat_txt):
	return np.loadtxt(obsmat_txt)
