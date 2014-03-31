'''
Various utility functions.

@author: ntraft
'''
from __future__ import division
import numpy as np

def to_pixels(Hinv, loc):
	"""
	Given H^-1 and (x, y, z) in world coordinates, returns (c, r) in image
	pixel indices.
	"""
	loc = to_image_frame(Hinv, loc)
	return (loc[1], loc[0])

def to_image_frame(Hinv, loc):
	"""
	Given H^-1 and (x, y, z) in world coordinates, returns (u, v, 1) in image
	pixel indices.
	"""
	loc = np.dot(Hinv, loc) # to camera frame
	return (loc/loc[2]).astype(int) # to pixels (from millimeters)
