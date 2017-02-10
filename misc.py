import numpy as np
import tensorflow as tf

def get_channels(data):

	"""Returns all values from data dict with keys starting with 'ch' """
	
	if 'byte_channels' in data.keys():
		return ecg_preprocessing.unflac(data['byte_channels'])
	else:
		return [data[k] for k in channels_names(data)]
		
def write_channels(data, channels):

	"""Rewrites channels to appropriate fields in data dict"""

	channels_keys = channels_names(data)
	assert len(channels) == len(channels_keys)
	
	for k, channel in zip(channels_keys, channels):
		data[k] = channel 
		
	return data

def convert_channels_to_flac(data):

	"""Pops channels field in data dict and places flac-compressed 
		`byte_channels` instead.
	"""
	channels = get_channels(data)
	for k in channels_names(data):
		data.pop(k)

	data['byte_channels'] = ecg_preprocessing.flac(channels)
	return data

def channels_names(data):
	return [k for k in sorted(data.keys()) if k[:2] == 'ch']

def convert_channels_from_easi(channels, names):

	"""Transforms 3 EASI ECG channels to some of 12 classical channels.

	Args:

		channels: list of 3 EASI channels in the following order: ES, AS, AI.
		names: list string containing names of desired channels.

	Raises: 

		ValueError: if names is not a valid list (with valid channels names), or
			`channels` is not a list of 3 EASI channels.
		TypeError: if name is not an instance of list."""

	conversion_matrix = {'I': 	np.array([[ 0.026, -0.174,  0.701]]),
						'II': 	np.array([[-0.002,  1.098, -0.763]]),
						'III': 	np.array([[-0.028,  1.272, -1.464]]),
						'aVR': 	np.array([[-0.012, -0.462,  0.031]]),
						'aVL': 	np.array([[ 0.027, -0.723,  1.082]]),
						'aVF': 	np.array([[-0.015,  1.185, -1.114]]),
						'V1': 	np.array([[ 0.641, -0.391,  0.080]]),
						'V2': 	np.array([[ 1.229, -1.050,  1.021]]),
						'V3': 	np.array([[ 0.947, -0.539,  0.987]]),
						'V4': 	np.array([[ 0.525,  0.004,  0.841]]),
						'V5': 	np.array([[ 0.179,  0.278,  0.630]]),
						'V6': 	np.array([[-0.043,  0.431,  0.213]]),
						'ES':	np.array([[ 1.000,  0.000,  0.000]]),
						'AS':	np.array([[ 0.000,  1.000,  0.000]]),
						'AI':	np.array([[ 0.000,  0.000,  1.000]])}

	if not len(channels) == 3 or not isinstance(channels, list):
		raise ValueError('`channels` should be a list of 3 EASI channels.')		 
	if not isinstance(names, list):
		raise TypeError('`names` should be a list of strings.')
	if not set(names) <= set(conversion_matrix.keys()):
		raise ValueError('Names list contains invalid channel(s) name(s).')		
	if set(names) & {'ES', 'AS', 'AI'}:
		print('Warning: Conversion for EASI channels is just an identity transform.')

	W = tf.placeholder(tf.float32, [1, 3])
	x = tf.placeholder(tf.float32, [None, len(channels)])
	res = tf.matmul(x, W, transpose_b=True)
	sess = tf.Session()

	new_channels = []
	for name in names:
		new_channels.append(np.squeeze(sess.run(res, 
						feed_dict={x: np.vstack(channels).T,
								   W: conversion_matrix[name]}), 1).astype(np.float16))

	return new_channels
