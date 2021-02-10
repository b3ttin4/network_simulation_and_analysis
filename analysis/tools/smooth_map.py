#!/usr/bin/python

'''
filter functions
'''

import numpy as np
import scipy.ndimage as snd
import warnings
warnings.filterwarnings("ignore")

import analysis.tools.conf as c

def low_normalize(frame, mask=None, sigma=c.sigma_high):
	''' apply lowpass filter to frame
	specify mask and standard deviation sigma of gaussian filter
	'''
	# recursion for frames
	if len(frame.shape)>2:
		result=np.empty_like(frame)
		for i in range(frame.shape[0]):
			result[i]=low_normalize(frame[i],mask,sigma)
		return result
		
	# recursion for complex images
	if np.iscomplexobj(frame):
		result=np.empty_like(frame)
		result.real=low_normalize(frame.real,mask,sigma)
		result.imag=low_normalize(frame.imag,mask,sigma)
		return result
		
	if mask is None:
		mask=np.ones(frame.shape,dtype=np.bool)

	data = np.copy(frame)
	m = np.zeros(mask.shape)
	m[mask] = 1.0
	m[~np.isfinite(frame)] = 0.
	data[np.logical_not(m)] = 0.
	normalized_data = 1.*snd.gaussian_filter(data,sigma,mode='constant',cval=0)/snd.gaussian_filter(m,sigma,mode='constant',cval=0)
	normalized_data[np.logical_not(mask)]=np.nan
	return normalized_data

def high_normalize(frame, mask=None, sigma=c.sigma_high):
	''' apply highpass filter to frame
	specify mask and standard deviation sigma of gaussian filter
	'''
	# recursion for frames
	if len(frame.shape)>2:
		result=np.empty_like(frame)
		for i in range(frame.shape[0]):
			result[i]=high_normalize(frame[i],mask,sigma)
		return result
		
	# recursion for complex images
	if np.iscomplexobj(frame):
		result=np.empty_like(frame)
		result.real=high_normalize(frame.real,mask,sigma)
		result.imag=high_normalize(frame.imag,mask,sigma)
		return result
		
	if mask is None:
		mask=np.ones(frame.shape,dtype=np.bool)

	data = np.copy(frame)
	m = np.zeros(mask.shape)
	m[mask] = 1.0
	m[~np.isfinite(frame)] = 0.
	data[np.logical_not(m)] = 0.
	normalized_data = data - 1.*snd.gaussian_filter(data,sigma,mode='constant',cval=0)/snd.gaussian_filter(m,sigma,mode='constant',cval=0)
	normalized_data[np.logical_not(mask)]=np.nan
	return normalized_data
	

def lowhigh_normalize(frame, mask=None, sig_high=c.sigma_high, sig_low=c.sigma_low):
	''' apply bandpass filter to frame
	specify mask and standard deviations sig_high (highpass) and sig_low (lowpass) of gaussian filters
	'''
	# recursion for frames
	if len(frame.shape)>2:
		result=np.empty_like(frame)
		for i in range(frame.shape[0]):
			result[i]=lowhigh_normalize(frame[i],mask, sig_high, sig_low)
		return result
		
	# recursion for complex images
	if np.iscomplexobj(frame):
		result=np.empty_like(frame)
		result.real=lowhigh_normalize(frame.real,mask,sig_high,sig_low)
		result.imag=lowhigh_normalize(frame.imag,mask,sig_high,sig_low)
		return result
	
	if mask is None:
		mask=np.ones(frame.shape,dtype=np.bool)
	data = np.copy(frame)
	m = np.zeros(mask.shape)
	m[mask] = 1.0
	m[~np.isfinite(frame)] = 0.
	data[np.logical_not(m)] = 0.
	m2 = np.copy(m)
	
	## gaussian low pass
	low_mask = snd.gaussian_filter(m2,sig_low,mode='constant',cval=0)
	low_data = 1.*snd.gaussian_filter(data,sig_low,mode='constant',cval=0)/low_mask
	
	low_data[np.logical_not(m)] = 0
	high_mask = snd.gaussian_filter(m,sig_high,mode='constant',cval=0)
	highlow_data = low_data - 1.*snd.gaussian_filter(low_data,sig_high,mode='constant',cval=0)/high_mask
	highlow_data[np.logical_not(mask)]=np.nan
	return highlow_data




	


