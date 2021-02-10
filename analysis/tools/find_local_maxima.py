import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion



def detect_peaks(image,local_neighbours,roi=None):
	"""
	Takes an image and detect the peaks using the local maximum filter.
	Returns a boolean mask of the peaks (i.e. 1 when
	the pixel's value is the neighborhood maximum, 0 otherwise)
	"""

	# define an 8-connected neighborhood
	#neighborhood = generate_binary_structure(2,2)
	neighborhood = np.ones((local_neighbours,local_neighbours),dtype=bool)

	#apply the local maximum filter; all pixel of maximal value 
	#in their neighborhood are set to 1
	local_max = maximum_filter(image, footprint=neighborhood, output=None, mode='reflect', origin=0)==image
	#local_max is a mask that contains the peaks we are 
	#looking for, but also the background.
	#In order to isolate the peaks we must remove the background from the mask.

	#we create the mask of the background
	background = (image==0)

	#a little technicality: we must erode the background in order to 
	#successfully subtract it from local_max, otherwise a line will 
	#appear along the background border (artifact of the local maximum filter)
	eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

	#we obtain the final mask, containing only peaks, 
	#by removing the background from the local_max mask
	detected_peaks = np.logical_xor(local_max, eroded_background)
	
	if roi is not None:
		detected_peaks[np.logical_not(roi)] = 0

	return detected_peaks