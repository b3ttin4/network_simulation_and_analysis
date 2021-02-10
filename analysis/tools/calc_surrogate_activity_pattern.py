import numpy as np
from analysis.tools import smooth_map

from cv2 import warpAffine,getRotationMatrix2D
from scipy.ndimage import gaussian_filter


def get_shifted_activity(data,Nsur,nframes,norm_activity,roi,surrogate_in_2d=False,seed=None,do_shift=False):
	'''
	data : 		activity frames, shape is: # frames x height x width
	Nsur:		# of surrogate frames per data frame
	nframes		# frames you actually want to have surrogates from (can be smaller than # of frames of data)
	norm_activity		normalise activity by this array (e.g. normalise sd to 1)
	roi 				roi
	surrogate_in_2d		if True output shape is: # of frames x Nsur x height x width else # of frames x Nsur x # pixels in roi
	seed				seed for random conversions on activity patterns
	do_shift:			if False activity patterns will not be shifted in x,y direction (only rotation and reflection)
	'''
	
	nall_frames,h,w = data.shape
	npatterns = np.sum(roi)
	idy,idx = np.where(roi)
	miny,maxy = min(idy),max(idy)
	minx,maxx = min(idx),max(idx)
	ctry,ctrx = (maxy-miny)//2+miny,(maxx-minx)//2+minx
	delta_gamma = 10		# rotate in units of 10 degrees


	data_sh = np.empty((nframes,Nsur,npatterns))*np.nan
	if seed is not None:
		np.random.seed(seed)
	gammas = np.random.choice(np.arange(0,1440,delta_gamma),size=Nsur*nframes,replace=True)	#4*360=1440
	gammas = gammas.reshape(Nsur,nframes)
	
	do_replace = nall_frames<nframes
	for isur in range(Nsur):
		selected_frames = np.random.choice(np.arange(nall_frames),size=nframes,replace=do_replace)#np.arange(nall_frames)#
		data_part = data[selected_frames,:,:]
		#if norm_activity is not None:
			#data_part = data_part/np.nanstd(data_part.reshape(nframes,h*w),axis=1)[:,None,None]*norm_activity[:,None,None]
		
		for im in range(nframes):
		### rotate and mirror original activity pattern
			frame = data_part[im,:,:]
			gamma = gammas[isur,im]

			''' rotation '''
			M = getRotationMatrix2D((ctrx,ctry),gamma%360,1)
			dst = warpAffine(frame,M,(w,h),borderValue=np.nan)
			
			''' translation (can make effective ROI smaller)'''
			if do_shift:
				np.random.seed(gamma)
				shift = np.random.choice(np.arange(-18,19,1),size=2, replace=False)		#changed on june 22th 2017, before: (-18,19,1)
				Mshift = np.float32([[1,0,shift[0]],[0,1,shift[1]]])
				dst = warpAffine(dst,Mshift,(w,h),borderValue=np.nan)
			
			''' reflection '''
			if gamma>720:
				flip = dst[miny:maxy+1,:]
				dst[miny:maxy+1,:] = flip[::-1,:]
			if ((gamma//delta_gamma)%2)!=0:
				flip = dst[:,minx:maxx+1]
				dst[:,minx:maxx+1] = flip[:,::-1]
			
			this_dst = dst[roi]
			if norm_activity is not None:
				this_dst = this_dst/np.nanstd(this_dst)*norm_activity[selected_frames[im]]
			data_sh[im,isur,:] = this_dst
			
			
	if surrogate_in_2d:
		data_sd_2d = np.empty(((nframes,Nsur)+roi.shape))*np.nan
		data_sd_2d[:,:,roi] = data_sh
		return data_sd_2d
	else:
		return data_sh
	
	

