import numpy as np
from analysis.tools import find_local_maxima


def get_peaks_and_distance(cc, inds, roi, rough_patch_size, pbc=True):
	''' 
	detect peaks in correlation patterns
	output: values at peaks and distance to reference point
	'''
	corr2d = cc[roi.flatten(),:,:]
	corr2d = corr2d[inds,:,:]
	M,N = cc.shape[1:]
	vals,radius = [],[]

	yroi,xroi = np.where(roi)
	for ip,ipattern in enumerate(corr2d):
		this_idy,this_idx = yroi[inds[ip]],xroi[inds[ip]]
		
		maxima = find_local_maxima.detect_peaks(ipattern,rough_patch_size,roi=roi)
		## get rid of double counted peaks at edges
		maxima[:3,:] = 0
		maxima[:,:3] = 0
		maxima[-3:,:] = 0
		maxima[:,-3:] = 0
		idys,idxs = np.where(maxima)
		for y,x in zip(idys,idxs):
			if ipattern[y,x]<0:
				#print('negative peak',ip,y,x,ipattern[y,x])
				continue
			deltay = np.abs(this_idy - y)
			deltax = np.abs(this_idx - x)
			if pbc:
				delta = np.sqrt( (deltax - N*(deltax>(N//2)))**2 + (deltay - M*(deltay>(M//2)))**2 )
			else:
				delta = np.sqrt( deltax**2 + deltay**2 )
			if delta<3:
				continue
			vals.append( ipattern[y,x] )
			radius.append(delta)
		
		
	radius = np.array(radius)[np.isfinite(vals)]
	vals = np.array(vals)[np.isfinite(vals)]
	

	id_radius = np.argsort(radius)
	vals_sort = vals[id_radius]
	rad_sort = radius[id_radius]
	
	vals_sort = vals_sort[rad_sort>=1.0]
	rad_sort = rad_sort[rad_sort>=1.0]
	vals_sort = np.concatenate([np.array([1]), vals_sort])
	rad_sort = np.concatenate([np.array([0]), rad_sort])

	return rad_sort,vals_sort
	
