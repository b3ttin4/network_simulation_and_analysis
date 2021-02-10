import numpy as np

''' for definition see abbott, rajan, sompolinsky, 2011 '''

def calc_dimension(array, inp='covariance',output=0):
	if inp=='covariance':
		w = np.linalg.eigvalsh(array)		# eigenvalues in ascending order
	elif inp=='patterns':
		array = array.reshape(array.shape[0],np.prod(array.shape[1:]))
		array[np.logical_not(np.isfinite(array))] = 0
		
		## interested in pixel covariance matrix <r_i(t)r_j(t)>_t
		array_norm = (array - np.nanmean(array,axis=0)[None,:])
		s = np.linalg.svd(array_norm,compute_uv=False)
		w = s**2
	else:
		print('Input should either be covariance matrix or activity patterns!')
		w = None
	eff_dim = np.sum(w)**2/np.sum(w**2)
	if output==0:
		return eff_dim
	elif output==1:
		return eff_dim,w

