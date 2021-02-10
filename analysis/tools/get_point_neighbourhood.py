import numpy as np



def get_point_neighbourhood(points_of_interest,array3d,rough_patch_size):
	''' extracts region of size rough_patch_size x rough_patch_size around each point in  points_of_interest out of 3 or 3d array'''
	
	if isinstance(points_of_interest,list):
		idys,idxs = points_of_interest
	else:
		idys,idxs = np.where(points_of_interest)
	
	if array3d.ndim==3:
		npx,M,N = array3d.shape
		assert npx==len(idxs), 'npx not equal to len(idxs)'
		
		poi = np.ones((rough_patch_size*2+1,2*rough_patch_size+1))
		poi_full = np.zeros(array3d.shape[1:])
		offsety = array3d.shape[1]%2
		offsetx = array3d.shape[2]%2
		poi_full[array3d.shape[1]//2-rough_patch_size+offsety:array3d.shape[1]//2+rough_patch_size+1+offsety,array3d.shape[2]//2-rough_patch_size+offsetx:array3d.shape[2]//2+rough_patch_size+1+offsetx] = poi
		poi_fft = np.fft.fft2(poi_full)
		hood = np.zeros(array3d.shape)
		for i in range(npx):
			hood[i,idys[i],idxs[i]] = 1
		convolved = np.abs( np.fft.fftshift(np.fft.ifft2(np.fft.fft2(hood,axes=(1,2))*poi_fft[None,:,:],axes=(1,2)),axes=(1,2)))>0.5
		
		center = poi.size//2
		Ns = (2*rough_patch_size+1)
		part = np.zeros((npx,Ns*Ns))
		for i in range(npx):
			part[i,:] = array3d[i,convolved[i,:,:]]
			
			if np.nanmin([np.abs(M-idys[i]), idys[i]])<rough_patch_size:
				this_part = part[i,:].reshape(Ns,Ns)
				idy = idys[i]
				if idys[i]>rough_patch_size:
					idy = Ns - np.abs(N-idys[i])
				this_part = np.roll(this_part,rough_patch_size-idy,axis=0)
				part[i,:] = this_part.reshape(Ns**2)
			if np.nanmin([np.abs(N-idxs[i]), idxs[i]])<rough_patch_size:
				this_part = part[i,:].reshape(Ns,Ns)
				idx = idxs[i]
				if idxs[i]>rough_patch_size:
					idx = Ns - np.abs(N-idxs[i])
				this_part = np.roll(this_part,rough_patch_size-idx,axis=1)
				part[i,:] = this_part.reshape(Ns**2)
		return part.reshape(npx,Ns,Ns),convolved
	
	elif array3d.ndim==2:
		array2d = array3d
		h_new,w_new = array3d.shape
		neighbourhood = []
		for idx,idy in zip(idxs,idys):
			lowery = idy-rough_patch_size if (idy-rough_patch_size) > 0 else 0
			lowerx = idx-rough_patch_size if (idx-rough_patch_size) > 0 else 0
			pattern_part = array2d[lowery:idy+rough_patch_size+1,lowerx:idx+rough_patch_size+1]
			hp,wp = pattern_part.shape
			if (4*rough_patch_size+2-hp-wp) is not 0:
				pattern_part_padded = np.empty((2*rough_patch_size+1,2*rough_patch_size+1))*np.nan
				
				ly = -idy+rough_patch_size if (idy-rough_patch_size)<0 else 0
				ry = h_new-idy-rough_patch_size-1 if (idy+rough_patch_size+1)>h_new else None
				lx = -idx+rough_patch_size if (idx-rough_patch_size)<0 else 0
				rx = w_new-idx-rough_patch_size-1 if (idx+rough_patch_size+1)>w_new else None
				
				pattern_part_padded[ly:ry,lx:rx] = pattern_part
			else:
				pattern_part_padded = pattern_part
			neighbourhood.append(pattern_part_padded)
		return np.array(neighbourhood)

