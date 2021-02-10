#!/usr/bin/python

'''
calculates spatial scale of the activity patterns, determines fractures,
the spatial scale of the correlation structure, the eccentricity of the
local correlations, and the dimensionality of the activity patterns.
'''

import numpy as np
import os
import sys
import h5py


import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

import cv2
from skimage.morphology import skeletonize
from scipy.ndimage import measurements, binary_fill_holes, binary_erosion, binary_dilation,label
from scipy.optimize import curve_fit
from scipy.ndimage.morphology import distance_transform_edt
from scipy.interpolate import interp2d

from analysis.tools import auto_correlation,find_local_maxima,get_point_neighbourhood,\
ellipse_fitting,calc_surrogate_activity_pattern,get_peak_corr_vals,\
dimension_abbott,smooth_map

from network_model.tools import save_activity

def expfct_full(x,tau,a):
    return np.exp(-x/tau)*(1.-a)+a



matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
sz = 25	#fontsize
lw = 2	#linewidth
fontdict = {'fontsize' : sz}


if __name__=="__main__":
	## version to analyse
	VERSION = int(sys.argv[1])
	
	file_path = save_activity.global_path + 'data2d/'
	image_path = save_activity.global_path + 'image/'
	
	## SCRIPT SETTINGS ##
	save_data = True			## save computed data in hdf5 file
	frame_no = -1				## which activity frame to analyse (per event)
	
	## choose which analyses to run
	spatial_scale_analysis = True
	fracture_analysis = True
	elli_analysis = True
	long_range_analysis = True
	dimensionality_analysis = True
	
	
	
	## loading and storing paths
	listfiles = os.listdir(file_path)	
	str_to_search = 'params_new_v{}.0.hdf5'.format(VERSION)	#	'corr_matrix' 
	matched_files = np.array(listfiles)[np.array(['{}'.format(str_to_search) in string for string in listfiles])]
	
	for ifolder in ['activity','envelope','fracture','corr']:
		if not os.path.exists(image_path + 'activity_v{}/{}'.format(VERSION,ifolder)):
			os.makedirs(image_path + 'activity_v{}/{}'.format(VERSION,ifolder))

	
	'''fracture processing parameters'''
	filter_thr = 0
	delta = 1

	## folder name of generated pdfs
	where_to = 'activity_v{}'.format(VERSION)
	if save_data:
		full_name = file_path + 'fracture_analysis_v{}.hdf5'.format(VERSION)
		vals_to_plot = h5py.File(full_name,'a')
	
		
	for item in matched_files:
		network_params = h5py.File(file_path+item,"r")
		network_output = h5py.File(file_path+'activity_v{}.0.hdf5'.format(VERSION),"r")
		all_keys = np.array(sorted(network_output.keys())).astype(int)
		
		for iidx in all_keys:
			try:
				iidx = str(iidx)
				print('*** {} ***'.format(iidx));sys.stdout.flush()

				ecc = network_params[iidx]['ecc'].value
				if True:
					## activity shape = events x number of frames x N x M
					activity = network_output[iidx]['activity'].value
					N,M = network_params[iidx]['shape'].value
					final_pattern = activity[:,frame_no,:,:]
					npatterns = final_pattern.shape[0]
					
					
					''' correlation over time'''
					cc = np.corrcoef(final_pattern.reshape(npatterns,N*M),rowvar=0)

					roi = np.sum(np.isfinite(cc),axis=0)>0
					roi = roi.reshape(M,N)
					bnd,end = 0,N
					nid = np.sum(roi)
					
					''' estimate wavelength '''
					sigma1 = network_params[iidx]['sigmax'].value
					sigma2 = network_params[iidx]['inh_factor'].value*sigma1
					wavelength_mh = np.sqrt( 4*np.pi**2/4.*(sigma1**2-sigma2**2)/np.log(sigma1/sigma2) )
					wavelength_pxl = 1./np.sqrt( np.sum((1.*network_params[iidx]['kmax'].value/np.array([N,M]))**2) )
					
					
					''' gradient of corr patterns '''
					normed = (cc - np.nanmean(cc,axis=0)[None,:])/np.nanstd(cc,axis=0)[None,:]
					normed = normed.reshape(N*M,N,M)
					grad_x = 1-np.nanmean(normed*np.roll(normed,1,axis=1),axis=0)
					grad_y = 1-np.nanmean(normed*np.roll(normed,1,axis=2),axis=0)
					grad = np.sqrt((grad_x)**2 + (grad_y)**2 )
					
				
					estimate_wavelength = 1.*wavelength_mh
					'''estimate spatial scale of activity patterns'''
					if (spatial_scale_analysis or save_data):
						try:
							estimate_wavelength = vals_to_plot['{}/{}'.format(iidx , 'est_wavelength')].value
							wavelength_mh = vals_to_plot['{}/{}'.format(iidx , 'wavelength')].value
						except:
							idcs = np.random.choice(np.arange(npatterns),size=npatterns,replace=False)
							autocorr = auto_correlation.get_autocorr(final_pattern[idcs,:,:],max_lag=N//2,method='wiener_khinchin')
							
							rough_patch_size = int(np.nanmax([wavelength_mh//2,7]))	#410microns in data
							maxima = np.zeros_like(autocorr,dtype=int)
							for i,iautocorr in enumerate(autocorr):
								maxima[i,:,:] = find_local_maxima.detect_peaks(iautocorr,rough_patch_size,roi=None)
							maxima[:,M//2,N//2] = 0
							
							coord_x,coord_y = np.meshgrid(np.arange(-(N//2),N//2+1,1),np.arange(-(M//2),M//2+1,1))
							distance_to_center = np.sqrt((1.*coord_x)**2 + (1.*coord_y)**2)
							
							maxima_sum = np.sum(maxima,axis=0)
							distances = distance_to_center[maxima_sum>0]
							estimate_wavelength = np.nanmin(distances)
							
							ring = (distance_to_center<(estimate_wavelength*1.3))*(distance_to_center>(estimate_wavelength*0.8))
							maxima[:,np.logical_not(ring)] = 0
							
							estimate_wavelength = np.nanmean(distance_to_center[np.sum(maxima,axis=0)>0])
							
							if save_data:
								try:
									vals_to_plot.create_dataset('{}/{}'.format(iidx , 'wavelength'), data=wavelength_mh)
									vals_to_plot.create_dataset('{}/{}'.format(iidx , 'est_wavelength'), data=estimate_wavelength)
								except:
									dset=vals_to_plot['{}'.format(iidx)]
									for jkey in ['wavelength','est_wavelength']:
										if jkey in dset.keys():
											del dset[jkey]
									dset['wavelength'] = wavelength_mh
									dset['est_wavelength'] = estimate_wavelength
									
						print('***');sys.stdout.flush()
						print('Spatial scale Lambda = {:.2f} (unit pixels)'.format(estimate_wavelength))
						print('*** Spatial scale analysis done ***');sys.stdout.flush()
						
					''' same processing as experimental data for finding fractures'''
					if fracture_analysis:
						print('start fracture_analysis')
						grad = cv2.medianBlur(grad.astype('float32'), 3)
						grad_with_nan = np.copy(grad)
						ynan,xnan = np.where(np.logical_not(np.isfinite(grad)))
						for iy,ix in zip(ynan,xnan):
							grad[iy,ix] = np.nanmean(grad[[(iy-1)%M,iy,(iy+1)%M]*3,[(ix-1)%N,ix,(ix+1)%N,ix,(ix+1)%N,(ix-1)%N,(ix+1)%N,(ix-1)%N,ix]])
						
						''' histogramm span full range from a to b, histogram normalisation/stretching'''
						mingrad = np.nanmin(grad)
						maxgrad = np.nanmax(grad)
						a,b = 0,256
						grad_normalised = (grad-mingrad)/(maxgrad-mingrad)*(b-a)+a		#a=-1,b=1

						clahe = cv2.createCLAHE(clipLimit=20, tileGridSize=(10,10) )
						grad_normalised = clahe.apply(grad_normalised.astype('uint8'))
						
						''' apply a highpass filter'''
						grad_filter = smooth_map.high_normalize(grad_normalised, mask=None, sigma=15)#, sig_low=2)

						''' Thresholding + remove small objects(morphology) '''
						furrow = grad_filter > filter_thr		#(1-np.exp(-4.))
						''' get rid of holes '''
						furrow_erosion = np.copy(furrow)
						labels,num_features = measurements.label(furrow, np.ones((3,3),dtype=int))
						for ifeature in range(1,num_features+1):
							furrow_part = np.sum(labels==ifeature)
							if furrow_part<3:
								labels[labels==ifeature]=0
						furrow_dilated = labels>0
						furrow_skeleton = skeletonize(furrow_dilated)
						furrow_skeleton_mask = furrow_skeleton*np.isfinite(grad_with_nan)
						
						distance_map_fg = distance_transform_edt(np.logical_not(furrow_skeleton_mask))
						try:
							wavelength_pxl = estimate_wavelength
						except:
							wavelength_pxl = np.nanmax(distance_map_fg)*2
						max_peaks = distance_map_fg>(wavelength_pxl/7.)

						
						'''fracture strength'''
						grad[np.logical_not(np.isfinite(grad_with_nan))] = np.nan
						dist_furrow = grad[furrow_skeleton_mask]
						dist_peaks = grad[max_peaks]
						strength_val = np.nanmedian(dist_furrow[np.isfinite(dist_furrow)]) - np.nanmedian(dist_peaks[np.isfinite(dist_peaks)])
						
						''' plotting fracture'''
						if True:
							fig=plt.figure()
							ax = fig.add_subplot(111)
							cmap = 'binary'
							MU = np.nanmean(grad)
							SD = np.nanstd(grad)
							im=ax.imshow(grad*estimate_wavelength*10/1000.,interpolation='nearest',cmap=cmap,vmin=0,vmax=0.1)
							plt.colorbar(im)
							fig.savefig(image_path + '{}/fracture/fract_{}_010.pdf'.format(where_to,iidx),dpi=200,format='pdf')
							plt.close(fig)


						if save_data:
							try:
								vals_to_plot.create_dataset('{}/{}'.format(iidx , 'strength_val'), data=strength_val)
							except:
								dset=vals_to_plot['{}'.format(iidx)]
								for jkey in ['strength_val']:
									if jkey in dset.keys():
										del dset[jkey]
								dset['strength_val'] = strength_val
						
						print('***');sys.stdout.flush()
						print('Fracture strength = {:.5f} (per Lambda)'.format(strength_val*estimate_wavelength*10/1000.))
						print('*** Fracture analysis done ***');sys.stdout.flush()
					
					'''compare ellipse props on fractures from regions away from fractures'''
					if elli_analysis:
						cc = cc.reshape(M,N,M,N)
						local_patch_size = 9
						interpolate_by = 3
						
						try:
							elli_params = vals_to_plot['{}/{}'.format(iidx , 'elli_params')].value
						except:
							print('No ellipse_params found. Refit ellipses!');sys.stdout.flush()
							if interpolate_by>1:
								local_patch_size *= 3
								
								cc2d = np.copy(cc)
								roi_float = roi.astype(float)
								cc2d[np.logical_not(np.isfinite(cc2d))] = 0
								roiy,roix = np.where(roi)
								
								roi_fft = np.pad( np.fft.fftshift(np.fft.fft2(roi_float)),((N,N),(N,N)),'constant')
								roi_intp = np.real(np.fft.ifft2(np.fft.fftshift(roi_fft))*9)
								roi_intp = roi_intp>0.5
								
								if (final_pattern.shape[1]>50 and estimate_wavelength<16):
									local_patch_size = 5*3
									corrs = []
									for irow in range(M):
										if np.sum(roiy==irow)==0:
											continue
										cc2d_fft = np.pad( np.fft.fftshift(np.fft.fft2(cc2d[irow,:,:,:],axes=(1,2)),axes=(1,2)),((0,0),(N,N),(N,N)), 'constant')
										cc2d_intp = np.real(np.fft.ifft2(np.fft.fftshift(cc2d_fft,axes=(1,2)),axes=(1,2))*9)
										cc2d_intp[:,np.logical_not(roi_intp)] = np.nan
										cc2d_intp = cc2d_intp[roi[irow,:],:,:]
									
										points_of_interest = [roiy[roiy==irow]*3,roix[roiy==irow]*3]
										icorrs,convolved = get_point_neighbourhood.get_point_neighbourhood(points_of_interest,cc2d_intp,local_patch_size)
										if len(corrs)==0:
											corrs = icorrs
										else:
											corrs = np.concatenate([corrs,icorrs])
									corrs = np.array(corrs)
									corrs = corrs.reshape(nid,2*local_patch_size+1,2*local_patch_size+1)
								else:
									cc2d = cc2d[roi,:,:]
									cc2d_fft = np.pad( np.fft.fftshift(np.fft.fft2(cc2d,axes=(1,2)),axes=(1,2)),((0,0),(N,N),(N,N)), 'constant')
									cc2d_intp = np.real(np.fft.ifft2(np.fft.fftshift(cc2d_fft,axes=(1,2)),axes=(1,2))*9)
									cc2d_intp[:,np.logical_not(roi_intp)] = np.nan
								
									points_of_interest = [roiy*3,roix*3]
									corrs,convolved = get_point_neighbourhood.get_point_neighbourhood(points_of_interest,cc2d_intp,local_patch_size)
							else:
								cc2d_intp=cc[roi,:,:]
								points_of_interest = roi
								corrs,convolved = get_point_neighbourhood.get_point_neighbourhood(points_of_interest,cc2d_intp,local_patch_size)
							
							
							fig = plt.figure()
							for i in range(6):
								for j in range(6):
									ax = fig.add_subplot(6,6,i*6+j+1)
									ax.imshow(corrs[i*6+j+1,:,:],interpolation='nearest',cmap='RdBu_r',vmin=-0.75,vmax=0.75)
							fig.savefig(image_path + '{}/fracture/corrs_{}.pdf'.format(where_to,iidx),dpi=200,format='pdf')
							plt.close(fig)
							
							
							p1 = 1
							np.random.seed(46756)
							part = np.random.choice(np.arange(2),size=nid,replace=True,p=[1-p1,p1]).astype(bool)
							threshold_mode = 0.7
							
							ellies_corr,ellies_thr,check_ellc,check_cntc = ellipse_fitting.get_fit_ellipse(corrs[part,:,:].copy(),\
							 'corr', threshold_mode, full_output=False)
							''' check how many local regions were fitted '''
							keysc = ellies_corr.keys()
							print('Fits in corr={} of total={}'.format(len(keysc),np.sum(part)));sys.stdout.flush()
							
							elli_params = np.empty((4,N,M))*np.nan
							yroi,xroi = np.where(roi)
							for ikey in sorted(ellies_corr.keys()):
								elli_params[0,yroi[int(ikey)],xroi[int(ikey)]] = ellies_corr[ikey][0]/180*np.pi		#ori in rad
								elli_params[1,yroi[int(ikey)],xroi[int(ikey)]] = ellies_corr[ikey][1]				#ecc
								elli_params[2,yroi[int(ikey)],xroi[int(ikey)]] = ellies_corr[ikey][2]				#a (height)
								elli_params[3,yroi[int(ikey)],xroi[int(ikey)]] = ellies_corr[ikey][3]				#b (width)
							eccs = np.argsort(elli_params[1,roi])

							try:
								del ellies_corr
							except:
								pass
						
						mean_ecc = np.nanmean(elli_params[1,bnd:end,bnd:end])
						sd_ecc = np.nanstd(elli_params[1,bnd:end,bnd:end])
						
						if fracture_analysis:
							yf,xf = np.where(furrow_skeleton_mask)
							distance_from_fracture = distance_transform_edt(np.logical_not(furrow_skeleton_mask))
							not_fracture = distance_from_fracture>(wavelength_pxl/7.)
							ynf,xnf = np.where(not_fracture)
							nfrct = len(yf)
							if np.sum(not_fracture)>nfrct:
								idx_not = np.random.choice(np.arange(np.sum(not_fracture)),size=nfrct,replace=False)
								ynf = ynf[idx_not]
								xnf = xnf[idx_not]
						
							## difference in and variability of eccentricity values 
							## between locations with fracture (xf,yf) and without (xnf,ynf)
							diff_ecc = np.nanmean(elli_params[1,ynf,xnf])-np.nanmean(elli_params[1,yf,xf])
							err_ecc = np.sqrt(np.nanstd(elli_params[1,ynf,xnf])**2 + np.nanstd(elli_params[1,yf,xf])**2)
							
							
						if save_data:
							try:
								if fracture_analysis:
									vals_to_plot.create_dataset('{}/{}'.format(iidx , 'diff_ecc'), data=diff_ecc)
									vals_to_plot.create_dataset('{}/{}'.format(iidx , 'err_ecc'), data=err_ecc)
								vals_to_plot.create_dataset('{}/{}'.format(iidx , 'mean_ecc'), data=mean_ecc)
								vals_to_plot.create_dataset('{}/{}'.format(iidx , 'sd_ecc'), data=sd_ecc)
								vals_to_plot.create_dataset('{}/{}'.format(iidx , 'elli_params'), data=elli_params)
							except:
								dset=vals_to_plot['{}'.format(iidx)]
								for jkey in ['diff_ecc','mean_ecc','sd_ecc','err_ecc','elli_params']:
									if jkey in dset.keys():
										del dset[jkey]
								if fracture_analysis:
									dset['diff_ecc'] = diff_ecc
									dset['err_ecc'] = err_ecc
								dset['mean_ecc'] = mean_ecc
								dset['sd_ecc'] = sd_ecc
								dset['elli_params'] = elli_params
						cc = cc.reshape(N*M,N*M)
						
						print('***');sys.stdout.flush()
						print('Mean eccentricity = {:.2f} (per Lambda)'.format(mean_ecc))
						print('*** Local correlation done***');sys.stdout.flush()	
					
					''' long-range decay of correlation pattern'''
					if long_range_analysis:
						bins = 1.*np.arange(0,76,3)/estimate_wavelength
						cc = cc.reshape(N*M,N,M)
						part = 1

						inds = np.random.choice(np.arange(nid),size=nid//part,replace=False)
						rough_patch_size = int(estimate_wavelength//2)	#410microns in data
						if (N>50 and (N/estimate_wavelength)>7):
							threshold_lo = 60./estimate_wavelength#units of pixel
							threshold_hi = 65./estimate_wavelength
							thr_lo_near = 20./estimate_wavelength
							thr_hi_near = 25./estimate_wavelength
						else:
							threshold_lo = 2.
							threshold_hi = 2.5
							thr_lo_near = 2.
							thr_hi_near = 2.5
							
						pbc = True		## periodic boundary conditions of activity patterns
						rad_sort,vals_sort = get_peak_corr_vals.get_peaks_and_distance(cc, inds, roi, rough_patch_size, pbc=pbc)
						rad_sort = 1.*rad_sort/estimate_wavelength
						
						## correlation at 2 Lambda
						med_decay_to = np.nanmean(vals_sort[(rad_sort>thr_lo_near)*(rad_sort<thr_hi_near)])


						'''shuffle activity to get baseline'''
						norm_activity = None
						Nsur = 10			## number of surrogate datasets
						npatterns_sh = npatterns
						med_sh,med_near,med_near_std, med_std = [],[],[],[]
						for isur in range(Nsur):
							shuffles = calc_surrogate_activity_pattern.get_shifted_activity(final_pattern,\
							1,npatterns_sh,norm_activity,roi,surrogate_in_2d=True,do_shift=False)
						
							cc_sh = np.corrcoef(shuffles[:,0,:,:].reshape(npatterns_sh,N*M),rowvar=0)
							cc_sh = cc_sh.reshape(N*M,N,M)

							inds = np.random.choice(np.arange(nid),size=nid//part,replace=False)
							rad_sort_sh,vals_sort_sh = get_peak_corr_vals.get_peaks_and_distance(cc_sh,\
							 inds, roi, rough_patch_size, pbc=False)
							rad_sort_sh = 1.*rad_sort_sh/estimate_wavelength
							med_sh.append( np.nanmean(vals_sort_sh[(rad_sort_sh>threshold_lo)*(rad_sort_sh<threshold_hi)]) )
							med_near.append( np.nanmean(vals_sort_sh[(rad_sort_sh>thr_lo_near)*(rad_sort_sh<thr_hi_near)]) )

						med_decay_to_sh = np.nanmean(med_sh)
						med_decay_to_near = np.nanmean(med_near)
							
						''' fit to decay based on baseline from near distances'''
						tau_guess = 1.0
						def expfct_near(x,tau): return expfct_full(x,tau,med_decay_to_near)
						pexp_near,covexp = curve_fit(expfct_near,rad_sort[rad_sort<thr_hi_near],vals_sort[rad_sort<thr_hi_near],p0=[tau_guess])
						
						''' fit to decay based on baseline from far distances'''
						def expfct_baseline_sh(x,tau): return expfct_full(x,tau,med_decay_to_sh)
						pexp_sh,covexp_sh = curve_fit(expfct_baseline_sh,rad_sort[rad_sort<threshold_hi],vals_sort[rad_sort<threshold_hi],p0=[tau_guess])
						
						
						'''Figure'''
						fig_lrd=plt.figure(figsize=(22,5))
						ax_lrd = fig_lrd.add_subplot(121)
						
						if 1000<rad_sort.size:
							idcs = np.random.choice(np.arange(np.sum(rad_sort.size)),size=1000,replace=False)
						else:
							idcs = np.arange(np.sum(rad_sort.size))
						#rad_sort = rad_sort#*estimate_wavelength
						ax_lrd.plot(rad_sort[idcs],vals_sort[idcs],'ok',markersize=3,rasterized=True)
						maxradius = np.nanmax(rad_sort)
						x = np.arange(0,maxradius,0.1)
						ax_lrd.plot(x,expfct_baseline_sh(x,pexp_sh[0]),'--c',label='Decay_sh = {:.2f}'.format(pexp_sh[0]),lw=lw)
						ax_lrd.plot(x,expfct_near(x,pexp_near[0]),'--m',label='near = {:.2f}'.format(pexp_near[0]),lw=lw)
						ax_lrd.legend(loc='best')
						ax_lrd.set_ylim(0.01,1)
						ax_lrd.set_xlim(0,3.5)#(0,7)
						
						
						rad_sort_sh = rad_sort_sh
						ax_lrd = fig_lrd.add_subplot(122)
						if 1000<rad_sort_sh.size:
							idcs = np.random.choice(np.arange(np.sum(rad_sort_sh.size)),size=1000,replace=False)
						else:
							idcs = np.arange(np.sum(rad_sort_sh.size))
						ax_lrd.plot(rad_sort_sh[idcs],vals_sort_sh[idcs],'og',markersize=3,rasterized=True)
						ax_lrd.plot([np.nanmin(rad_sort_sh[idcs]),np.nanmax(rad_sort_sh[idcs])],[med_decay_to_sh]*2,'-c')
						ax_lrd.plot([np.nanmin(rad_sort_sh[idcs]),np.nanmax(rad_sort_sh[idcs])],[med_decay_to_near]*2,'-m')
						ax_lrd.set_ylim(0.0,1)
						ax_lrd.set_xlim(0,3.5)#(0,7)
						fig_lrd.savefig(image_path + '{}/envelope/env_{}.pdf'.format(where_to,int(iidx)),dpi=200,format='pdf')
						

						print('pexp',iidx,'Decay_sh = {:.3f}'.format(pexp_sh[0]),'Decay = {:.3f}'.format(pexp_near[0]));sys.stdout.flush()
						
						if save_data:
							try:
								vals_to_plot.create_dataset('{}/{}'.format(iidx , 'med_decay_to'), data=med_decay_to)
								vals_to_plot.create_dataset('{}/{}'.format(iidx , 'decay_const'), data=pexp_sh[0])
								vals_to_plot.create_dataset('{}/{}'.format(iidx , 'med_decay_to_sh'), data=med_decay_to_sh)
								vals_to_plot.create_dataset('{}/{}'.format(iidx , 'med_decay_near'), data=med_decay_to_near)
								vals_to_plot.create_dataset('{}/{}'.format(iidx , 'decay_const_near'), data=pexp_near[0])
							except:
								dset=vals_to_plot['{}'.format(iidx)]
								for jkey in ['med_decay_to','decay_const', 'med_decay_to_sh','decay_const_near','med_decay_near']:
									if jkey in dset.keys():
										del dset[jkey]
								dset['med_decay_to'] = med_decay_to
								dset['decay_const'] = pexp_sh[0]
								dset['med_decay_to_sh'] = med_decay_to_sh
								dset['decay_const_near'] = pexp_near[0]
								dset['med_decay_near'] = med_decay_to_near
						
						print('***     ***');sys.stdout.flush()						
						print('Spatial scale of correlations = {:.2f}'.format(pexp_near[0]))
						print('Correlation at 2 Lambda (baseline subtracted) = {:.2f}'.format(med_decay_to-med_decay_to_near))
						print('*** Long range decay done ***');sys.stdout.flush()

					'''dimensionality of activity'''
					if dimensionality_analysis:
						''' effective dimensionality of activity between pixels '''
						eff_dim,ew = dimension_abbott.calc_dimension(final_pattern, inp='patterns',output=1)
						
						''' effective dimension of activity patterns '''
						final_pattern = final_pattern.reshape(npatterns,N*M)
						eff_dim_act, ew_act = dimension_abbott.calc_dimension(final_pattern.T, inp='patterns',output=1)

						if save_data:
							try:
								vals_to_plot.create_dataset('{}/{}'.format(iidx , 'eff_dim_pxl'), data=eff_dim)
								vals_to_plot.create_dataset('{}/{}'.format(iidx , 'ew'), data=ew)
								vals_to_plot.create_dataset('{}/{}'.format(iidx , 'eff_dim_act'), data=eff_dim_act)
								vals_to_plot.create_dataset('{}/{}'.format(iidx , 'ew_act'), data=ew_act)
							except:
								dset=vals_to_plot['{}'.format(iidx)]
								for jkey in [\
								'eff_dim_pxl','fix_eff_dim_pxl','fix_ew',\
								'ew','ew_act','eff_dim_act']:
									if jkey in dset.keys():
										del dset[jkey]
								dset['eff_dim_pxl'] = eff_dim
								dset['ew'] = ew
								dset['eff_dim_act'] = eff_dim_act
								dset['ew_act'] = ew_act
								
						print('***');sys.stdout.flush()
						print('Dimensionality = {:.0f}'.format(eff_dim))
						print('*** Cluster score done ***');sys.stdout.flush()	
						
			except Exception as e:
				print('EXCEPTION',e)
				
		network_params.close()
		network_output.close()
	
	if save_data:
		vals_to_plot.close()


	
print('The end')




