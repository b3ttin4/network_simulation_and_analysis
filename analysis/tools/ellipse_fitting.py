import numpy as np
import sys
from cv2 import fitEllipse,findContours,RETR_TREE,CHAIN_APPROX_NONE,pointPolygonTest

def get_ellipse_params(img,threshold,check_size=False,conversion_factor=1):
	h,w = img.shape
	centery,centerx = h//2,w//2
	
	notnan = np.isfinite(img)
	if np.sum(np.logical_not(notnan))>2:
		ythr,xthr = np.where((img>(threshold*0.9))*(img<(threshold*1.1)))
		ynan,xnan = np.where(img==np.logical_not(notnan))
		dist = (ythr[:,None]-ynan[None,:])**2 + (xthr[:,None]-xnan[None,:])**2
		if np.sum(np.nanmin(dist,axis=0)<1.1)>2:
			return None
	
	img_thr = (img>threshold).astype('uint8')
	
	above_thr = np.sum(img_thr)
	try:
		## Version 2 of cv2 gives two arguments
		cont,hier = findContours(img_thr,RETR_TREE,CHAIN_APPROX_NONE)
	except:
		## Version 3 of cv2 gives three arguments
		_,cont,hier = findContours(img_thr,RETR_TREE,CHAIN_APPROX_NONE)
	## if more than one contourline, find the one which goes around center and is longest
	if len(cont)>1:
		lengths = []
		is_inside = []
		for item in cont:
			lengths.append(len(item))
			is_inside.append( pointPolygonTest(item,(centery,centerx),measureDist=False)>=0 )
		if np.sum(is_inside)>1:
			this_cont = np.argmax(np.array(lengths)[np.array(is_inside)])
		elif np.sum(is_inside)==1:
			this_cont = np.where(np.array(is_inside))[0][0]
		else:
			return None
	else:
		this_cont = 0
	
	
	cnt = cont[this_cont]
	
	## need at least 4 points in contour to fit ellipse with 3 params
	if len(cnt)>4:
		ellipse = fitEllipse(cnt)
		too_big = (ellipse[1][0]>h or ellipse[1][1]>w)
		area = ellipse[1][0]*ellipse[1][1]
		too_small = False
		if area<3:
			too_small = True
		#if check_size:
			#too_big = False
		if (not too_big and not too_small):
			a2 = ellipse[1][0]**2
			b2 = ellipse[1][1]**2
			eccentricity = np.sqrt(abs(a2-b2)/max(np.array([a2,b2])))
			return ellipse,np.array([ellipse[2],eccentricity,max(ellipse[1])*conversion_factor,min(ellipse[1])*conversion_factor]),cnt,above_thr
		else:
			return None
	else:
		return None


def get_fit_ellipse(region, region_mod, threshold_mode,full_output=False):
	ellies_reg = {}
	check_ell = {}
	check_cnt = {}
	elli_thr = {}
	for i in range(region.shape[0]):
		iregion = region[i,:,:]
		try:
			if region_mod=='opm':
				check_size = True
				thr = threshold_mode*(iregion[iregion.shape[0]//2,iregion.shape[1]//2]-np.nanmin(iregion))+np.nanmin(iregion)
			else:
				check_size = False
				thr = threshold_mode
			e,ellipse,cnt,above_thr = get_ellipse_params(iregion,thr,check_size=check_size)
			
			if full_output:
				check_ell.update({str(i) : e})
				check_cnt.update({str(i) : cnt})
			
			ellipse2 = np.array([ellipse[0],ellipse[1],ellipse[2],ellipse[3]])
			ellies_reg.update({str(i) : ellipse2})
			elli_thr.update({str(i) : above_thr})
		except Exception as e:
			print("Exception",e)
			pass
	return ellies_reg,elli_thr,check_ell,check_cnt
