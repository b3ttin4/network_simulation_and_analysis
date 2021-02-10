#!/usr/bin/python
import numpy as np
import os,sys
import h5py

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt

from network_model.tools import save_activity

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'


if __name__=="__main__":
	## version to analyse
	VERSION = int(sys.argv[1])
	
	file_path = save_activity.global_path + 'data2d/'
	image_path = save_activity.global_path + 'image/'
	
	listfiles = os.listdir(file_path)
	str_to_search = 'activity_v{}.0.'.format(VERSION)
	matched_files = np.array(listfiles)[np.array(['{}'.format(str_to_search) in string for string in listfiles])]
	
	
	save_path = image_path + 'activity_v{}/'.format(VERSION)
	for ifolder in ['activity','corr']:
		if not os.path.exists(save_path + '{}'.format(ifolder)):
			os.makedirs(save_path + '{}'.format(ifolder))
	
	frame_no = -1		## plot last time point of activity


	#activities = []
	for item in matched_files:
		network_output = h5py.File(file_path+item,"r")
		all_keys = np.array(sorted(network_output.keys())).astype(int)
		

		for iidx in sorted(network_output.keys()):
			iidx = str(iidx)
			try:
				activity = network_output[iidx]['activity'].value

				N,M = network_output[iidx]['shape'].value
				timepoints = activity.shape[1]
				npatterns = activity.shape[0]

				
				''' plot activity patterns of all events '''
				fig = plt.figure(figsize=(5,5))#()#
				n_patterns = 10
				for time in range(n_patterns):
					for event in range(n_patterns):
						i = event+time*n_patterns
						if i>=network_output[iidx]['nevents'].value:
							continue
						ax = fig.add_subplot(n_patterns,n_patterns,i+1)
						MU = np.nanmean(activity[i,frame_no,:,:])
						SD = np.nanstd(activity[i,frame_no,:,:])
						im=ax.imshow(activity[i,frame_no,:,:],interpolation='nearest',cmap='binary')#,vmin=0,vmax=MU+3*SD)

						ax.set_axis_off()
				fig.savefig(save_path+'activity/final_act_{}.pdf'.format(iidx),dpi=300,format='pdf',bbox_inches='tight')
				plt.close(fig)
				

				''' correlation over events'''
				activity_reshaped = activity[:,frame_no,:,:].reshape(network_output[iidx]['nevents'].value,N*M)
				cc = np.corrcoef(activity_reshaped,rowvar=0)
					
					
				''' plot correlation patterns of 100 randomly selected seed points'''
				ylim = 0.75
				fig = plt.figure(figsize=(50,50))
				ncols = 10
				for iy in range(ncols):
					for ix in range(ncols):
						i = ix+iy*ncols
						ax = fig.add_subplot(ncols,ncols,i+1)
						ax.set_axis_off()

						idx = np.random.randint(0,N*M,1)
						ax.imshow(cc[idx,:].reshape(N,M),interpolation='nearest',cmap='RdBu_r',vmin=-ylim,vmax=ylim)

						rect = plt.Circle(((idx)%N,(idx)//N),radius=1,color='chartreuse')
						ax.add_patch(rect)

				fig.savefig(save_path+'corr/corr_{}.pdf'.format(iidx),dpi=200,format='pdf',bbox_inches='tight')
				plt.close()
					
					
			except Exception as e:
				print(e)
			
		network_output.close()
	
	
print('The end')
	




