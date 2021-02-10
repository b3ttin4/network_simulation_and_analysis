import h5py
import numpy as np
import os

## where to store files
global_path = os.environ["HOME"] + "/output/"
if not os.path.exists(global_path):
	os.makedirs(global_path)
	print("Creating folder {}".format(global_path))

def save_activity(activity, network_params, filename, folder_index, activity_key='activity', additional_params_file='params_new_v0',dim=2):
	filepath = global_path + 'data{}d/'.format(dim)
	full_name = filepath + filename
	f = h5py.File(full_name,'a')
	
	if dim==2:
		if activity.ndim>3:
			N,M = activity.shape[2:]
		else:
			N,M = activity.shape[:2]
	else:
		N = activity.shape[2]

	
	f.create_dataset(folder_index + activity_key, data=activity)
	if activity_key=='activity':
		if dim==2:
			f.create_dataset(folder_index + 'shape', data=np.array([N,M]))
		else:
			f.create_dataset(folder_index + 'shape', data=np.array([N]))
		f.create_dataset(folder_index + 'nevents', data=activity.shape[0])
	
	for key in network_params.keys():
		f.create_dataset(folder_index + key, data=network_params[key])

	f.close()
	print('f close')
	
	
	'''network settings'''
	if additional_params_file:
		full_name = filepath + '{}.hdf5'.format(additional_params_file)
		f = h5py.File(full_name,'a')
		
		if dim==2:
			if activity.ndim>3:
				N,M = activity.shape[2:]
			else:
				N,M = activity.shape[:2]
		else:
			N = activity.shape[2]
			
		if dim==2:
			f.create_dataset(folder_index + 'shape', data=np.array([N,M]))
		else:
			f.create_dataset(folder_index + 'shape', data=np.array([N]))
		f.create_dataset(folder_index + 'nevents', data=activity.shape[0])
		
		for key in network_params.keys():
			if key in ('activity','inputs','eigenvals'):
				continue
			f.create_dataset(folder_index + key, data=network_params[key])
		f.close()
		print('f2 close')
		
	
		

def gimme_index(filename,dim=2):
	filepath = global_path + 'data{}d/'.format(dim)
	if not os.path.exists(filepath):
		print("creating {}".format(filepath))
		os.mkdir(filepath)
	full_name = filepath + filename
	print('Save under: {}'.format(full_name))
	try:
		f = h5py.File(full_name,'a')
		indices = [int(item) for item in sorted(f.keys())]
		max_index = np.max(indices)
		f.create_group('{}'.format(max_index+1))
		f.close()
	except Exception as e:
		#print(e)
		max_index = -1
	return max_index + 1
	

