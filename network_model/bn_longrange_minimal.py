#!/usr/bin/env python

''' 
Simulating two dimensional rate network dr/dt = -r + [Mr + I]_+
r: activity
M: connectivity
I: external input
'''

import sys
import os
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from network_model import brain_network_t as bn
from network_model.tools import generate_noisy_mh
from network_model.tools import save_activity



## input parameters for connectivity
param1 = float(sys.argv[1])		## eccentricity
param2 = float(sys.argv[2])		## input modulation
param3 = float(sys.argv[3])		## gamma/recurrent strength
print('Parameter settings are:\n ecc={}, eta={}, gamma={}'.format(param1,param2,param3))
## input parameters for filename for storing activity patterns
param4 = float(sys.argv[4])		## VERSION
VERSION = param4



save_data = True
## Network settings
N, M = 60,60						## Size of network
nevents = 10						## Number of simulated events

start_activity_strength = 0.1		## amplitude of initial conditions
timepoints = 2						## number of time points of simulated activity to save

network_params = {'input_shape' 		: np.array([N, M]),
				  'mode'				: 'short_range',
				  'noise_type'			: 'postsyn',
				  'sigmax'				: 1.8,
				  'sigmax_sd'			: param1*0.15/0.8,
				  'ecc'					: param1,
				  'ecc_sd'				: 0.13*param1,
				  'orientation'			: 0,
				  'orientation_sd'		: 1.0,
				  'amplitude'			: 1.,
				  'inh_factor'			: 2.5,
				  'input_noise_level'	: param2,
				  'nonlinearity_rule'	: 'rectification',
				  'dt'					: 0.15,
				  'runtime'				: 3334,		## number of integration steps
				  'nonlin_fac'			: param3
				  }

if save_data:
	filename = 'activity_v{}'.format(VERSION)
	index = save_activity.gimme_index('{}.hdf5'.format(filename))
	#index = 0
else:
	index = 27752#np.random.randint(33440)
print('Index = {}'.format(index));sys.stdout.flush()

nonlinearity_rule = network_params['nonlinearity_rule']
total_t = network_params['runtime'] ### measured in units of tau, number of time steps

         
## generate connectivity matrix M
w_rec,copy_index = generate_noisy_mh.noisy_mh_wrap(N,M,network_params['mode'],
								   network_params['noise_type'],
								   network_params['sigmax'],
								   network_params['sigmax_sd'],
								   network_params['ecc'],
								   network_params['ecc_sd'],
								   network_params['orientation'],
								   network_params['orientation_sd'],
								   network_params['amplitude'],
								   network_params['inh_factor'],
								   pbc=True,		## periodic boundary conditions
								   index=index,		## determines random seed
								   version=VERSION)

	
## get estimate of spatial scale of pattern by looking at peak in spectrum of M
w_rec0 = abs(np.fft.fftshift(np.fft.fft2(w_rec[N//2,M//2,:,:])))
kmax_flat = np.argmax(w_rec0)
kmax = np.array([abs(kmax_flat%N-N//2),abs(kmax_flat//N-M//2)])
w_rec = w_rec.reshape(N*M,N*M)

## normalize M such that real part of maximal eigenvalue is given by network_params['nonlin_fac']
all_eigenvals = np.linalg.eigvals(w_rec)
max_eigenval = np.nanmax(np.real(all_eigenvals))
w_rec = network_params['nonlin_fac']*w_rec/np.real(max_eigenval)


## Input parameters
N_inp, M_inp = network_params['input_shape']
w_inp = np.diagflat(np.ones(N*M,dtype=float),k=0)
input_noise_level = network_params['input_noise_level']
rng_start_activity = np.random.RandomState(index)
input_rnd = rng_start_activity.rand(nevents,N_inp,M_inp)

''' use convolution with MH to get spatial scale in noisy input'''
x,y = np.meshgrid(np.linspace(-N//2+1,N//2,N),np.linspace(-M//2+1,M//2,M))
sig1 = network_params['sigmax']
sig2 = 2*sig1
kern1 = 1./(np.sqrt(np.pi*2)*sig1)**2*np.exp((-x**2-y**2)/2./sig1**2)
kern2 = 1./(np.sqrt(np.pi*2)*sig2)**2*np.exp((-x**2-y**2)/2./sig2**2)
x1d = np.linspace(-total_t//2+1,total_t//2,total_t)
sig11d = 100
sig21d = 2*sig11d
kern11d = 1./(np.sqrt(np.pi*2)*sig11d)*np.exp((-x1d**2)/2./sig11d**2)
kern21d = 1./(np.sqrt(np.pi*2)*sig21d)*np.exp((-x1d**2)/2./sig21d**2)

input_smo = np.real(np.fft.ifft2(np.fft.fft2(kern1-kern2)[None,:,:]*np.fft.fft2(input_rnd,axes=(1,2)), axes=(1,2)))
additive_noise = input_smo*input_noise_level
additive_noise = additive_noise.reshape(nevents,N_inp*M_inp)
constant_input = np.ones((N_inp*M_inp))



## setup network
integrator = 'runge_kutta'	#'forward_euler_constInp'	# runge_kutta	runge_kutta2
print('using',integrator);sys.stdout.flush()
bnn = bn.BrainNetwork(w_rec=w_rec,
							nonlinearity_rule=nonlinearity_rule,
							integrator=integrator,
							delta_t=network_params['dt'],
                            tau=1.,
                            tsteps=network_params['runtime'],
							data_type=np.float32,
							)




activity = np.empty((nevents, timepoints, N, M))
inputs = np.empty((nevents, N, M))
np.random.seed(index*2)
print( "run!");sys.stdout.flush()
for i in range(nevents):
	## initial conditions of activity
	start_activity = np.random.rand(N*M)*start_activity_strength

	iinputs = constant_input + additive_noise[i,:]
	iinputs[iinputs<0] = 0
	inputs[i,:,:] = iinputs.reshape(N,M)
	
	## simulate
	iactivity = bnn.run(iinputs.reshape(1,N*M), start_activity)
	iactivity = iactivity.reshape(total_t,N,M)	## shape is runtime x N x M
	
	## store only specific time points of simulated activity traces
	activity[i,:,:,:] = iactivity[total_t//timepoints-1::total_t//timepoints,:,:]

## comment out next two lines
plt.imshow(activity[0,-1,:,:],interpolation='nearest',cmap='binary')
plt.savefig('test.pdf')

print('Simulation done');sys.stdout.flush()

network_params.update({'inputs'		:	inputs})
network_params.update({'eigenvals'	:	all_eigenvals})
network_params.update({'kmax'		:	kmax})


''' save acticity'''
if save_data:
	save_activity.save_activity(activity, network_params, '{}.hdf5'.format(filename),\
	 '{}/'.format(index), additional_params_file='params_new_v{}'.format(VERSION))
print('Done');sys.stdout.flush()




