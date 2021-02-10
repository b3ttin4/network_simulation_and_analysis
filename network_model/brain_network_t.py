'''
Model for a "brain network". Template for use with numpy (np), theano (t), etc.
'''


### Needed for fixing theano problem with an old version, probably not necessary after theano update
from __future__ import print_function
import sys
import numpy
import numpy.distutils
import numpy.distutils.__config__
###
import numpy as np
import theano
import theano.tensor as T
import types

print(theano.__version__)
print(numpy.__version__)

from network_model.tools import bn_tools_t as bnt
from network_model.tools import integration_methods_t as im

floatX = theano.config.floatX
rng_noise = np.random.RandomState(1)

#theano.config.compute_test_value = 'warn'

# Network support
class BrainNetwork:
	def __init__(self, 
				w_rec, 
				nonlinearity_rule,
				integrator='forward_euler',
				delta_t=0.01, 
				tau=10.,
				tsteps=None,
				data_type=floatX):
		
		if type(data_type) == str:
			data_type = np.typeDict[data_type]
		self.data_type_np = data_type
		num_neurons = w_rec.shape[0]
		self.num_neurons = num_neurons
		
		num_frames = 1 # Unknown, placeholder value
		# Convert variables to new data type
		
		if w_rec is None:
			w_rec = np.zeros((num_neurons, num_neurons))
		w_rec = self.data_type_np(w_rec)
		
		
		delta_t = self.data_type_np(delta_t)
		tau = self.data_type_np(tau)
		if tsteps is not None:
			tsteps = self.data_type_np(tsteps)
		
		self._theano_init = False
		self.w_rec = w_rec.astype(self.data_type_np)

		self.delta_t = np.float32(delta_t)
		if tsteps is not None:
			self.tsteps = np.int32(tsteps)
		else:
			self.tsteps = None
		if isinstance(tau, np.ndarray):
			self.tau = tau.astype(self.data_type_np)
		else:
			self.tau = np.float32(tau)
		
		self.t_w_rec = theano.shared(self.w_rec)
		
		self.nonlinearity_rule = nonlinearity_rule
		self.integrator = integrator
		self.t_delta_t = theano.shared(self.delta_t)
		self.t_tsteps = theano.shared(self.tsteps)
		self.t_tau = theano.shared(self.tau)
		self.t_time_steps = theano.shared(0)
		
		
		self.inputs = None
		self.t_inputs = T.matrix('inputs')
		self.t_start_activity = T.vector('start_activity')
					
		self._init_nonlinearity()
		self._init_integrator()
		
		self._init_theano_computations()

		
	def update_w_rec(self, w_rec):
		self.t_w_rec.set_value(w_rec.astype(self.data_type_np))
		#if (not self.t_w_rec is None) and (w_rec.shape == self.t_w_rec.get_value().shape):
		#    print 'WARNING (bn_t): w_rec potentially as wrong shape'
		
	def _theano_function_update(self):
		self._theano_init = False
			
	def _init_nonlinearity(self):
		if self.nonlinearity_rule == 'linear': 
			self.t_nonlinearity = bnt.nl_linear_t
		elif self.nonlinearity_rule == 'rectification':
			self.t_nonlinearity = bnt.nl_rect_t
		elif self.nonlinearity_rule == 'shallow_rectification':
			self.t_nonlinearity = bnt.nl_shallow_rect_t
		elif self.nonlinearity_rule == 'clipping':
			self.t_nonlinearity = bnt.nl_clip_t
		elif self.nonlinearity_rule == 'sigmoid':
			self.t_nonlinearity = bnt.nl_sigmoid_t
		elif self.nonlinearity_rule == 'tanh':
			self.t_nonlinearity = bnt.nl_tanh_t
		elif self.nonlinearity_rule =='fermi':
			self.t_nonlinearity = bnt.nl_fermi_t
		elif self.nonlinearity_rule =='rectification_squared':
			self.t_nonlinearity = bnt.nl_rect_squared_t
		else:
			raise Exception('Unknown nonlinearity rule')
		self._theano_function_update()
	
	
	def inspect_inputs(self,i, node, fn):
		print(i, node, "input(s) value(s):", [input[0] for input in fn.inputs], end='')

	def inspect_outputs(self,i, node, fn):
		print(" output(s) value(s):", [output[0] for output in fn.outputs])


	
	def _init_integrator(self):
		if self.integrator == 'runge_kutta':
			self.integrator_function = im.runge_kutta_explicit
			#print('WARNING (bn_t): Using RK 2th order instead of 4th order')
		elif self.integrator == 'runge_kutta2':
			self.integrator_function = im.runge_kutta2
		elif self.integrator == 'forward_euler_constInp':
			self.integrator_function = im.forward_euler_constInp
		else:
			raise Exception("Unknown integrator ({0})".format(self.integrator))
		self._theano_function_update()
		
	def _init_theano_computations(self):

		if not self._theano_init:
			# Compute the activity by plugging the variables into an external
			# integrator function.
			self.t_activity = self.integrator_function(self.t_inputs,
													self.t_w_rec, 
													self.t_start_activity,
													self.t_delta_t,
													self.t_time_steps, 
													self.t_tau, 
													self.t_nonlinearity,
													data_type_np=self.data_type_np,
													k=self.t_tsteps) 
			

			self.tf_activity = theano.function([self.t_inputs,self.t_start_activity], self.t_activity,\
			on_unused_input='ignore')
			##,mode=theano.compile.MonitorMode(pre_func=self.inspect_inputs,post_func=self.inspect_outputs)

			self._theano_init = True
	def run(self, inputs, start_activity):

		if isinstance(inputs, types.FunctionType) or isinstance(inputs, types.MethodType):
			# Inputs is a function
			self.t_inputs = inputs
		else:
			# Inputs is an array
			inputs = inputs.astype(self.data_type_np)
		
		start_activity = start_activity.astype(self.data_type_np)
		
		self._init_theano_computations()        
		self._check_variables()
		

		activity = self.tf_activity(inputs,start_activity)
		
		# Store variables
		#self.inputs = inputs
		#self.activity = activity

		return activity
	
	def _check_variables(self):
		if (self.t_w_rec is None or
			self.nonlinearity_rule is None or
			self.integrator is None or
			self.t_delta_t is None or
			self.t_tau is None):
			raise Exception("Not all required settings set.")
		
