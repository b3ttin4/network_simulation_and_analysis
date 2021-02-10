'''
GENERAL INFORMATION ABOUT THEANO IMPLEMENTATION:
Any numeric inputs have to be theano variables (that includes activity, proc_inputs, etc.) 
'''
import numpy as np
import theano
import theano.tensor as T
from . import im_runge_kutta as imrk
from theano.tensor.shared_randomstreams import RandomStreams

# rng = np.random.RandomState(123)
# theano_rng = RandomStreams(rng.randint(2**30))


def _gauss_1d(x_coords, s1):
        return 1./(2.*np.pi*s1**2)*(np.exp(-(x_coords ** 2)/2./s1**2))

def forward_euler_constInp(inputs,
                  w_rec, 
                  start_activity, 
                  delta_t, 
                  time_steps, 
                  tau,
                  nonlinearity,
                  k=None,
                  data_type_np=np.float):
    """ Theano implementation of the forward euler method with input constant in time. 
    For explanation see numpy implementation. """

    proc_inputs = inputs[0,:]
    num_neurons = proc_inputs.shape[0]


    # Calculate the activities of all neurons for all times by scanning over "time". 
    # Use the actual neuronal calculation here.
    # Shapes:
    # tl_neuronal_activity: (num_samples, num_neurons)
    # tl_input_activity: (num_frames, num_samples, num_neurons)
    if start_activity is None:
        t_start_activity = T.zeros((num_neurons), dtype=data_type_np)
    else:
        t_start_activity = start_activity
    
    t_full_activity, _ = \
        theano.scan(fn=lambda tl_neuronal_activity,proc_inputs,delta_t,tau,w_rec: 
                    tl_neuronal_activity + (delta_t / tau) * (- tl_neuronal_activity + nonlinearity(T.dot(tl_neuronal_activity, w_rec.T) + proc_inputs)),
                    outputs_info=[t_start_activity],
                    non_sequences=[proc_inputs,delta_t,tau,w_rec],
                    n_steps=k)

    final_result = t_full_activity[-1,:]

    return final_result


def runge_kutta_explicit(inputs,
                  w_rec, 
                  start_activity, 
                  delta_t, 
                  time_steps, 
                  tau,
                  nonlinearity,
                  k=None,
                  data_type_np=np.float):
    """ Theano implementation of the runge kutta method, 4th order. 
    For explanation see numpy implementation. """

    proc_inputs = inputs[0,:]
    num_neurons = inputs.shape[0] 
    #RK = imrk.Runge_Kutta_Simpson
    #RK = imrk.Runge_Kutta_Fehlberg
    
    def fprime(x, proc_inputs, tau, nonlinearity, w_rec):
        f = (-x + nonlinearity(proc_inputs + T.dot(w_rec, x)))/tau
        return f

    def rk4step(delta_t, x, fprime, proc_inputs, tau, nonlinearity, w_rec):
        k1 = fprime(x, proc_inputs, tau, nonlinearity, w_rec)
        k2 = fprime(x + 0.5*k1*delta_t, proc_inputs, tau, nonlinearity, w_rec)
        k3 = fprime(x + 0.5*k2*delta_t, proc_inputs, tau, nonlinearity, w_rec)
        k4 = fprime(x + k3*delta_t, proc_inputs, tau, nonlinearity, w_rec)
        x_new = x + delta_t*( (1./6.)*k1 + (1./3.)*k2 + (1./3.)*k3 + (1./6.)*k4 )
        return x_new
   
    # Calculate the activities of all neurons for all times by scanning over "time". 
    # Use the actual neuronal calculation here.
    # Shapes:
    # tl_neuronal_activity: (num_samples, num_neurons)
    # tl_input_activity: (num_frames, num_samples, num_neurons)
    if start_activity is None:
        t_start_activity = T.zeros((num_neurons), dtype=data_type_np)
    else:
        t_start_activity = start_activity
        
    t_full_activity, _ = \
    theano.scan(fn=lambda tl_neuronal_activity,proc_inputs,delta_t,tau,w_rec: rk4step(delta_t,\
            tl_neuronal_activity, fprime, proc_inputs, tau, nonlinearity, w_rec),\
            outputs_info=[t_start_activity],\
            non_sequences=[proc_inputs,delta_t,tau,w_rec],\
            n_steps=k)
    
    final_result = t_full_activity[-1,:]
    
    return t_full_activity


def runge_kutta2(inputs, 
                  w_rec, 
                  start_activity, 
                  delta_t, 
                  time_steps, 
                  tau,
                  nonlinearity,
                  k=None,
                  data_type_np=np.float):
    """ Theano implementation of the runge kutta method, 2nd order.  """

    proc_inputs = inputs[0,:]
    num_neurons = proc_inputs.shape[0]
    
    def fprime(x, proc_inputs, tau, nonlinearity, w_rec):
        f = (-x + nonlinearity(proc_inputs + T.dot(w_rec, x)))/tau
        return f

    def rk2step(delta_t, x, fprime, proc_inputs, tau, nonlinearity, w_rec):
        k1 = fprime(x, proc_inputs, tau, nonlinearity, w_rec)
        x_new = x + delta_t*fprime(x+k1*0.5*delta_t, proc_inputs, tau, nonlinearity, w_rec)
        return x_new

    # Calculate the activities of all neurons for all times by scanning over "time". 
    # Use the actual neuronal calculation here.
    # Shapes:
    # tl_neuronal_activity: (num_samples, num_neurons)
    # tl_input_activity: (num_samples, num_neurons)
    if start_activity is None:
        t_start_activity = T.zeros((num_neurons), dtype=data_type_np)
    else:
        t_start_activity = start_activity
     
  
    t_full_activity, _ = \
    theano.scan(fn=lambda tl_neuronal_activity,proc_inputs,delta_t,tau,w_rec: rk2step(delta_t,\
            tl_neuronal_activity, fprime, proc_inputs, tau, nonlinearity, w_rec),\
            outputs_info=[t_start_activity],\
            non_sequences=[proc_inputs,delta_t,tau,w_rec],\
            n_steps=k)

    final_result = t_full_activity[-1,:]

    return final_result


