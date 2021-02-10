import numpy as np


# Nonlinearity functions (Numpy implementation)
nl_linear = lambda x: x
nl_tanh = lambda x: np.tanh(x)
nl_sigmoid = lambda x: 1./(1+np.exp(-x)) 
nl_rect = lambda x: np.clip(x, 0, np.inf)
#nl_rect = lambda x: np.clip(x, -np.inf, np.inf)
nl_shallow_rect = lambda x: np.clip(0.1*x, 0, np.inf)
nl_clip = lambda x: np.clip(x, 0, 1)
nl_softplus = lambda x: np.log(1. + np.exp(x)) #
#'''
# Nonlinearity functions (Theano implementation)
import numpy, theano
import numpy.distutils
import numpy.distutils.__config__
import theano.tensor as T
nl_linear_t = lambda x: x
nl_tanh_t = lambda x: T.tanh(x)            
nl_sigmoid_t = lambda x: T.nnet.sigmoid(x)      
nl_fermi_t = lambda x: T.nnet.sigmoid(x*50)
nl_clip_t = lambda x: T.clip(x, 0., 1.)
nl_rect_t = lambda x: T.maximum(x, 0.)
nl_rect_squared_t = lambda x: T.maximum(x**2, 0.)
nl_shallow_rect_t = lambda x:  T.maximum(0.1*x, 0.)
#'''
def convert_input_const_to_time(inp, num_frames):
    if inp.shape[0] != 1:
        raise Exception("First axis of inp has to be 1-dim.")
    if inp.shape[1] != 1:
        inp = inp[:, 0:1, :]
        print('WARNING (bn_tools): Input has more than one frame. Only first frame will be broadcast.')
        
    inp = np.tile(inp, (1, num_frames, 1))
    return inp
 
def check_nonlinearities():
    import matplotlib.pyplot as plt
    x_np=np.arange(-5,5,0.1).astype('float32')
    x=theano.shared(x_np) 
#    for fkt in [nl_linear_t,nl_rect_t,nl_clip_t,nl_sigmoid_t, nl_tanh_t]:
    for fkt in [nl_clip_t,nl_sigmoid_t]:

        y=  fkt(x)
        tf = theano.function([],y)
        plt.plot(x_np, tf())
    plt.show()
    
if __name__=='__main__':
    check_nonlinearities()
