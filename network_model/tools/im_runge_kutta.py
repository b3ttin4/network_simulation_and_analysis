'''
Created on 26 Jul 2013

'''
import numpy as np

# Not usable without larger changes.
# class Runge_Kutta_Radau:
#     ''' Runge-Kutta-Radau scheme  (implicit)'''
#     s = 2
#     a = [[5./12, -1./12],
#          [0.75, 0.25]]
#     b = [0.75, 0.25]
#     c = [1./3, 1.]
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)

class Runge_Kutta_Simpson:
	''' explicit, order 4'''
	s = 4
	a = [[0, 0, 0, 0],
		 [0.5, 0, 0, 0],
		 [0, 0.5, 0, 0],
		 [0, 0, 1, 0]]
	b = [1./6, 2./6, 2./6, 1./6]
	c = [0, 0.5, 0.5, 1.]
	a = np.array(a)
	b = np.array(b)
	c = np.array(c)
	
class Runge_Kutta_Fehlberg:
	''' explicit, order 4'''
	s = 6
	a = [[0, 0, 0, 0,0,0],
		 [0.25,0,0, 0, 0, 0],
		 [3./32, 9./32, 0,0,0, 0],
		 [1932./2197, -7200./2197, 7296./2197, 0,0,0],
		 [439./216, -8., 3680./513, -845./4104,0,0],
		 [-8./27, 2., -3544./2565, 1859./4104, -11./40,0]]
	b = [16./135, 0, 6656./12825, 28561./56430, -9./50, 2./55]
	c = [0, 0.25, 3./8, 12./13, 1., 0.5]
	a = np.array(a)
	b = np.array(b)
	c = np.array(c)
