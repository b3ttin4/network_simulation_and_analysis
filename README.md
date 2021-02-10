# Heterogeneous Mexican hat network

Implementation of the network simulation and the analysis tools described in the following publication:

Smith GB, Hein B, Whitney DE, Fitzpatrick D, Kaschube M (2018) [Distributed network interaction and their emergence in developing neocortex.](https://rdcu.be/9PiY) Nat Neurosci. 21(11) 1600-1608.


The code is tested in Python 2 and Python 3. In addition to standard scientific Python libraries (numpy, matplotlib), the code expects:
* [Theano](http://deeplearning.net/software/theano/)
* [H5py](https://www.h5py.org/)
* [OPENCV](https://opencv.org/)

The code is tested with Theano versions 0.8.1, 1.0.2, 1.0.3.


## Network simulation

Network parameters used in the publication can be found in file Default_parameters.
To run a network simulation, run theano/bn_longrange_minimal.py.
The script expects four parameters: mean eccentricity of local connectivity, input modulation, the strength of recurrency and some index number for a filename. 

plots/plot_activity.py generates plots of the activity patterns and examples of correlation patterns.

analysis/run_analysis.py performs the analyses described in the above publication. 
