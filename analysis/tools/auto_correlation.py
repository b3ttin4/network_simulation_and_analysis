import numpy as np

def get_autocorr(opm,max_lag,method='wiener_khinchin'):
	''' calculate auto correlation function using wiener-khinchin theorem'''
	if opm.ndim==2:
		opm = opm[None,:,:]
	nframes,hs,ws = opm.shape
	
	if method=='wiener_khinchin':
		opm = (opm - np.nanmean(opm.reshape(nframes,hs*ws),axis=1)[:,None,None])/np.nanstd(opm.reshape(nframes,hs*ws),axis=1)[:,None,None]
		n = 3
		fft_spectrum = abs(np.fft.fft2(opm, s=(n*hs,n*ws),axes=(-2,-1)))**2
		autocorr = np.fft.ifft2( fft_spectrum, axes=(-2,-1) )
		nframes,h_padded,w_padded = autocorr.shape
		autocorr = np.fft.fftshift(autocorr,axes=(-2,-1))[:, h_padded//2-max_lag:h_padded//2+max_lag+1,w_padded//2-max_lag:w_padded//2+max_lag+1]
		norm = (hs-abs(np.arange(-max_lag,max_lag+1)))[:,None]*(ws-abs(np.arange(-max_lag,max_lag+1)))[None,:]
		autocorr = autocorr/norm			# only biased through sigma and mean
	return np.real(np.squeeze(autocorr))
