import numpy as np
from astropy.convolution import Gaussian2DKernel, Box2DKernel
import scipy.signal as sig

from ..utils.arrays import myconvolve
from ..utils.arrays import safe_array
from ..utils.interpolate import scale_image



def dc_run(
    image_super,
    psf,
    factor,
    resolved_init,
    unresolved_init,
    reskernel,
    n_iter,
    alpha,
    weight_std,
    sigma=2
):
    psf = scale_image(psf,factor)
    psf = np.abs(psf)
    psf /= np.sum(psf)
    sigma = Box2DKernel(sigma).array
    sigma /= np.sum(sigma)
    for i in (range(n_iter)):
        
        MODEL = sig.fftconvolve(resolved_init+unresolved_init,psf, mode='same')    
        # MODEL = myconvolve(resolved_init+unresolved_init,psf) 
        
        # Normalize MODEL to have the same flux as image
        MODEL = (MODEL / (np.nansum(MODEL + 1e-15)))
        MODEL = safe_array(MODEL)
        MODEL *= np.sum(image_super) / np.sum(MODEL + 1e-15)
        
        
        # Chi = myconvolve(resolved_init, reskernel)
        Chi = sig.fftconvolve(resolved_init, reskernel, mode='same')
        # Chi = np.clip(Chi, 0, None)
        
        partial_derivative = -1*(np.log((resolved_init + 1e-15) / (Chi + 1e-15)) - 1) / (np.sum(resolved_init) + 1e-15)
        partial_derivative = np.multiply(partial_derivative,weight_std)
        partial_derivative = np.multiply(partial_derivative,((1/np.sum(unresolved_init))))
        resolved_init = safe_array(resolved_init)
        unresolved_init = safe_array(unresolved_init)
        Chi = safe_array(Chi)
        partial_derivative = safe_array(partial_derivative)

        ratio = image_super / (MODEL + 1e-15)
        # Cj = myconvolve(ratio, psf[::-1, ::-1]) 
        Cj = sig.fftconvolve(ratio, psf[::-1, ::-1], mode='same')    

        
        resolved_init = sig.fftconvolve(resolved_init,sigma, mode='same')

        resolved_init *= (Cj + alpha*partial_derivative ) 
        resolved_init = np.clip(resolved_init, 1e-12, None)

        unresolved_init *= Cj
        flux_correction = (np.sum(image_super) / ((np.sum(resolved_init + unresolved_init)) + 1e-15))
        resolved_init *= flux_correction
        unresolved_init *= flux_correction 

    RESOLVED = sig.fftconvolve(resolved_init,psf, mode='same')
    UNRESOLVED = sig.fftconvolve(unresolved_init,psf, mode='same')

    RESOLVED = scale_image(RESOLVED,1/factor)
    UNRESOLVED = scale_image(UNRESOLVED,1/factor)
    return RESOLVED, UNRESOLVED

