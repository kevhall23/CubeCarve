import numpy as np
from astropy.convolution import Gaussian2DKernel, Box2DKernel

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
    for i in (range(n_iter)):
        
        MODEL = myconvolve(resolved_init+unresolved_init,psf) 
        
        # Normalize MODEL to have the same flux as image
        MODEL = (MODEL / (np.nansum(MODEL + 1e-15)))
        MODEL = safe_array(MODEL)
        MODEL *= np.sum(image_super) / np.sum(MODEL + 1e-15)
        
        
        Chi = myconvolve(resolved_init, reskernel)
        # Chi = np.clip(Chi, 0, None)
        
        partial_derivative = -1*(np.log((resolved_init + 1e-15) / (Chi + 1e-15)) - 1) / (np.sum(resolved_init) + 1e-15)
        partial_derivative = np.multiply(partial_derivative,weight_std)
        partial_derivative = np.multiply(partial_derivative,((1/np.sum(unresolved_init))))
        resolved_init = safe_array(resolved_init)
        unresolved_init = safe_array(unresolved_init)
        Chi = safe_array(Chi)
        partial_derivative = safe_array(partial_derivative)

        ratio = image_super / (MODEL + 1e-15)
        Cj = myconvolve(ratio, psf[::-1, ::-1]) 
    
        resolved_init = myconvolve(resolved_init,Box2DKernel(sigma))

        resolved_init = resolved_init*(Cj + alpha*partial_derivative ) 
        resolved_init = np.clip(resolved_init, 1e-12, None)

        unresolved_init = (unresolved_init * Cj) #* weight_map
        flux_correction = (np.sum(image_super) / ((np.sum(resolved_init + unresolved_init)) + 1e-15))
        resolved_init *= flux_correction
        unresolved_init *= flux_correction 

    RESOLVED = myconvolve(resolved_init,psf)
    UNRESOLVED = myconvolve(unresolved_init,psf) 

    RESOLVED = scale_image(RESOLVED,1/factor)
    UNRESOLVED = scale_image(UNRESOLVED,1/factor)
    return RESOLVED, UNRESOLVED
     

        