import numpy as np
import scipy.signal as sig 

def safe_array(arr):
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

def myconvolve(f, g) :                                                                                                                                                            
    return sig.fftconvolve(f, g, mode='same')        
