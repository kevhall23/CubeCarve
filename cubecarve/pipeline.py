import numpy as np
from astropy.convolution import Gaussian2DKernel, Box2DKernel

from .psf.empirical import make_empirical_psf
from .utils.interpolate import scale_image
from .psf.empirical import recenter_psf
from .utils.get_valid_indices_exclude_wavelengths import get_valid_indices_exclude_wavelengths
from .psf.empirical import shift_psf, fix_psf
from .psf.pointsource_fit import fit_2d_gaussians
from .algorithms.create_unresolved_insert import create_unresolved_insert
from .algorithms.dual_channel import dc_run

class CubeCarve:

    def __init__(self, cube, factor, point_sources, xpixelsize, ypixelsize):
        self.cube = cube
        self.factor = factor
        self.point_sources = point_sources
        self.xpixelsize = xpixelsize
        self.ypixelsize = ypixelsize

        self.psf_cube = None
        self.resolved = np.zeros_like(cube)
        self.unresolved = np.zeros_like(cube)
        self.index = None
    def build_psf_cube(self,
          wl_off=np.array([]),
          nlayers=200,
          Xpsf=4, 
          Ypsf=4,
          index=None):
        
        self.wl_off = wl_off
        self.nlayers = nlayers
        if(index == None):
            self.index = np.arange(0,len(self.cube))
            print('Defaulting to full wavelength range.\n')
        else:
            self.index = index

        
        self.psf_cube = np.zeros_like(self.cube)
        counter = 0
        for i in self.index:
            PSF = []
            add_index = 0
            while(len(PSF) == 0):

                PSF = make_empirical_psf(self.cube,
                                         self.point_sources,
                                         i+add_index,
                                         wl_off,
                                         nlayers,
                                         self.factor,
                                         Xpsf,
                                         Ypsf,
                                         self.xpixelsize,
                                         self.ypixelsize)
                

            # Recenter PSF to avoid misalignment issues
            PSF = recenter_psf(PSF)
            PSF /= np.sum(PSF)
            PSF = scale_image(PSF,1/self.factor)
            if(counter == 0):
                y, x = np.shape(PSF)
                self.psf_cube = np.zeros((len(self.cube),y,x))
                counter += 1

            self.psf_cube[i] = PSF
        
        
        print('Empirical PSF Model(s) Built\nReady to run') # feel free to comment this out

    def run(self,n_iters, Rsize,alpha=9e-4,psf_sigma_native=1,sigma=2):
        
        for i in self.index:
            image = self.cube[i]

            # Supersize Input Image Slice:
            image_super = scale_image(image,self.factor)
            image_super_copy = np.copy(image_super)
            # Initialize Point Source Array
            PS_SUPER = np.zeros_like(image_super)
            noise_std = np.mean(image_super[1:10*self.factor,1:10*self.factor])
            wl_index = np.arange(0,len(self.cube))
            image_super = np.clip(image_super,0,None)
            # selected_indices = get_valid_indices_exclude_wavelengths(self.wl_index, center_index=i,total_layers=self.nlayers) 
            selected_indices = get_valid_indices_exclude_wavelengths(self.index,self.nlayers,wl_index,self.wl_off,tol=1e-4)
            full_image = np.nanmedian(self.cube[selected_indices],0)
            full_image = scale_image(full_image,self.factor)
            
            # Initialize Resolved Channel
            
            RESOLVED = np.ones_like(image_super_copy) # to avoid masked pixels in blank image
            RESOLVED /= np.sum(RESOLVED)*2 
            PSF = fix_psf(self.psf_cube[i])
            
            N_STARS = np.count_nonzero(self.point_sources)
            shapey,shapex = np.shape(full_image)

            PS_SUPER = fit_2d_gaussians(full_image, self.point_sources, self.factor)

            # Create Resolution Kernel
            SIZE = Rsize
            RESKERNEL = Gaussian2DKernel(((SIZE)/self.xpixelsize)*self.factor,((SIZE)/self.ypixelsize)*self.factor).array
            # RESKERNEL = match_psf_shape(PSF, RESKERNEL)
            RESKERNEL /= np.sum(RESKERNEL)
            
            weight_std = (image_super/noise_std + 1e-15)**2
            
            Cj = np.zeros_like(image_super_copy)
            
            point_source_coords = np.argwhere(PS_SUPER!=0)
            UNRESOLVED, pix_range = create_unresolved_insert(image_super, point_source_coords, self.factor, psf_sigma_native)
            UNRESOLVED /= np.sum(UNRESOLVED)*2

            # for n in range(n_iters):
            # RESOLVED, UNRESOLVED = dc_run(image_super,self.psf_cube[i],RESOLVED,UNRESOLVED,n_iters,
            #                               alpha,weight_std)
            
            RESOLVED, UNRESOLVED = dc_run(image_super,self.psf_cube[i],self.factor, RESOLVED, UNRESOLVED, RESKERNEL, n_iters,alpha,weight_std,sigma)
            self.resolved[i] = RESOLVED
            self.unresolved[i] = UNRESOLVED
            




