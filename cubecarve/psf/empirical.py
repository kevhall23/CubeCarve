import numpy as np
from astropy.stats import sigma_clip
from scipy.ndimage import shift
from scipy.ndimage import center_of_mass

from .pointsource_fit import fit_point_sources
from ..utils.interpolate import scale_image
from ..utils.get_valid_indices_exclude_wavelengths import get_valid_indices_exclude_wavelengths

def make_empirical_psf(
          CUBE,
          PS,
          index,
          wl_off=np.array([]),
          nlayers=200,
          FACTOR=10,
          Xpsf=4, 
          Ypsf=4,
          xpixelsize=1,
          ypixelsize=1):
        xsize = int(Xpsf/xpixelsize) * FACTOR
        ysize = int(Ypsf/ypixelsize) * FACTOR
        if(xsize % 2 != 0):
            xsize += 1
        if(ysize % 2 != 0):
            ysize += 1
        CUBE = CUBE / np.sum(CUBE, axis=(1, 2), keepdims=True)
        wl_index = np.arange(0,len(CUBE))
        selected_indices = get_valid_indices_exclude_wavelengths(index,nlayers,wl_index,wl_off,tol=1e-4)
        # selected_indices = get_valid_indices_exclude_wavelengths(wl_index, center_index=index,
        #                                                         forbidden_wavelengths=wl_off,total_layers=nlayers)
        full_image = np.nanmean(CUBE[selected_indices],0)
        orig = np.copy(full_image)
        full_image = scale_image(full_image, FACTOR)  # scales image
        PS_SUPER = np.zeros_like(full_image)
        coords = np.argwhere(PS != 0)
        for i in range(len(coords)):
            Y, X = coords[i] * FACTOR
            PS_SUPER[Y, X] = 1

        coordinates = fit_point_sources(full_image, PS_SUPER, FACTOR)  # 
        
        Y, X = full_image.shape
        L = len(coordinates)
        PSF = np.zeros((L, 2 * ysize, 2 * xsize))

        for i in range(L):
            x, y = round(coordinates[i]['x0']), round(coordinates[i]['y0'])

            # Desired crop coordinates
            x1, x2 = x - xsize, x + xsize
            y1, y2 = y - ysize, y + ysize

            # Clip to within bounds
            x1_clip, x2_clip = max(0, x1), min(X, x2)
            y1_clip, y2_clip = max(0, y1), min(Y, y2)

            # Extract the cutout
            cutout = full_image[y1_clip:y2_clip, x1_clip:x2_clip]

            # Offsets for placing cutout into center of output
            dy = y1_clip - y1  # amount missing on top
            dx = x1_clip - x1  # amount missing on left

            psf_patch = np.full((2 * ysize, 2 * xsize), np.nan)
            psf_patch[dy:dy + cutout.shape[0], dx:dx + cutout.shape[1]] = cutout
            psf_patch /= np.nansum(psf_patch)
            PSF[i] = psf_patch
        
        
        # Sigma-clipped average
        clipped_PSF = sigma_clip(PSF, axis=0, sigma=4, maxiters=5)
        PSF = np.nanmean(clipped_PSF, axis=0)
        PSF /= np.nansum(PSF)
        
        return PSF

def shift_psf(psf, dx, dy, shape):
    """
    Shift PSF to subpixel location (dx, dy), cropped to shape.
    """
    shifted = shift(psf, shift=(dy, dx), order=3, mode='nearest')
    cy, cx = shifted.shape[0] // 2, shifted.shape[1] // 2
    sy, sx = shape
    y_min = cy - sy // 2
    x_min = cx - sx // 2
    return shifted[y_min:y_min+sy, x_min:x_min+sx]

def recenter_psf(psf, shape=None):
    """
    Shift a PSF so that its peak (center of mass) is exactly centered in the image.
    
    Parameters:
        psf (2D array): The input PSF to be shifted.
        shape (tuple or None): If provided, crop/pad to this shape after shifting.
    
    Returns:
        psf_centered (2D array): PSF shifted so the peak is at the center.
    """
    

    # Find the current center of mass (subpixel)
    cy0, cx0 = center_of_mass(psf)

    # Desired center location
    ny, nx = psf.shape
    cy_desired = (ny - 1) / 2.0
    cx_desired = (nx - 1) / 2.0

    # Compute shifts required
    dy = cy_desired - cy0
    dx = cx_desired - cx0

    # Apply subpixel shift using spline interpolation (order=3)
    psf_centered = shift(psf, shift=(dy, dx), order=3, mode='nearest', prefilter=True)

    if shape is not None:
        sy, sx = shape
        cy, cx = psf_centered.shape[0] // 2, psf_centered.shape[1] // 2
        y_min = cy - sy // 2
        x_min = cx - sx // 2
        psf_centered = psf_centered[y_min:y_min+sy, x_min:x_min+sx]

    return psf_centered

def fix_psf(psf):
    yshape, xshape = np.shape(psf)
    y,x = np.unravel_index(psf.argmax(), psf.shape)
    yshift, xshift = (yshape/2) - y, (xshape/2) - x
    psf = shift(psf, (-1*yshift, -1*xshift),mode='nearest')
    psf /= np.sum(psf)
    return psf