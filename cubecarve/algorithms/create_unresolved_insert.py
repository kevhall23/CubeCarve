import numpy as np
from astropy.convolution import Gaussian2DKernel

from ..utils.arrays import myconvolve

def create_unresolved_insert(image_super, point_source_coords, super_factor, psf_sigma_native=1.25):
    """
    Creates and places inserts at point source locations in a supersampled image.

    Parameters:
    - image_super: 2D numpy array, supersampled image.
    - point_source_coords: list of (x, y) tuples in supersampled coordinates.
    - super_factor: int, supersampling factor (e.g. 4).
    - psf_sigma_native: float, native PSF sigma (default = 1.0 pixel).

    Returns:
    - U: 2D array of unresolved channel with inserts placed.
    """
    # Effective PSF in supersampled image
    psf_sigma_super = psf_sigma_native * super_factor

    # Insert size ~5× PSF sigma → round to nearest odd integer
    insert_radius = int(np.round(3 * psf_sigma_super))  # 2.5 * σ on each side
    if insert_radius % 1 == 0:
        insert_radius = int(insert_radius)
    else:
        insert_radius = int(np.floor(insert_radius))
    insert_size = 2 * insert_radius + 1

    # Build insert
    insert = np.zeros((insert_size, insert_size))
    insert[insert_radius, insert_radius] = .5
    insert = myconvolve(insert, Gaussian2DKernel(1.5))

    # Allocate unresolved channel
    U = np.zeros_like(image_super)

    total_flux = 0
    flux = []
    for (x,y) in point_source_coords:
        flux_i = np.sum(image_super[y-2:y+2,x-2:x+2])
        total_flux += flux_i
        flux.append(flux_i)

    # Add inserts at each point source
        i =0
    for (x, y) in point_source_coords:
        x, y = int(round(x)), int(round(y))
        for dx in range(-insert_radius, insert_radius + 1):
            for dy in range(-insert_radius, insert_radius + 1):
                xi = x + dx
                yi = y + dy
                if 0 <= xi < image_super.shape[0] and 0 <= yi < image_super.shape[1]:
                    U[xi, yi] += insert[insert_radius + dx, insert_radius + dy]
        U[x,y] *= flux[i]
        i = i + 1


    return U, range(-insert_radius, insert_radius + 1)