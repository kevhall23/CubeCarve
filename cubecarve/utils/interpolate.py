import numpy as np
from scipy.interpolate import RectBivariateSpline

def scale_image(image, scale_factor):
    """
    Scales a 2D image using bicubic interpolation with proper alignment correction
    to preserve centroid positions.

    Parameters:
        image : 2D numpy array
            The input image to be scaled.
        scale_factor : float
            Scaling factor for both axes.

    Returns:
        scaled_image : 2D numpy array
            Rescaled image, with preserved coordinate alignment.
    """
    ny, nx = image.shape
    y = np.arange(ny)
    x = np.arange(nx)

    spline = RectBivariateSpline(y, x, image, kx=3, ky=3)

    # New grid WITH HALF-PIXEL CORRECTION
    new_ny = int(round(ny * scale_factor))
    new_nx = int(round(nx * scale_factor))
    y_new = (np.arange(new_ny) + 0.5) / scale_factor - 0.5
    x_new = (np.arange(new_nx) + 0.5) / scale_factor - 0.5

    scaled_image = spline(y_new, x_new)

    return scaled_image/(scale_factor**2)