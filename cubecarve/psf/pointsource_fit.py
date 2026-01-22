import numpy as np
from scipy.optimize import curve_fit

def twoD_gaussian(coords, x0, y0, sigma_x, sigma_y, amplitude, offset):
    x, y = coords
    exponent = ((x - x0)**2 / (2 * sigma_x**2)) + ((y - y0)**2 / (2 * sigma_y**2))
    return amplitude * np.exp(-exponent) + offset
def fit_point_sources(image, initial_mask,factor, box_size=5):
    """
    Fits 2D asymmetric Gaussian to all point sources in image.

    Parameters:
    -----------
    image : 2D np.ndarray
        The input noisy image.
    initial_mask : 2D np.ndarray
        Same shape as `image`, with 1 at initial guess locations for point sources.
    box_size : int
        Size of box to extract around each source (should be odd).

    Returns:
    --------
    fits : list of dicts
        Each dict contains fitted parameters: x0, y0, sigma_x, sigma_y, amplitude, offset.
    """
    # assert image.shape == initial_mask.shape
    half_box = (box_size*factor) // 2
    y_idxs, x_idxs = np.argwhere(initial_mask == 1).T #*factor
    fits = []
    median_vals = []
    # print(np.shape(image))
    for x0_guess, y0_guess in zip(x_idxs, y_idxs):
        # Define cutout bounds with clipping at image edges
        x_min = max(x0_guess - half_box, 0)
        x_max = min(x0_guess + half_box + 1, image.shape[1])
        y_min = max(y0_guess - half_box, 0)
        y_max = min(y0_guess + half_box + 1, image.shape[0])
        cutout = image[y_min:y_max, x_min:x_max]
        
        # Create coordinate grid for actual pixel positions
        y_grid, x_grid = np.mgrid[y_min:y_max, x_min:x_max]

        # Flatten for fitting
        x_flat = x_grid.ravel()
        y_flat = y_grid.ravel()
        data_flat = cutout.ravel()

        # Initial guess for fitting parameters
        initial_params = [
            x0_guess, y0_guess,   # x0, y0 (absolute coords)
            1, 5,             # sigma_x, sigma_y
            cutout.max() - np.median(cutout),  # amplitude
            np.median(cutout)     # offset
        ]

        try:
            popt, _ = curve_fit(
                twoD_gaussian,
                (x_flat, y_flat),
                data_flat,
                p0=initial_params,
                maxfev=10000
            )
            fit_dict = {
                "x0": popt[0],
                "y0": popt[1],
                "sigma_x": popt[2],
                "sigma_y": popt[3],
                "amplitude": popt[4],
                "offset": popt[5]
            }

            Y,X = np.argwhere(cutout==np.max(cutout))[0]
            x,y = fit_dict['x0'], fit_dict['y0']
            dist = np.sqrt( (x0_guess-x)**2 + (y0_guess-y)**2 )
            if(dist > 100):
                continue
            else:

                fits.append(fit_dict)
        except RuntimeError:
            print(f"Fit failed at ({x0_guess}, {y0_guess})")
        ################
    
    return fits


def two_d_gaussian(xy, amp, x0, y0, sigma_x, sigma_y):
    """2D Gaussian model."""
    x, y = xy
    exp = np.exp(-(((x - x0) ** 2) / (2 * sigma_x**2) + ((y - y0) ** 2) / (2 * sigma_y**2)))
    return amp * exp

def fit_2d_gaussians(image_sup, guess_positions, factor):
    """
    Fit 2D Gaussians to sources in a supersampled image.
    
    Parameters:
        image_sup (2D array): Supersampled image.
        guess_positions (array): Nx2 array of (y, x) positions in native pixel coordinates.
        factor (int): Supersampling factor.

    Returns:
        result_image (2D array): Supersampled image with 1s at refined source locations.
    """
    result_image = np.zeros_like(image_sup)
    size = 5 * factor  # Size of cutout box around each source (adjustable)
    guess_positions = np.argwhere(guess_positions!=0)

    Y, X = image_sup.shape
    for y, x in guess_positions:
        y_sup = int(round(y * factor))
        x_sup = int(round(x * factor))

        # Define cutout bounds
        y1 = max(y_sup - size, 0)
        y2 = min(y_sup + size + 1, Y)
        x1 = max(x_sup - size, 0)
        x2 = min(x_sup + size + 1, X)

        cutout = image_sup[y1:y2, x1:x2]

        # Meshgrid for fitting
        yy, xx = np.mgrid[0:cutout.shape[0], 0:cutout.shape[1]]
        initial_guess = (cutout.max(), cutout.shape[1] // 2, cutout.shape[0] // 2, factor, factor)

        try:
            popt, _ = curve_fit(two_d_gaussian, (xx.ravel(), yy.ravel()), cutout.ravel(), p0=initial_guess)
            _, x0_fit, y0_fit, _, _ = popt

            # Convert local fit coords back to global supersampled coords
            x_final = x1 + x0_fit
            y_final = y1 + y0_fit
            # print(int(round(y_final)), int(round(x_final)))
            if 0 <= int(round(y_final)) < Y and 0 <= int(round(x_final)) < X:
                result_image[int(round(y_final)), int(round(x_final))] = 1.0
        except RuntimeError:
            # If fitting fails, skip
            continue

    return result_image