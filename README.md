# CubeCarve

This repository is home to the Python based dual-channel deconvolution code to extract resolved emission from 3D datacubes.

![cubecarve gif](cubecarve_movie.gif)

## Installation

*CubeCarve* is under active development. To install the latest version from source:

```bash
git clone https://github.com/kevhall23/CubeCarve.git
cd CubeCarve
pip install .
```
## Primary Use

*CubeCarve* is intended to be used on 3D IFU datasets, as the development was originally focused on Keck Cosmic Web Imager (KCWI) observations. Specifically, KCWI observations of z ~ 2 Quasars to image the extended emission within the Circumgalactic Medium (CGM). Through KCWI, the Quasar will act as a bright point source (star) and interfers with the extraction of the diffuse emission that is present within the larger CGM environment. The primary purpose of *CubeCarve* is to extract this *resolved* emission by disentagling it from the bright *unresolved* Quasar. However, this code can be used for a wide variety of science cases. As an example, one could run *CubeCarve* to extract only the point source flux mixed in resolved emission.

Please review the "Example-notebook.ipynb" file to see a full rundown of *CubeCarve* on simulated 3D IFU data. I will provide a basic methodology breakdown below:

## Basic Methodology

_Initialize *CubeCarve*_

- The user must import *CubeCarve* into their python notebook file (ipynb) in the following way:
```python
from cubecarve import CubeCarve
```
- Once the user opens their data, such as a fits file, they must save it to a 3D numpy array in the following standard format: $(\lambda, y, x)$.
- Initialize *CubeCarve* by feeding it the 3D data, as well as the:
    - Scale factor to supersize the input data (useful for low-res KCWI data, but less useful for higher-res images)
    - Initial Guesses on Pointsource locations in the cube (should not be varying across spectral axis)
    - pixel size in arcseconds. As an example, KCWI has rectangular pixels, which must be specified prior to running the code. 
- Here is an example snippet:
```python

# Declare/Guess the Positions of Point Sources in your Image/Cube
# CubeCarve will build Model PSF based on those positions
pointsources = np.zeros_like(cube[0])
pointsources[20,19] = 1 # Star 1 
pointsources[9,12] = 1 # Star 2
# scale factor to supersize the input grid
scale_factor = 6

# cube is our test datacube 3D numpy array
CC = CubeCarve(cube,scale_factor,pointsources,xpixelsize=1,ypixelsize=1)
```

_Build PSF models_

- *CubeCarve* constructs empirical Point-Spread-Function (PSF) models directly from the bright point sourcs present in the data. The user must provide initial guesses on the location of all pointsources prior to running the extraction, as *CubeCarve* will perform a fit using that initial guess as the starting point. The window size of the PSF model must be specified by the user, and the model is built by taking the mean of some number of spectral layers of the datacube. As a general rule of thumb, you want to ensure that your PSF has a high S/N to avoid poor deconvolutions. I recommend inspecting your PSF models prior to running deconvolutions to ensure that they look good.

- We must tell *CubeCarve* which layers to build a PSF model for. For example, if we want to subtract out the Quasar/Point source for every wavelength layer, we do not specify which index values - *CubeCarve* will build a PSF model for each wavelength layer.

- We also want to tell *CubeCarve* which spectral layers to remove from model creation. Specifically, which layers contain extended emission that may interfere with the PSF modeling. The user can opt to keep this empty, and *CubeCarve* will use any layer.

- Example function call:

```python
# restrict layers 180 -> 220 from being used to generate the PSF model
deactivate = np.arange(180,220,1)
# Only build a PSF model for index 203, not the whole cube
index = np.array([203])
N_layers = 200 # Number of cube layers combined via mean to build PSF model -> higher S/N
Xpsf = 4 # X/2 size of PSF window in arcseconds
Ypsf = 4 # Y/2 size of PSF window in arcseconds
CC.build_psf_cube(deactivate,N_layers,Xpsf,Ypsf,index)
```

_Run *Deconvolution*_

- With our empirical PSF models at hand, we can run *CubeCarve* to extract the unresolved and resolved emission. 

- *CubeCarve* requires a resolution kernel to avoid instabilities (see algorithm breakdown in pdf)

- User must specify how many iterations to run before final solutions are produced.

- Here is a code snippet:

```python
N_steps = 800
R_size = 6 # resolution FWHM size: R(FWHM) = R_size x PSF(FWHM)
CC.run(800,R_size)
```

- Once the run finishes, the resolved and unresolved flux components are saved within the CC object, and can be accessed by the user at any time.
