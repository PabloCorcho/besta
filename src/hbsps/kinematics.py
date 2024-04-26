import numpy as np
from scipy.signal import fftconvolve

from hbsps import specBasics


def convolve_ssp(config, los_sigma, los_vel):
    velscale = config["velscale"]
    oversampling = config["oversampling"]
    extra_pixels = config["extra_pixels"]
    ssp_sed = config["ssp_sed"]
    flux = config["flux"]
    # Kinematics
    sigma_pixel = los_sigma / (velscale / oversampling)
    veloffset_pixel = los_vel / (velscale / oversampling)
    x = np.arange(-8 * sigma_pixel, 8 * sigma_pixel) - veloffset_pixel
    losvd_kernel = specBasics.losvd(x, sigma_pixel=sigma_pixel)
    sed = fftconvolve(ssp_sed, np.atleast_2d(losvd_kernel), mode="same", axes=1)
    # Rebin model spectra to observed grid
    sed = (
        sed[:, extra_pixels * oversampling : -(extra_pixels * oversampling + 1)]
        .reshape((sed.shape[0], flux.size, oversampling))
        .mean(axis=2)
    )
    ### Mask pixels at the edges with artifacts produced by the convolution
    mask = np.ones_like(flux, dtype=bool)
    mask[: int(5 * sigma_pixel)] = False
    mask[-int(5 * sigma_pixel) :] = False
    return sed, mask
