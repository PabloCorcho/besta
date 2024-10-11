import numpy as np
from scipy.signal import fftconvolve
from astropy import units as u
from hbsps import specBasics


def convolve_ssp(config, los_sigma, los_vel, los_h3=0., los_h4=0.):
    velscale = config["velscale"]
    oversampling = config["oversampling"]
    extra_pixels = config["extra_pixels"]
    ssp_sed = config["ssp_sed"]
    flux = config["flux"]
    # Kinematics
    sigma_pixel = los_sigma / (velscale / oversampling)
    veloffset_pixel = los_vel / (velscale / oversampling)
    x = np.arange(-5 * sigma_pixel, 5 * sigma_pixel) - veloffset_pixel
    losvd_kernel = specBasics.losvd(x, sigma_pixel=sigma_pixel,
                                    h3=los_h3, h4=los_h4)
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

def convolve_ssp_model(config, los_sigma, los_vel, h3=0.0, h4=0.0):
    velscale = config["velscale"]
    oversampling = int(config["oversampling"])
    extra_pixels = int(config["extra_pixels"])
    ssp = config["ssp_model"]
    wl = config['wavelength']
    # Kinematics
    sigma_pixel = los_sigma / (velscale / oversampling)
    veloffset_pixel = los_vel / (velscale / oversampling)
    x = np.arange(-8 * sigma_pixel, 8 * sigma_pixel) - veloffset_pixel
    losvd_kernel = specBasics.losvd(x, sigma_pixel=sigma_pixel, h3=h3, h4=h4)
    ssp.L_lambda = fftconvolve(ssp.L_lambda.value,
                      losvd_kernel[np.newaxis, np.newaxis], mode="same", axes=2
                      ) * ssp.L_lambda.unit

    # Rebin model spectra to observed grid
    pixels = slice(extra_pixels * oversampling,  -(extra_pixels * oversampling + 1))
    print(pixels)
    new_sed = ssp.L_lambda[:, :, pixels].reshape((ssp.L_lambda.shape[0], ssp.L_lambda.shape[1],
                                                  wl.size, oversampling)).mean(axis=-1)
    ssp.L_lambda = new_sed
    if not isinstance(wl, u.Quantity):
        ssp.wavelength = wl * ssp.wavelength.unit
    else:
        ssp.wavelength = wl
    ### Mask pixels at the edges with artifacts produced by the convolution
    mask = np.ones(wl.size, dtype=bool)
    mask[: int(5 * sigma_pixel)] = False
    mask[-int(5 * sigma_pixel) :] = False
    return ssp, mask


def convolve_spectra(spectra, los_vel, los_sigma):
    x = np.arange(-8 * los_sigma, 8 * los_sigma) - los_vel
    losvd_kernel = specBasics.losvd(x, sigma_pixel=los_sigma)

    conv_spectra = fftconvolve(spectra, losvd_kernel, mode="same")
    return conv_spectra