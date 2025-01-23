"""This module contains the tools for modelling kinematic effects on spectra."""
import numpy as np
import re
from scipy.signal import fftconvolve
from astropy.modeling import Fittable1DModel
from astropy.modeling.models import Gaussian1D, Hermite1D
from astropy.convolution.kernels import Model1DKernel

from astropy import units as u
from astropy.convolution import convolve, convolve_fft

from besta import spectrum
from besta import config


class GaussHermite(Fittable1DModel):
    """Gauss-Hermite model."""

    _param_names = ()

    def __init__(self, order, *args, **kwargs):
        self._order = int(order)
        if self._order < 3:
            self._order = 0

        self._gaussian = Gaussian1D()
        # Hermite series
        if self._order:
            self._hermite = Hermite1D(self._order)
        else:
            self._hermite = None

        self._param_names = self._generate_coeff_names()
        super(GaussHermite, self).__init__(*args, **kwargs)

    def _generate_coeff_names(self):
        names = list(self._gaussian.param_names)  # Gaussian parameters
        names += ["h{}".format(i) for i in range(3, self._order + 1)]  # Hermite coeffs

    def _hi_order(self, name):
        # One could store the compiled regex, but it will crash the deepcopy:
        # "cannot deepcopy this pattern object"

        match = re.match("h(?P<order>\d+)", name)  # h3, h4, etc.
        order = int(match.groupdict()["order"]) if match else 0

        return order

    def _generate_coeff_names(self):
        names = list(self._gaussian.param_names)  # Gaussian parameters
        names += ["h{}".format(i) for i in range(3, self._order + 1)]  # Hermite coeffs

        return tuple(names)

    def __getattr__(self, attr):
        if attr[0] == "_":
            super(GaussHermite, self).__getattr__(attr)
        elif attr in self._gaussian.param_names:
            return self._gaussian.__getattribute__(attr)
        elif self._order and self._hi_order(attr) >= 3:
            return self._hermite.__getattribute__(attr.replace("h", "c"))
        else:
            super(GaussHermite, self).__getattr__(attr)

    def __setattr__(self, attr, value):
        if attr[0] == "_":
            super(GaussHermite, self).__setattr__(attr, value)
        elif attr in self._gaussian.param_names:
            self._gaussian.__setattr__(attr, value)
        elif self._order and self._hi_order(attr) >= 3:
            self._hermite.__setattr__(attr.replace("h", "c"), value)
        else:
            super(GaussHermite, self).__setattr__(attr, value)

    @property
    def param_names(self):
        return self._param_names

    def evaluate(self, x, *params):
        a, m, s = params[:3]  # amplitude, mean, stddev
        f = self._gaussian.evaluate(x, a, m, s)
        if self._order:
            f *= 1 + self._hermite.evaluate((x - m) / s, 0, 0, 0, *params[3:])

        return f


# TODO : remove and homogeneize
def losvd(vel_pixel, sigma_pixel, h3=0, h4=0):
    y = vel_pixel / sigma_pixel
    g = (
        np.exp(-(y**2) / 2)
        / sigma_pixel
        / np.sqrt(2 * np.pi)
        * (
            1
            + h3 * (y * (2 * y**2 - 3) / np.sqrt(3))  # H3
            + h4 * ((4 * (y**2 - 3) * y**2 + 3) / np.sqrt(24))  # H4
        )
    )
    return g


def get_losvd_kernel(kernel_model, x_size):
    """Create a ``Model1DKernel`` from an input ``Model``.

    Parameters
    ----------
    kernel_model : :class:`astropy.models.FittableModel`
        Model used to build the kernel.
    x_size : int
        Kernel size

    Returns
    -------
    kernel : :class:`Model1DKernel`
        Kernel model
    """
    return Model1DKernel(kernel_model, x_size=x_size, mode="oversample", factor=10)


def convolve_spectra_with_kernel(spectra, kernel):
    """Convolve an input spectra with a given kernel.

    Parameters
    ----------
    kernel_model : :class:`Model1DKernel`
        Kernel model
    spectra : np.ndarray
        Target spectra to convolve with the kernel

    Returns
    -------
    convolved_spectra : np.ndarray
        Spectra convolved with the input kernel.
    """
    return convolve(
        spectra, kernel, boundary="fill", fill_value=0.0, normalize_kernel=True
    )

def convolve_ssp_with_lsf(ssp, lsf_sigma_pixels):
    """Convolve a given SSP model with an LSF.
    
    Parameters
    ----------
    ssp : pst.SSP.SSPBase
    lsf_sigma_pixels : float, np.ndarray
        Gaussian standard deviation of the LSF.
    """
    # Account for LSF variability
    if lsf_sigma_pixels.size == ssp.wavelength.size:
        ssp.L_lambda = np.array([
            fftconvolve(
                ssp.L_lambda.value,
                losvd(np.arange(-config.kinematics["lsf_sigma_truncation"] * sigma,
                                config.kinematics["lsf_sigma_truncation"] * sigma),
                                sigma_pixel=sigma)[np.newaxis, np.newaxis],
                                mode="same", axes=2)[:, :, ith]
            for ith, sigma in enumerate(lsf_sigma_pixels)]) * ssp.L_lambda.unit
    # Constant LSF
    elif np.atleast_1d(lsf_sigma_pixels).size == 1:
        ssp.L_lambda = fftconvolve(
                ssp.L_lambda.value,
                losvd(np.arange(-config.kinematics["lsf_sigma_truncation"] * lsf_sigma_pixels,
                                config.kinematics["lsf_sigma_truncation"] * lsf_sigma_pixels),
                                sigma_pixel=lsf_sigma_pixels)[np.newaxis, np.newaxis],
                                mode="same", axes=2) * ssp.L_lambda.unit
    else:
        raise ArithmeticError("Dimensions of SSP and LSF do not match")

def convolve_ssp(config, los_sigma, los_vel, los_h3=0.0, los_h4=0.0):
    velscale = config["velscale"]
    extra_pixels = config["extra_pixels"]
    ssp_sed = config["ssp_sed"]
    flux = config["flux"]
    # Kinematics
    sigma_pixel = los_sigma / velscale
    veloffset_pixel = los_vel / velscale
    x = (
        np.arange(
            -config.kinematics["lsf_sigma_truncation"] * sigma_pixel,
            config.kinematics["lsf_sigma_truncation"] * sigma_pixel,
        )
        - veloffset_pixel
    )
    losvd_kernel = losvd(x, sigma_pixel=sigma_pixel, h3=los_h3, h4=los_h4)
    sed = fftconvolve(ssp_sed, np.atleast_2d(losvd_kernel), mode="same", axes=1)
    # Rebin model spectra to observed grid
    sed = sed[:, extra_pixels:-extra_pixels]
    ### Mask pixels at the edges with artifacts produced by the convolution
    mask = np.ones_like(flux, dtype=bool)
    mask[: int(config.kinematics["lsf_sigma_truncation"] * sigma_pixel)] = False
    mask[-int(config.kinematics["lsf_sigma_truncation"] * sigma_pixel) :] = False
    return sed, mask


def convolve_ssp_model(config, los_sigma, los_vel, h3=0.0, h4=0.0):
    velscale = config["velscale"]
    extra_pixels = int(config["extra_pixels"])
    ssp = config["ssp_model"]
    wl = config["wavelength"]
    # Kinematics
    sigma_pixel = los_sigma / velscale
    veloffset_pixel = los_vel / velscale
    x = (
        np.arange(
            -config.kinematics["lsf_sigma_truncation"] * sigma_pixel,
            config.kinematics["lsf_sigma_truncation"] * sigma_pixel,
        )
        - veloffset_pixel
    )
    losvd_kernel = spectrum.losvd(x, sigma_pixel=sigma_pixel, h3=h3, h4=h4)
    ssp.L_lambda = (
        fftconvolve(
            ssp.L_lambda.value,
            losvd_kernel[np.newaxis, np.newaxis],
            mode="same",
            axes=2,
        )
        * ssp.L_lambda.unit
    )
    # Rebin model spectra to observed grid
    pixels = slice(extra_pixels, -extra_pixels)
    new_sed = ssp.L_lambda[:, :, pixels]
    ssp.L_lambda = new_sed
    if not isinstance(wl, u.Quantity):
        ssp.wavelength = wl * ssp.wavelength.unit
    else:
        ssp.wavelength = wl
    ### Mask pixels at the edges with artifacts produced by the convolution
    mask = np.ones(wl.size, dtype=bool)
    mask[: int(config.kinematics["lsf_sigma_truncation"] * sigma_pixel)] = False
    mask[-int(config.kinematics["lsf_sigma_truncation"] * sigma_pixel) :] = False
    return ssp, mask
