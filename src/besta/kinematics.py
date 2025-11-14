"""This module contains the tools for modelling kinematic effects on spectra."""
import numpy as np
import re
from scipy.signal import fftconvolve
from scipy.special import erf
from scipy import sparse

from astropy.modeling import Fittable1DModel
from astropy.modeling.models import Gaussian1D, Hermite1D
from astropy.convolution.kernels import Model1DKernel

from astropy import units as u
from astropy.convolution import convolve, convolve_fft

from besta import spectrum
from besta import config as CONFIG


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
    return Model1DKernel(kernel_model, x_size=x_size, mode="center")


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
                losvd(np.arange(-CONFIG.kinematics["lsf_sigma_truncation"] * sigma,
                                CONFIG.kinematics["lsf_sigma_truncation"] * sigma),
                                sigma_pixel=sigma)[np.newaxis, np.newaxis],
                                mode="same", axes=2)[:, :, ith]
            for ith, sigma in enumerate(lsf_sigma_pixels)]) * ssp.L_lambda.unit
    # Constant LSF
    elif np.atleast_1d(lsf_sigma_pixels).size == 1:
        ssp.L_lambda = fftconvolve(
                ssp.L_lambda.value,
                losvd(np.arange(-CONFIG.kinematics["lsf_sigma_truncation"] * lsf_sigma_pixels,
                                CONFIG.kinematics["lsf_sigma_truncation"] * lsf_sigma_pixels),
                                sigma_pixel=lsf_sigma_pixels)[np.newaxis, np.newaxis],
                                mode="same", axes=2) * ssp.L_lambda.unit
    else:
        raise ArithmeticError("Dimensions of SSP and LSF do not match")

def convolve_ssp(module_config, los_sigma, los_vel, los_h3=0.0, los_h4=0.0):
    velscale = module_config["velscale"]
    extra_pixels = module_config["extra_pixels"]
    ssp_sed = module_config["ssp_sed"]
    flux = module_config["flux"]
    # Kinematics
    sigma_pixel = los_sigma / velscale
    veloffset_pixel = los_vel / velscale
    x = (
        np.arange(
            -CONFIG.kinematics["lsf_sigma_truncation"] * sigma_pixel,
            CONFIG.kinematics["lsf_sigma_truncation"] * sigma_pixel,
        )
        - veloffset_pixel
    )
    losvd_kernel = losvd(x, sigma_pixel=sigma_pixel, h3=los_h3, h4=los_h4)
    sed = fftconvolve(ssp_sed, np.atleast_2d(losvd_kernel), mode="same", axes=1)
    # Rebin model spectra to observed grid
    sed = sed[:, extra_pixels:-extra_pixels]
    ### Mask pixels at the edges with artifacts produced by the convolution
    mask = np.ones_like(flux, dtype=bool)
    mask[: int(CONFIG.kinematics["lsf_sigma_truncation"] * sigma_pixel)] = False
    mask[-int(CONFIG.kinematics["lsf_sigma_truncation"] * sigma_pixel) :] = False
    return sed, mask


def convolve_ssp_model(module_config, los_sigma, los_vel, h3=0.0, h4=0.0):
    velscale = module_config["velscale"]
    extra_pixels = int(module_config["extra_pixels"])
    ssp = module_config["ssp_model"]
    wl = module_config["wavelength"]
    # Kinematics
    sigma_pixel = los_sigma / velscale
    veloffset_pixel = los_vel / velscale
    x = (
        np.arange(
            -CONFIG.kinematics["lsf_sigma_truncation"] * sigma_pixel,
             CONFIG.kinematics["lsf_sigma_truncation"] * sigma_pixel,
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
    mask[: int(CONFIG.kinematics["lsf_sigma_truncation"] * sigma_pixel)] = False
    mask[-int(CONFIG.kinematics["lsf_sigma_truncation"] * sigma_pixel) :] = False
    return ssp, mask

def normal_cdf(x, mu=0.0, sigma=1.0):
    """Normal cumulative density function."""
    return (1.0 + erf((x - mu) / sigma / np.sqrt(2.0))) / 2.0

# def convolve_variable_gaussian_kernel(spectra, sigma_pixel,
#                                       kappa_sigma_thresh=3.0):
#     """Convolve an input spectra with a Gaussian kernel of varying width.
    
#     Parameters
#     ----------
#     spectra : :class:`np.ndarray` or :class:`u.Quantity`
#         N-dimensional spectra. The last dimension must correspond to the
#         wavelength axis.
#     sigma_pixel : :class:`np.ndarray`
#         Value of the standard deviation of the Gaussian LSF for each spectral
#         resolution element, expressed in pixel units.
#     kappa_sigma_thresh : float, optional
#         Threshold in units of the Gaussian sigma. If a **single pixel** contains
#         at least ``kappa_sigma_thresh`` sigmas of the LSF, that pixel is left
#         untouched (i.e. the convolution kernel is clamped to a delta function
#         at that pixel). Default is ``3.0``.

#     Returns
#     -------
#     convolved_spectra : :class:`np.ndarray` or :class:`u.Quantity`
#         A convolved version of ``spectra``.
#     """
#     # Convert sigma_pixel to a plain ndarray (it may be a Quantity or list)
#     sigma_pixel = np.asarray(sigma_pixel, dtype=float)
#     n_pix = sigma_pixel.size

#     # Identify pixels where the LSF is effectively contained within a single pixel:
#     # 0.5 pixel half-width contains >= kappa_sigma_thresh sigmas.
#     # 0.5 / sigma >= kappa  ->  sigma <= 0.5 / kappa
#     small_lsf_mask = (0.5 / sigma_pixel) >= kappa_sigma_thresh
#     # Guard against sigma == 0 as well (also treated as delta kernel)
#     small_lsf_mask |= (sigma_pixel <= 0.0)

#     # Use a "safe" sigma for computing the CDF to avoid division issues;
#     # rows that will be clamped later can use any finite sigma.
#     safe_sigma = sigma_pixel.copy()
#     safe_sigma[small_lsf_mask] = 1.0

#     # mean_pixel has shape (n_pix, n_pix + 1) and represents bin edges
#     mean_pixel = (np.arange(-0.5, n_pix + 0.5, 1)[np.newaxis, :]
#                   - np.arange(0, n_pix, 1)[:, np.newaxis])

#     cmf = normal_cdf(mean_pixel / safe_sigma[:, np.newaxis])
#     weights = cmf[:, 1:] - cmf[:, :-1]      # (n_pix, n_pix)

#     # Normalise each row
#     weights /= np.sum(weights, axis=1)[:, np.newaxis]

#     # Clamp rows where the LSF is narrower than a pixel:
#     # replace the whole row by a delta kernel.
#     if np.any(small_lsf_mask):
#         weights[small_lsf_mask, :] = 0.0
#         idx = np.nonzero(small_lsf_mask)[0]
#         # Put the delta on the diagonal: output[i] = input[i]
#         weights[idx, idx] = 1.0

#     # Broadcast to match the input spectra shape (last axis is spectral axis)
#     extra_dim = spectra.ndim - 1
#     if extra_dim > 0:
#         axis = [dim for dim in np.arange(0, extra_dim, 1)]
#         weights = np.expand_dims(weights, axis=axis)

#     # Perform the (variable) convolution along the last axis
#     return np.sum(np.expand_dims(spectra, axis=-1) * weights, axis=-1)

def convolve_variable_gaussian_kernel(
    spectra,
    sigma_pixel,
    kappa_sigma_thresh=3.0,
    kappa_trunc=5.0,
):
    """
    Convolve an input spectra with a Gaussian kernel of varying width,
    using a sparse banded convolution matrix.

    Parameters
    ----------
    spectra : np.ndarray or astropy.units.Quantity
        N-dimensional spectra. The last dimension must correspond to the
        wavelength axis.
    sigma_pixel : np.ndarray
        Standard deviation of the Gaussian LSF for each spectral resolution
        element, expressed in pixel units (length = n_pix).
    kappa_sigma_thresh : float, optional
        Threshold in units of the Gaussian sigma. If a single pixel contains
        at least ``kappa_sigma_thresh`` sigmas of the LSF, that pixel is left
        untouched (i.e. the convolution kernel is clamped to a delta function
        at that pixel). Default is ``3.0``.
    kappa_trunc : float, optional
        Truncation radius of the Gaussian in units of sigma. Only pixels within
        ``±kappa_trunc * sigma`` of the central pixel contribute to the kernel.
        Default is ``5.0``.

    Returns
    -------
    convolved_spectra : np.ndarray or astropy.units.Quantity
        A convolved version of ``spectra``.
    """
    # Handle astropy units if present
    has_unit = hasattr(spectra, "unit")
    if has_unit:
        unit = spectra.unit
        spec_vals = np.asarray(spectra.value, dtype=float)
    else:
        unit = None
        spec_vals = np.asarray(spectra, dtype=float)

    sigma_pixel = np.asarray(sigma_pixel, dtype=float)
    n_pix = sigma_pixel.size

    if spec_vals.shape[-1] != n_pix:
        raise ValueError(
            "Last axis of 'spectra' must have the same length as 'sigma_pixel'."
        )

    # Build sparse convolution matrix W of shape (n_pix, n_pix)
    rows = []
    cols = []
    data = []

    for i, sigma in enumerate(sigma_pixel):
        # Clamp to delta kernel if LSF is effectively within one pixel
        # 0.5 / sigma >= kappa  ->  sigma <= 0.5 / kappa
        if sigma <= 0.0 or (0.5 / sigma) >= kappa_sigma_thresh:
            rows.append(i)
            cols.append(i)
            data.append(1.0)
            continue

        # Local window around pixel i
        half_width = int(np.ceil(kappa_trunc * sigma))
        j_min = max(0, i - half_width)
        j_max = min(n_pix - 1, i + half_width)

        # Pixel indices contributing to output pixel i
        j = np.arange(j_min, j_max + 1)

        # Pixel edges in "pixel number" coordinates, relative to the LSF centre at i
        # We define edges from (j_min - 0.5) to (j_max + 0.5), step 1
        edges = np.arange(j_min, j_max + 2) - i - 0.5  # length len(j) + 1

        # CDF at edges, sigma is already in pixel units
        cmf = normal_cdf(edges / sigma)
        w = cmf[1:] - cmf[:-1]  # length len(j)

        # Normalise row
        s = w.sum()
        if s > 0:
            w /= s
        else:
            # fallback: delta if something went very wrong numerically
            j = np.array([i])
            w = np.array([1.0])

        rows.extend([i] * len(j))
        cols.extend(j.tolist())
        data.extend(w.tolist())

    # Sparse matrix with non-zero entries only in the local Gaussian windows
    W = sparse.csr_matrix((data, (rows, cols)), shape=(n_pix, n_pix))

    # Flatten all non-spectral dimensions and apply the convolution
    orig_shape = spec_vals.shape
    spec_flat = spec_vals.reshape(-1, n_pix)           # (N_other, n_pix)
    # out = spec_flat @ W.T  (since rows of W map input->output)
    out_flat = spec_flat @ W.T
    out = out_flat.reshape(orig_shape)

    if has_unit:
        return out * unit
    return out