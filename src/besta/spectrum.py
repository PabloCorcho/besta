"""
This module contains classes and functions related
to dealing with spectra

"""
import numpy as np
import scipy
from scipy import ndimage
from scipy.special import legendre
from astropy import constants
from astropy import units as u

def get_legendre_polynomial_array(wavelength, order, bounds=None, scale=None,
                                  clip_first_zero=True):
    """
    Compute an array of Legendre polynomials evaluated at normalized wavelengths.

    Parameters
    ----------
    wavelength : numpy.ndarray
        Array of wavelength values.
    order : int
        The maximum order of the Legendre polynomial to compute.
    bounds : tuple, optional
        A tuple specifying the minimum and maximum bounds for normalization
        (bounds[0], bounds[1]). If None, the normalization is based on the
        minimum and maximum of the `wavelength` array.
    scale : float, optional
        A maximum scale to probe by the polynomials. If provided, the set of
        polynomial will comprise the range that is sensitive to scales smaller
        the input value (i.e. lower order polynomials are not included).
    clip_first_zero : bool, optional
        If ``True``, the values of each polynomial below the first and las zero
        of the Legendre polynomial are set to 0. This prevents the edges to reach
        extremelly large values when the order of the polynomial is high.

    Returns
    -------
    numpy.ndarray
        A 2D array where each row corresponds to the values of a Legendre
        polynomial of a given degree, evaluated at the normalized wavelengths.
        The shape of the array is (order + 1, len(wavelength)).
    """
    if bounds == None:
        bounds = wavelength.min(), wavelength.max()
    norm_wl = 2 * (wavelength - bounds[0]) / (bounds[1] - bounds[0]) - 1
    norm_wl = norm_wl.clip(-1, 1)

    if scale is not None:
        min_order = np.round((bounds[1].value - bounds[0].value) / scale, 0)
    else:
        min_order = 1

    if isinstance(norm_wl, u.Quantity):
        norm_wl = norm_wl.decompose().value
    print("Pol order == ", np.arange(min_order, min_order + order + 1))
    poly_set = []
    for deg in [0, *np.arange(min_order, min_order + order)]:
        pol = legendre(deg)
        pol_wl = pol(norm_wl)
        # Clip the values on the edges to avoid extremes
        if clip_first_zero and deg > 0:
            first_zero = pol.roots.real.min()
            pol_wl[(norm_wl < first_zero) | (norm_wl > -first_zero)] = 0
        poly_set.append(pol_wl)

    legendre_arr = np.array(poly_set)
    return legendre_arr

def legendre_decorator(make_observable_mthd):
    """Include multiplicative Legendre polynomials during a fit."""
    def wrapper(*args, **kwargs):
        if "legendre_pol" in args[0].config:
            legendre_pol = args[0].config["legendre_pol"]
            # Get the coefficients from the input DataBlock
            coeffs = np.array([1.0] + [args[1]["parameters", f"legendre_{ith}"] for ith in range(1, legendre_pol.shape[0])])
            output = make_observable_mthd(*args, **kwargs)
            if isinstance(output, tuple):
                return output[0] * np.sum(legendre_pol * coeffs[:, np.newaxis], axis=0), output[1]
            else:
                return output * np.sum(legendre_pol * coeffs[:, np.newaxis], axis=0)
        else:
            return make_observable_mthd(*args, **kwargs)
    return wrapper
