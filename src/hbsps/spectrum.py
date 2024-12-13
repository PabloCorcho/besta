"""
This module contains classes and functions related
to dealing with spectra

"""
import numpy as np
import scipy
from scipy import ndimage
from scipy.special import legendre
from astropy import constants


def log_rebin(lam, spec, velscale=None, oversample=1, flux=False):
    """
    Logarithmically rebin a spectrum, or the first dimension of an array of
    spectra arranged as columns, while rigorously conserving the flux. The
    photons in the spectrum are simply redistributed according to a new grid of
    pixels, with logarithmic sampling in the spectral direction.

    When `flux=True` keyword is set, this program performs an exact integration
    of the original spectrum, assumed to be a step function constant within
    each pixel, onto the new logarithmically-spaced pixels. When `flux=False`
    (default) the result of the integration is divided by the size of each
    pixel to return a flux density (e.g. in erg/(s cm^2 A)). The output was
    tested to agree with the analytic solution.

    Input Parameters
    ----------------

    lam: either [lam_min, lam_max] or wavelength `lam` per spectral pixel.
        * If this has two elements, they are assumed to represent the central
          wavelength of the first and last pixels in the spectrum, which is
          assumed to have constant wavelength scale.
          log_rebin is faster with regular sampling.
        * Alternatively one can input the central wavelength of every spectral
          pixel and this allows for arbitrary irregular sampling in wavelength.
          In this case the program assumes the pixels edges are the midpoints
          of the input pixels wavelengths.

        EXAMPLE: For uniform wavelength sampling, using the values in the
        standard FITS keywords (but note that the format can be different)::

            lam = CRVAL1 + CDELT1*np.arange(NAXIS1)

    spec: array_like with shape (npixels,) or (npixels, nspec)
        Input spectrum or array of spectra to rebin logarithmically.
        This can be a vector `spec[npixels]` or array `spec[npixels, nspec]`.
    oversample: int
        Can be used, not to degrade spectral resolution, especially for
        extended wavelength ranges and to avoid aliasing. Default:
        `oversample=1` implies same number of output pixels as input.
    velscale: float
        Velocity scale in km/s per pixels. If this variable is not defined, it
        will be computed to produce the same number of output pixels as the
        input. If this variable is defined by the user it will be used to set
        the output number of pixels and wavelength scale.
    flux: bool
        `True` to preserve total flux, `False` to preserve the flux density.
        When `flux=True` the log rebinning changes the pixels flux in
        proportion to their dlam and the following command will show large
        differences between the spectral shape before and after `log_rebin`::

           plt.plot(exp(ln_lam), specNew)  # Plot log-rebinned spectrum
           plt.plot(np.linspace(lam[0], lam[1], spec.size), spec)

        By default `flux=`False` and `log_rebin` returns a flux density and the
        above two lines produce two spectra that almost perfectly overlap each
        other.

    Output Parameters
    -----------------

    spec_new:
        Logarithmically-rebinned spectrum flux.

    ln_lam:
        Natural logarithm of the wavelength.

    velscale:
        Velocity scale per pixel in km/s.

    """
    lam, spec = np.asarray(lam, dtype=float), np.asarray(spec, dtype=float)
    assert np.all(np.diff(lam) > 0), "`lam` must be monotonically increasing"
    n = len(spec)
    assert lam.size in [
        2,
        n,
    ], "`lam` must be either a 2-elements range or a vector with the length of `spec`"

    if lam.size == 2:
        dlam = np.diff(lam) / (n - 1)  # Assume constant dlam
        lim = lam + [-0.5, 0.5] * dlam
        borders = np.linspace(*lim, n + 1)
    else:
        lim = 1.5 * lam[[0, -1]] - 0.5 * lam[[1, -2]]
        borders = np.hstack([lim[0], (lam[1:] + lam[:-1]) / 2, lim[1]])
        dlam = np.diff(borders)

    ln_lim = np.log(lim)
    c = constants.c.to("km/s").value  # Speed of light in km/s

    if velscale is None:
        m = int(n * oversample)  # Number of output elements
        velscale = (
            c * np.diff(ln_lim) / m
        )  # Only for output (eq. 8 of Cappellari 2017, MNRAS)
        velscale = velscale.item()  # Make velscale a scalar
    else:
        ln_scale = velscale / c
        m = int(np.diff(ln_lim) / ln_scale)  # Number of output pixels

    newBorders = np.exp(ln_lim[0] + velscale / c * np.arange(m + 1))

    if lam.size == 2:
        k = ((newBorders - lim[0]) / dlam).clip(0, n - 1).astype(int)
    else:
        k = (np.searchsorted(borders, newBorders) - 1).clip(0, n - 1)

    specNew = np.add.reduceat((spec.T * dlam).T, k)[
        :-1
    ]  # Do analytic integral of step function
    specNew.T[...] *= np.diff(k) > 0  # fix for design flaw of reduceat()
    specNew.T[...] += np.diff(
        ((newBorders - borders[k])) * spec[k].T
    )  # Add to 1st dimension

    if not flux:
        specNew.T[...] /= np.diff(newBorders)  # Divide 1st dimension

    # Output np.log(wavelength): natural log of geometric mean
    ln_lam = 0.5 * np.log(newBorders[1:] * newBorders[:-1])

    return specNew, ln_lam, velscale


def smoothSpectrum(wavelength, spectrum, sigma):
    """Smooth spectrum to a given velocity dispersion.

    Args:
            wavelength: wavelength-array of the spectrum (should
                    be logarithmic for constant sigma-smoothing).
            spectrum: numpy array with spectral data.
            sigma: required velocity dispersion (km/s)

    Returns:
            spectrumSmooth: smoothed version of the spectrum.

    """

    clight = 299792.458
    cdelt = np.log(wavelength[1]) - np.log(wavelength[0])
    sigmaPixel = sigma / (clight * cdelt)
    smoothSpectrum = smoothSpectrumFast(spectrum, sigmaPixel)

    return smoothSpectrum


def smoothSpectra(wavelength, S, sigma):
    """Smooth spectra in matrix with stellar spectra to a given velocity dispersion.

    Args:
            wavelength: wavelength-array of the spectra (should
                    be logarithmic for constant sigma smoothing).
            S: matrix with stellar templates, spectra are assumed to be
                    int the columns of the matrix.
            spectrum: numpy array with spectral data.
            sigma: required velocity dispersion (km/s)

    Returns:
            S: smoothed version of the spectra in S.

    """
    clight = 299792.458
    cdelt = np.log(wavelength[1]) - np.log(wavelength[0])
    sigmaPixel = sigma / (clight * cdelt)

    nTemplates = S.shape[1]
    for tIdx in range(nTemplates):
        S[:, tIdx] = smoothSpectrumFast(S[:, tIdx], sigmaPixel)

    return S


def smoothSpectrumFast(spectrum, sigmaPixel):
    """Fast spectrum smoothing.

    This function smooths a spectrum given the
    standard deviation in pixel space.

    Args:
            spectrum: the input spectrum.
            sigmaPixel: smoothing scale in pixel space.

    Returns:
            smoothSpectrum: a smoothed version of the
                    input spectrum.

    """

    smoothSpectrum = scipy.ndimage.gaussian_filter(
        spectrum, sigma=(sigmaPixel), order=0
    )

    return smoothSpectrum

def get_legendre_polynomial_array(wavelength, order, bounds=None):
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

    Returns
    -------
    numpy.ndarray
        A 2D array where each row corresponds to the values of a Legendre 
        polynomial of a given degree, evaluated at the normalized wavelengths.
        The shape of the array is (order + 1, len(wavelength)).
    """
    if bounds == None:
        norm_wl = 2 * (wavelength - wavelength.min()
                   ) / (wavelength.max() - wavelength.min()) - 1
    else:
        norm_wl = 2 * (wavelength - bounds[0]) / (bounds[1] - bounds[0]) - 1
    legendre_arr = np.array(
        [np.array(legendre(deg)(norm_wl)) for deg in  np.arange(0, order + 1)])
    return legendre_arr
