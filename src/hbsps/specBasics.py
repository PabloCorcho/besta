"""spectra module.

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


def getGaussianLP(w, wc, wstd, norm):
    """Calculate Gaussian line profile for local covariance structure"""
    glp = norm * np.exp(-((w - wc) ** 2 / (2 * wstd**2)))

    return glp


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


def getLegendrePolynomial(wavelength, order, bounds=None):
    nBins = len(wavelength)
    if bounds == None:
        wavelengthN = -1.0 + 2.0 * (wavelength - wavelength[0]) / (
            wavelength[-1] - wavelength[0]
        )
    else:
        wavelengthN = -1.0 + 2.0 * (wavelength - bounds[0]) / (bounds[1] - bounds[0])

    AL = np.zeros([nBins, order + 1])
    for oIdx in range(order + 1):
        pDegree = oIdx
        legendreP = np.array(legendre(pDegree))
        for dIdx in range(pDegree + 1):
            AL[:, pDegree] += legendreP[dIdx] * wavelengthN ** (pDegree - dIdx)

    return AL


def rebin(x1, x2, y):
    """Rebin a spectrum from grid x1 to grid x2

    Use this function to rebin a spectrum from grid x1 to grid
    x2. This routine conserves flux density but not flux.

    Args:
            x1: wavelength grid on which the spectrum is defined.
            x2: wavelength grid to which spectrum should be
            rebinned.
            y: the spectrum (i.e. flux vector).

    """

    # Define number of pixels in the two wavelength grids
    nPix1 = len(x1)
    nPix2 = len(x2)

    # Define the boundaries of the pixels for the
    # two wavelength grids
    step1 = x1[1:] - x1[:-1]
    step2 = x2[1:] - x2[:-1]

    binB1 = np.zeros(len(x1) + 1)
    binB2 = np.zeros(len(x2) + 1)

    binB1[0] = x1[0] - step1[0] / 2.0
    binB1[1:-1] = x1[:-1] + step1 / 2.0
    binB1[-1] = x1[-1] + step1[-1] / 2.0

    binB2[0] = x2[0] - step2[0] / 2.0
    binB2[1:-1] = x2[:-1] + step2 / 2.0
    binB2[-1] = x2[-1] + step2[-1] / 2.0

    # Determine where to insert boundaries of original
    # array into boundaries of the new array
    x2p = np.searchsorted(binB1, binB2)

    # Define rebinned flux vector
    b = np.zeros(len(x2))

    # Process all pixels of new wavelength grid
    # and find corresponding pixels in original
    # wavelength grid that contribute to the flux
    # in these pixels
    for pix in range(len(x2)):
        idS = max(x2p[pix] - 1, 0)
        idE = min((x2p[pix + 1], len(x1)))
        for id1 in range(idS, idE):
            wL = max(binB1[id1], binB2[pix])
            wR = min(binB1[id1 + 1], binB2[pix + 1])
            if id1 == 0:
                wL = binB2[pix]
            if id1 == len(x1) - 1:
                wR = binB2[pix + 1]
            b[pix] += (
                y[id1]
                * (wR - wL)
                / (binB1[id1 + 1] - binB1[id1])
                * (binB1[id1 + 1] - binB1[id1])
            )
        b[pix] /= binB2[pix + 1] - binB2[pix]

    return b
