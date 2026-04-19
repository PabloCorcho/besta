"""
This module contains classes and functions related
to dealing with spectra

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional

import numpy as np
import scipy
from scipy import ndimage
from scipy.special import legendre
from astropy import constants
from astropy import units as u
from besta.logging import get_logger

logger = get_logger(__name__)

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
    logger.debug("Pol order == %s", np.arange(min_order, min_order + order + 1))
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
            coeffs = np.array([1.0] + [args[1]["legendre", f"legendre_{ith}"] for ith in range(1, legendre_pol.shape[0])])
            output = make_observable_mthd(*args, **kwargs)
            if isinstance(output, tuple):
                return output[0] * np.sum(legendre_pol * coeffs[:, np.newaxis], axis=0), output[1]
            else:
                return output * np.sum(legendre_pol * coeffs[:, np.newaxis], axis=0)
        else:
            return make_observable_mthd(*args, **kwargs)
    return wrapper

### Masking

# Telluric regions

@dataclass(frozen=True)
class TelluricBand:
    """Named wavelength interval affected by telluric absorption."""

    name: str
    wmin: float  # in Angstrom
    wmax: float  # in Angstrom

DEFAULT_TELLURIC_BANDS_AA: Tuple[TelluricBand, ...] = (
    # --- Optical ---
    # TelluricBand("O3 Chappuis (broad)", 5000.0, 7000.0),

    TelluricBand("O2 gamma (~0.63 um)", 6270.0, 6340.0),
    # TelluricBand("H2O (~0.65 um)",      6450.0, 6600.0),

    TelluricBand("O2 B-band",            6860.0, 6950.0),
    TelluricBand("H2O (~0.72 um)",       7160.0, 7400.0),

    TelluricBand("O2 A-band",            7580.0, 7700.0),
    TelluricBand("H2O (~0.82 um)",       8100.0, 8400.0),
    TelluricBand("H2O (~0.93 um)",       9000.0, 9900.0),

    # --- Near-IR ---
    TelluricBand("H2O (~1.13 um)",      11000.0, 11700.0),
    TelluricBand("H2O (~1.40 um)",      13400.0, 15000.0),
    TelluricBand("H2O/CO2 (~1.90 um)",  18000.0, 20000.0),
)

def mask_telluric_regions(
    wavelength: np.ndarray,
    *,
    weight: Optional[np.ndarray] = None,
    bands: Optional[Sequence[Tuple[float, float]]] = None,
    pad: float = 0.0,
    return_mask: bool = False,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """
    Mask regions affected by telluric absorption by setting weights to 0.

    Parameters
    ----------
    wavelength : ndarray
        1D array of wavelength values in Angstrom.

    bands : sequence of (wmin, wmax), optional
        Telluric band edges in the same units as `wavelength`.
        If None, uses a default set.

    pad : float, optional
        Extra padding (same units as `wavelength`) added to both sides of every
        band: [wmin - pad, wmax + pad].

    return_mask : bool, optional
        If True, also return the boolean mask of where weights were set to 0.

    Returns
    -------
    new_weight : ndarray
        Copy of input weight with telluric regions set to 0.
    mask : ndarray of bool, optional
        If ``return_mask`` is True, also return the boolean mask of where weights were set to 0.
    """
    w = np.asanyarray(wavelength)
    if weight is None:
        wt = np.ones_like(w)
    else:
        wt = np.asanyarray(weight)

    if w.ndim != 1 or wt.ndim != 1:
        raise ValueError("All inputs must be 1D arrays.")
    if not (w.size == wt.size):
        raise ValueError("All inputs must have the same length.")

    # Build list of bands in the same units as wavelength array (i.e. input units)
    if bands is None:
        # Defaults are in microns; convert to input units by dividing by to_um
        bands_in_input_units = [
            (b.wmin, b.wmax) for b in DEFAULT_TELLURIC_BANDS_AA]
    else:
        bands_in_input_units = list(bands)

    # Combine masks across all bands
    tell_mask = np.zeros_like(w, dtype=bool)
    for (wmin, wmax) in bands_in_input_units:
        a = min(wmin, wmax) - pad
        b = max(wmin, wmax) + pad
        tell_mask |= (w >= a) & (w <= b)

    new_weight = np.array(wt, copy=True)
    new_weight[tell_mask] = 0.0

    if return_mask:
        return new_weight, tell_mask
    return new_weight

# Emission lines

@dataclass(frozen=True)
class EmissionLine:
    """Emission line definition."""
    name: str
    rest_wavelength: float  # same units as wavelength array (e.g. Angstrom)
    # optional default half-width (mask window half-size) in same units
    default_half_width: float = 10.0


# A practical optical line list (air/vac differences ignored here; windows are wide anyway)
DEFAULT_EMISSION_LINES_A: Tuple[EmissionLine, ...] = (
    EmissionLine("[OII]3726", 3726.03, 8.0),
    EmissionLine("[OII]3729", 3728.82, 8.0),
    EmissionLine("Hδ",        4101.74, 10.0),
    EmissionLine("Hγ",        4340.47, 10.0),
    EmissionLine("[OIII]4363",4363.21, 8.0),
    EmissionLine("Hβ",        4861.33, 12.0),
    EmissionLine("[OIII]4959",4958.91, 10.0),
    EmissionLine("[OIII]5007",5006.84, 10.0),
    EmissionLine("[NI]5200",  5199.0,  10.0),
    EmissionLine("HeI5876",   5875.62, 12.0),
    EmissionLine("[OI]6300",  6300.30, 10.0),
    EmissionLine("[OI]6364",  6363.78, 10.0),
    EmissionLine("[NII]6548", 6548.05, 12.0),
    EmissionLine("Hα",        6562.80, 14.0),
    EmissionLine("[NII]6583", 6583.45, 12.0),
    EmissionLine("[SII]6716", 6716.44, 12.0),
    EmissionLine("[SII]6731", 6730.82, 12.0),
    EmissionLine("[ArIII]7136",7135.79, 12.0),
)


def mask_strong_emission_lines(
    wavelength: np.ndarray,
    flux: np.ndarray,
    uncertainty: np.ndarray,
    weight: np.ndarray,
    *,
    redshift: float = 0.0,
    lines: Optional[Sequence[EmissionLine]] = None,
    # detection controls
    snr_threshold: float = 5.0,
    min_continuum_snr: float = 1.0,
    # local continuum estimation
    cont_half_window: int = 75,
    cont_sigma_clip: float = 4.0,
    # mask width controls
    half_width: Optional[float] = None,
    width_mode: str = "fixed",  # "fixed" | "fwhm_pixels"
    fwhm_pixels: float = 3.0,
    max_half_width: Optional[float] = None,
    pad: float = 0.0,
    # bookkeeping
    return_mask: bool = False,
    return_lines_masked: bool = False,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, list[str]]
):
    """
    Identify strong emission lines and mask them by setting weight=0.

    This is a robust, low-assumption approach:
      1) For each expected line center (rest -> observed using `z`), estimate
         a local continuum with a running median in a window around the line.
      2) Compute line "excess" = flux - continuum and its S/N using uncertainty.
      3) If peak S/N within the line window exceeds `snr_threshold` (and the
         continuum is not completely noise-dominated), mask the line region.

    Parameters
    ----------
    wavelength, flux, uncertainty, weight : ndarray
        1D arrays of the same length.

    z : float, optional
        Redshift used to place the line centers: lambda_obs = lambda_rest*(1+z).

    lines : sequence of EmissionLine, optional
        Line list. If None, uses DEFAULT_EMISSION_LINES_A (Angstrom).

    snr_threshold : float, optional
        Minimum peak (flux-continuum)/sigma within the line window to mask.

    min_continuum_snr : float, optional
        Require median(abs(continuum)/sigma) in the continuum window to be at least
        this value; helps avoid false positives when everything is noise.

    cont_half_window : int, optional
        Half-window size in pixels for local continuum estimation around each line.

    cont_sigma_clip : float, optional
        Sigma-clipping level applied to the continuum residuals when estimating
        the median continuum (simple, robust clip).

    half_width : float, optional
        Half-width of the masked region in wavelength units. If None, uses
        each line's `default_half_width`.

    width_mode : {"fixed","fwhm_pixels"}, optional
        - "fixed": uses `half_width` (or per-line default) in wavelength units.
        - "fwhm_pixels": convert `fwhm_pixels` to wavelength half-width using the
          local dispersion (median delta-lambda near the line) and mask +-k*FWHM,
          where k=1.5 (roughly covers wings).

    fwhm_pixels : float, optional
        Only used if width_mode="fwhm_pixels". Instrumental/profile FWHM in pixels.

    max_half_width : float, optional
        Cap the computed half-width (in wavelength units) to avoid huge masks.

    pad : float, optional
        Extra padding added to half-width (same wavelength units).

    return_mask : bool, optional
        If True, also return boolean mask of where weights were set to 0.

    return_lines_masked : bool, optional
        If True, also return list of line names that were actually masked.

    Returns
    -------
    new_weight : ndarray
        Copy of weight with emission-line regions set to 0.

    mask : ndarray of bool, optional
        True where masked.

    lines_masked : list of str, optional
        Names of lines that triggered masking.

    Notes
    -----
    - This targets *strong, narrow-ish* features near known lines. It will not
      detect arbitrary lines at unknown wavelengths unless you expand the line list.
    - If you already have a model continuum, you can replace the continuum
      estimation with that for even more robustness.
    """
    w = np.asanyarray(wavelength)
    f = np.asanyarray(flux)
    s = np.asanyarray(uncertainty)
    wt = np.asanyarray(weight)

    if any(arr.ndim != 1 for arr in (w, f, s, wt)):
        raise ValueError("All inputs must be 1D arrays.")
    if not (w.size == f.size == s.size == wt.size):
        raise ValueError("All inputs must have the same length.")
    if w.size < 5:
        new_weight = np.array(wt, copy=True)
        if return_mask and return_lines_masked:
            return new_weight, np.zeros_like(w, dtype=bool), []
        if return_mask:
            return new_weight, np.zeros_like(w, dtype=bool)
        return new_weight

    # Ensure sorted wavelength (common in spectra). If not sorted, do it safely.
    # Keep mapping so we still return an array matching the original order.
    order = np.argsort(w)
    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(order.size)

    w_s = w[order]
    f_s = f[order]
    s_s = s[order]
    wt_s = wt[order]

    if lines is None:
        lines = DEFAULT_EMISSION_LINES_A

    # Precompute pixel dispersion (delta-lambda), robust median for local conversion
    dw = np.diff(w_s)
    # Replace non-positive steps (if duplicates) with nan for robust median
    dw = np.where(dw > 0, dw, np.nan)

    masked = np.zeros_like(w_s, dtype=bool)
    lines_masked: list[str] = []

    # Helper: continuum estimate around a given index window
    def _local_continuum(i0: int, i1: int) -> float:
        # simple robust median with sigma-clip on residuals
        seg_f = f_s[i0:i1]
        seg_s = s_s[i0:i1]
        # ignore already-zero-weight points for continuum
        seg_wt = wt_s[i0:i1]
        good = (seg_wt > 0) & np.isfinite(seg_f) & np.isfinite(seg_s) & (seg_s > 0)
        if good.sum() < 5:
            return float(np.nanmedian(seg_f))  # fallback
        x = seg_f[good]
        med = np.median(x)
        # clip using provided uncertainties if reasonable, else MAD
        sig = seg_s[good]
        # avoid division by zero
        sig = np.where(sig > 0, sig, np.nanmedian(sig[sig > 0]) if np.any(sig > 0) else 1.0)
        resid = x - med
        keep = np.abs(resid) <= cont_sigma_clip * sig
        if keep.sum() < 5:
            return float(med)
        return float(np.median(x[keep]))

    # Core loop over lines
    for line in lines:
        lam0 = line.rest_wavelength * (1.0 + redshift)
        # skip if out of wavelength coverage
        if lam0 < w_s[0] or lam0 > w_s[-1]:
            continue

        # Find nearest pixel to line center
        j = int(np.searchsorted(w_s, lam0))
        j = max(0, min(j, w_s.size - 1))

        # Define a pixel window for continuum around the line
        i0 = max(0, j - cont_half_window)
        i1 = min(w_s.size, j + cont_half_window + 1)

        cont = _local_continuum(i0, i1)
        if not np.isfinite(cont):
            continue

        # Require continuum not to be pure noise (helps avoid false triggers)
        seg_s = s_s[i0:i1]
        seg_wt = wt_s[i0:i1]
        good = (seg_wt > 0) & np.isfinite(seg_s) & (seg_s > 0)
        if good.sum() >= 5:
            cont_snr = np.median(np.abs(cont) / seg_s[good])
            if np.isfinite(cont_snr) and cont_snr < min_continuum_snr:
                # still allow masking if the line is extremely significant;
                # we keep going but require higher SNR implicitly via snr_threshold.
                pass

        # Determine mask half-width (in wavelength units)
        if width_mode.lower() == "fixed":
            hw = float(line.default_half_width if half_width is None else half_width)
        elif width_mode.lower() == "fwhm_pixels":
            # convert pixels -> wavelength using local dispersion near j
            k0 = max(0, j - 5)
            k1 = min(dw.size, j + 5)
            local_dw = np.nanmedian(dw[k0:k1])
            if not np.isfinite(local_dw) or local_dw <= 0:
                local_dw = np.nanmedian(dw)
            if not np.isfinite(local_dw) or local_dw <= 0:
                # cannot estimate dispersion; fallback to fixed default
                hw = float(line.default_half_width if half_width is None else half_width)
            else:
                # mask ~ +/- 1.5*FWHM (covers core+wings for most cases)
                hw = 1.5 * float(fwhm_pixels) * float(local_dw)
        else:
            raise ValueError(f"Unsupported width_mode={width_mode!r}. Use 'fixed' or 'fwhm_pixels'.")

        hw += float(pad)
        if max_half_width is not None:
            hw = min(hw, float(max_half_width))
        hw = max(hw, 0.0)

        # Evaluate peak SNR in the line window (use continuum-subtracted flux)
        # Build index window from wavelength bounds
        a = lam0 - hw
        b = lam0 + hw
        k0 = int(np.searchsorted(w_s, a, side="left"))
        k1 = int(np.searchsorted(w_s, b, side="right"))
        if k1 - k0 < 3:
            continue

        seg_f = f_s[k0:k1]
        seg_s = s_s[k0:k1]
        seg_wt = wt_s[k0:k1]
        good = (seg_wt > 0) & np.isfinite(seg_f) & np.isfinite(seg_s) & (seg_s > 0)
        if good.sum() < 3:
            continue

        excess = seg_f[good] - cont
        snr = excess / seg_s[good]
        peak_snr = float(np.nanmax(snr)) if snr.size else -np.inf

        if np.isfinite(peak_snr) and peak_snr >= snr_threshold:
            masked[k0:k1] = True
            lines_masked.append(line.name)

    new_wt_s = np.array(wt_s, copy=True)
    new_wt_s[masked] = 0.0

    # Restore original wavelength order
    new_weight = new_wt_s[inv_order]
    mask = masked[inv_order]

    if return_mask and return_lines_masked:
        return new_weight, mask, lines_masked
    if return_mask:
        return new_weight, mask
    return new_weight
