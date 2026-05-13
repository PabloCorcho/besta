"""
This module contains classes and functions related
to dealing with spectra

"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence, Tuple, Optional

import numpy as np
import scipy
from scipy import ndimage
from scipy.special import legendre
from scipy.signal import find_peaks
from scipy.ndimage import median_filter, label, find_objects
from scipy.interpolate import make_smoothing_spline, interp1d
from scipy.optimize import curve_fit

from astropy.table import Table
from astropy import constants
from astropy import units as u

from besta.logging import get_logger

logger = get_logger(__name__)


def get_legendre_polynomial_array(
    wavelength, order, bounds=None, scale=None, clip_first_zero=True
):
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
            coeffs = np.array(
                [1.0]
                + [
                    args[1]["legendre", f"legendre_{ith}"]
                    for ith in range(1, legendre_pol.shape[0])
                ]
            )
            output = make_observable_mthd(*args, **kwargs)
            if isinstance(output, tuple):
                return (
                    output[0] * np.sum(legendre_pol * coeffs[:, np.newaxis], axis=0),
                    output[1],
                )
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
    TelluricBand("O2 B-band", 6860.0, 6950.0),
    TelluricBand("H2O (~0.72 um)", 7160.0, 7400.0),
    TelluricBand("O2 A-band", 7580.0, 7700.0),
    TelluricBand("H2O (~0.82 um)", 8100.0, 8400.0),
    TelluricBand("H2O (~0.93 um)", 9000.0, 9900.0),
    # --- Near-IR ---
    TelluricBand("H2O (~1.13 um)", 11000.0, 11700.0),
    TelluricBand("H2O (~1.40 um)", 13400.0, 15000.0),
    TelluricBand("H2O/CO2 (~1.90 um)", 18000.0, 20000.0),
)


def mask_telluric_regions(
    wavelength: np.ndarray,
    *,
    weight: Optional[np.ndarray] = None,
    bands: Optional[Sequence[Tuple[float, float]]] = None,
    pad: float = 0.0,
    redshift: float = 0.0,
    return_mask: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, list[TelluricBand]]:
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
    logger.info("Masking telluric regions with pad=%.1f Angstrom", pad)
    if redshift != 0.0:
        logger.warning(
            "Redshift is non-zero (z=%.3f); telluric bands will be shifted accordingly. Make sure this is intended.",
            redshift,
        )
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
        logger.info(
            "Using default telluric bands with %d bands.",
            len(DEFAULT_TELLURIC_BANDS_AA),
        )
        bands = DEFAULT_TELLURIC_BANDS_AA

    # Combine masks across all bands
    tell_mask = np.zeros_like(w, dtype=bool)
    bands_used = []
    for band in bands:
        a = band.wmin - pad
        b = band.wmax + pad
        overlap = (w >= a) & (w <= b)
        if overlap.any():
            tell_mask |= overlap
            bands_used.append(band)

    logger.info(
        "Masked %d pixels in %d telluric bands", tell_mask.sum(), len(bands_used)
    )
    new_weight = np.array(wt, copy=True)
    new_weight[tell_mask] = 0.0

    if return_mask:
        return new_weight, tell_mask, bands_used
    return new_weight


# Emission lines


@dataclass(frozen=True)
class EmissionLine:
    """Emission line definition."""

    name: str
    rest_wavelength: float
    default_half_width: float = 10.0
    flux: Optional[float] = None
    flux_error: Optional[float] = None
    rest_wavelength_error: Optional[float] = None
    flag: int = 0
    metadata: dict = field(default_factory=dict)


class EmissionLineList:
    """Collection of emission lines with helper methods."""

    def __init__(self, lines: Iterable[EmissionLine]):
        self.lines = list(lines)
        self.size = len(self.lines)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return self.lines[item]

    @property
    def rest_wavelengths(self) -> np.ndarray:
        """Array of rest wavelengths of the lines."""
        return np.array([line.rest_wavelength for line in self.lines])

    @property
    def names(self) -> list[str]:
        """List of line names."""
        return [line.name for line in self.lines]

    @property
    def default_half_widths(self) -> np.ndarray:
        """Array of default half-widths of the lines."""
        return np.array([line.default_half_width for line in self.lines])

    @property
    def fluxes(self) -> np.ndarray:
        """Array of measured line fluxes."""
        return np.array(
            [np.nan if line.flux is None else line.flux for line in self.lines]
        )

    @property
    def flux_errors(self) -> np.ndarray:
        """Array of measured line-flux uncertainties."""
        return np.array(
            [
                np.nan if line.flux_error is None else line.flux_error
                for line in self.lines
            ]
        )

    @property
    def rest_wavelength_errors(self) -> np.ndarray:
        """Array of line-center uncertainties."""
        return np.array(
            [
                np.nan
                if line.rest_wavelength_error is None
                else line.rest_wavelength_error
                for line in self.lines
            ]
        )

    @property
    def flags(self) -> np.ndarray:
        """Array of line quality flags."""
        return np.array([line.flag for line in self.lines], dtype=int)

    def __iter__(self):
        """Allow iteration over the lines in the list."""
        return iter(self.lines)

    def get_observed_wavelengths(self, redshift: float) -> np.ndarray:
        """Compute observed wavelengths of the lines given a redshift.

        Parameters
        ----------
        redshift : float
            Redshift to apply to the rest wavelengths.

        Returns
        -------
        np.ndarray
            Array of observed wavelengths corresponding to the input redshift.
        """
        return np.array([line.rest_wavelength * (1 + redshift) for line in self.lines])

    def crossmatch_line_list(self, line_list: EmissionLineList):
        """Cross-match this line list with another line list and return the matched lines."""
        matched_lines = []
        for line in self.lines:
            for other_line in line_list:
                if np.isclose(
                    line.rest_wavelength, other_line.rest_wavelength, atol=1e-3
                ):
                    matched_lines.append(line)
                    break
        return EmissionLineList(matched_lines)

    def with_names(self, names: Sequence[str]) -> EmissionLineList:
        """Return a copy of the line list with updated names."""
        if len(names) != len(self.lines):
            raise ValueError("Number of names must match number of lines.")
        return EmissionLineList(
            [replace(line, name=name) for line, name in zip(self.lines, names)]
        )

    def get_closest_line(
        self, wavelength: float, *, redshift: float = 0.0
    ) -> Optional[EmissionLine]:
        """Return the line with observed wavelength closest to the input value."""
        if not self.lines:
            return None
        observed_wavelengths = self.get_observed_wavelengths(redshift)
        index = int(np.argmin(np.abs(observed_wavelengths - wavelength)))
        return self.lines[index]

    def get_line_by_observed_wavelength(
        self, wavelength: float, redshift: float, tol: float = 5.0
    ) -> Optional[EmissionLine]:
        """Match an input line to the list of emission lines.

        Parameters
        ----------
        wavelength : float
            Observed wavelength to match.
        redshift : float
            Redshift to apply to the rest wavelengths.
        tol : float, optional
            Tolerance for matching, by default 5.0.

        Returns
        -------
        Optional[EmissionLine]
            The matched emission line, or None if no match is found.
        """
        for line in self.lines:
            obs_wl = line.rest_wavelength * (1 + redshift)
            if abs(obs_wl - wavelength) <= tol:
                return line
        return None

    def to_mask(self, wavelength, redshift=0.0) -> np.ndarray:
        """Create a boolean mask for the input wavelength array where the lines are located.

        Parameters
        ----------
        wavelength : np.ndarray
            Array of wavelength values to create the mask for.
        redshift : float, optional
            Redshift to apply to the rest wavelengths, by default 0.0.

        Returns
        -------
        np.ndarray
            Boolean array of the same shape as `wavelength` where True indicates the presence of a line.
        """
        mask = np.zeros_like(wavelength, dtype=bool)
        for line in self.lines:
            obs_wl = line.rest_wavelength * (1 + redshift)  # Apply redshift
            line_mask = (wavelength >= obs_wl - line.default_half_width) & (
                wavelength <= obs_wl + line.default_half_width
            )
            mask |= line_mask

        return mask

    def to_table(self, filename: str = None, **table_kwargs) -> Table:
        """Save emission lines to a file.

        Parameters
        ----------
        filename : str, optional
            If provided, the table is saved to this file.

        Returns
        -------
        astropy.table.Table
            Table containing the emission line information.
        """
        table = Table()
        table["name"] = self.names
        table["rest_wavelength"] = self.rest_wavelengths
        table["default_half_width"] = self.default_half_widths
        if np.isfinite(self.fluxes).any():
            table["flux"] = self.fluxes
        if np.isfinite(self.flux_errors).any():
            table["flux_error"] = self.flux_errors
        if np.isfinite(self.rest_wavelength_errors).any():
            table["rest_wavelength_error"] = self.rest_wavelength_errors
        if np.any(self.flags != 0):
            table["flag"] = self.flags
        if filename is not None:
            table.write(filename, overwrite=True, **table_kwargs)
        return table

    @classmethod
    def from_table(cls, table: Table) -> EmissionLineList:
        """Create an EmissionLineList from an astropy Table.

        Parameters
        ----------
        table : astropy.table.Table
            Table containing the emission line information with columns: name, rest_wavelength, default_half_width.

        Returns
        -------
        EmissionLineList
            An instance of EmissionLineList created from the input table.
        """
        names = table["name"] if "name" in table.colnames else table[table.colnames[0]]
        rest_wavelengths = (
            table["rest_wavelength"]
            if "rest_wavelength" in table.colnames
            else table[table.colnames[1]]
        )

        if "default_half_width" in table.colnames:
            default_half_widths = table["default_half_width"]
        elif len(table.colnames) >= 3:
            default_half_widths = table[table.colnames[2]]
        else:
            default_half_widths = np.full(len(rest_wavelengths), 10.0)

        fluxes = (
            table["flux"]
            if "flux" in table.colnames
            else np.full(len(rest_wavelengths), np.nan)
        )
        flux_errors = (
            table["flux_error"]
            if "flux_error" in table.colnames
            else np.full(len(rest_wavelengths), np.nan)
        )
        rest_wavelength_errors = (
            table["rest_wavelength_error"]
            if "rest_wavelength_error" in table.colnames
            else np.full(len(rest_wavelengths), np.nan)
        )
        flags = (
            table["flag"]
            if "flag" in table.colnames
            else np.zeros(len(rest_wavelengths), dtype=int)
        )

        lines = [
            EmissionLine(
                name=str(name),
                rest_wavelength=float(rest_wl),
                default_half_width=float(half_width),
                flux=None if not np.isfinite(flux) else float(flux),
                flux_error=None if not np.isfinite(flux_err) else float(flux_err),
                rest_wavelength_error=None
                if not np.isfinite(rest_err)
                else float(rest_err),
                flag=int(flag),
            )
            for name, rest_wl, half_width, flux, flux_err, rest_err, flag in zip(
                names,
                rest_wavelengths,
                default_half_widths,
                fluxes,
                flux_errors,
                rest_wavelength_errors,
                flags,
            )
        ]
        return cls(lines)

    @classmethod
    def from_file(cls, filename: str, **table_kwargs) -> EmissionLineList:
        """Load emission lines from a file with columns.

        Parameters
        ----------
        filename : str
            Path to the file containing the emission line data.
        **table_kwargs
            Additional keyword arguments to pass to `astropy.table.Table.read`.

        Returns
        -------
        EmissionLineList
            An instance of `EmissionLineList` containing the loaded emission lines.
        """
        data = Table.read(filename, **table_kwargs)
        return cls.from_table(data)


def get_default_emission_lines():
    """Return the default optical emission-line list.

    Returns
    -------
    EmissionLineList
        Default list of common rest-frame galaxy emission lines with nominal
        masking half-widths in Angstrom.
    """
    return EmissionLineList(
        [
            EmissionLine("[OII]3726", 3726.03, 8.0),
            EmissionLine("[OII]3729", 3728.82, 8.0),
            EmissionLine("Hd", 4101.74, 10.0),
            EmissionLine("Hg", 4340.47, 10.0),
            EmissionLine("[OIII]4363", 4363.21, 8.0),
            EmissionLine("Hb", 4861.33, 12.0),
            EmissionLine("[OIII]4959", 4958.91, 10.0),
            EmissionLine("[OIII]5007", 5006.84, 10.0),
            EmissionLine("[NI]5200", 5199.0, 10.0),
            EmissionLine("HeI5876", 5875.62, 12.0),
            EmissionLine("[OI]6300", 6300.30, 10.0),
            EmissionLine("[OI]6364", 6363.78, 10.0),
            EmissionLine("[NII]6548", 6548.05, 12.0),
            EmissionLine("Ha", 6562.80, 14.0),
            EmissionLine("[NII]6583", 6583.45, 12.0),
            EmissionLine("[SII]6716", 6716.44, 12.0),
            EmissionLine("[SII]6731", 6730.82, 12.0),
            EmissionLine("[ArIII]7136", 7135.79, 12.0),
        ]
    )


def get_default_sky_emission_lines():
    """Return the default sky emission-line list.

    Returns
    -------
    EmissionLineList
        Default list of common sky emission lines with nominal masking
        half-widths in Angstrom.
    """
    return EmissionLineList(
        [
            EmissionLine("NaI 5890", 5890.0, 10.0),
            EmissionLine("NaI 5896", 5896.0, 10.0),
            EmissionLine("OI 5577", 5577.34, 10.0),
            EmissionLine("OI 6300", 6300.30, 10.0),
            EmissionLine("OI 6364", 6363.78, 10.0),
        ]
    )


def _parse_lines_param_decorator(func):
    """Decorator to parse the `lines` parameter."""

    def wrapper(*args, **kwargs):
        if kwargs.get("lines") is not None:
            lines_input = kwargs["lines"]
            if isinstance(lines_input, EmissionLineList):
                # Already an EmissionLineList; use as is
                pass
            elif isinstance(lines_input, str):
                # Assume it is a filename; load the line list from the file
                kwargs["lines"] = EmissionLineList.from_file(lines_input)
            elif isinstance(lines_input, Iterable) and all(
                isinstance(line, EmissionLine) for line in lines_input
            ):
                # It's already an iterable of EmissionLine objects; convert to EmissionLineList
                kwargs["lines"] = EmissionLineList(lines_input)
            else:
                raise ValueError(
                    "Invalid format for 'lines' parameter. Must be a filename or an iterable of EmissionLine objects."
                )
        return func(*args, **kwargs)

    return wrapper


@_parse_lines_param_decorator
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
    | tuple[np.ndarray, np.ndarray, list[EmissionLine]]
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
    logger.info(
        "Masking strong emission lines with redshift z=%.3f and snr_threshold=%.1f",
        redshift,
        snr_threshold,
    )
    w = np.asanyarray(wavelength)
    f = np.asanyarray(flux)
    s = np.asanyarray(uncertainty)
    wt = np.asanyarray(weight)

    if any(arr.ndim != 1 for arr in (w, f, s, wt)):
        raise ValueError("All inputs must be 1D arrays.")
    if not (w.size == f.size == s.size == wt.size):
        raise ValueError("All inputs must have the same length.")
    if w.size < 5:
        logger.warning(
            "Input arrays have less than 5 pixels; skipping emission line masking."
        )
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
        lines = get_default_emission_lines()
        logger.info("Using default emission line list with %d lines.", lines.size)

    # Precompute pixel dispersion (delta-lambda), robust median for local conversion
    dw = np.diff(w_s)
    # Replace non-positive steps (if duplicates) with nan for robust median
    dw = np.where(dw > 0, dw, np.nan)

    masked = np.zeros_like(w_s, dtype=bool)
    lines_masked: list[EmissionLine] = []

    # Helper: continuum estimate around a given index window
    def _local_continuum(i0: int, i1: int) -> float:
        # simple robust median with sigma-clip on residuals
        seg_f = f_s[i0:i1]
        seg_s = s_s[i0:i1]
        seg_wt = wt_s[i0:i1]
        good = (seg_wt > 0) & np.isfinite(seg_f) & np.isfinite(seg_s) & (seg_s > 0)
        if good.sum() < 5:
            return float(np.nanmedian(seg_f))  # fallback
        x = seg_f[good]
        med = np.median(x)
        # clip using provided uncertainties if reasonable, else MAD
        sig = seg_s[good]
        # avoid division by zero
        sig = np.where(
            sig > 0, sig, np.nanmedian(sig[sig > 0]) if np.any(sig > 0) else 1.0
        )
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
                hw = float(
                    line.default_half_width if half_width is None else half_width
                )
            else:
                # mask ~ +/- 1.5*FWHM (covers core+wings for most cases)
                hw = 1.5 * float(fwhm_pixels) * float(local_dw)
        else:
            raise ValueError(
                f"Unsupported width_mode={width_mode!r}. Use 'fixed' or 'fwhm_pixels'."
            )

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
            lines_masked.append(line)

    new_wt_s = np.array(wt_s, copy=True)
    new_wt_s[masked] = 0.0

    # Restore original wavelength order
    new_weight = new_wt_s[inv_order]
    mask = masked[inv_order]

    logger.info("Masked %d pixels in %d lines", mask.sum(), len(lines_masked))
    logger.debug("Lines used: %s", ", ".join(line.name for line in lines_masked))
    if return_mask and return_lines_masked:
        return new_weight, mask, lines_masked
    if return_mask:
        return new_weight, mask
    return new_weight


def mask_sky_emission_lines(
    *args, **kwargs
) -> np.ndarray | tuple[np.ndarray, np.ndarray, list[EmissionLine]]:
    """
    Specialized wrapper around mask_strong_emission_lines to target sky lines.

     This is a robust, low-assumption approach:
      1) For each expected line center (rest -> observed using `z`), estimate
         a local continuum with a running median in a window around the line.
      2) Compute line "excess" = flux - continuum and its S/N using uncertainty.
      3) If peak S/N within the line window exceeds `snr_threshold` (and the
         continuum is not completely noise-dominated), mask the line region.
    Parameters
    ----------
    See `mask_strong_emission_lines` for details. The only difference is that if
    `lines` is None, a default set of common sky emission lines (mostly OH) is used.
    Notes
    -----
    - This targets *strong, narrow-ish* features near known lines. It will not
      detect arbitrary lines at unknown wavelengths unless you expand the line list.

    - The default line list is focused on common sky lines in the optical/NIR, but
        you can provide your own list for other regimes.

    """
    if kwargs.get("lines") is None:
        lines = get_default_sky_emission_lines()
        logger.info("Using default sky emission line list with %d lines.", lines.size)

        kwargs["lines"] = lines
    if kwargs.get("redshift") != 0.0:
        logger.warning(
            "Redshift is non-zero (z=%.3f); sky lines will be shifted accordingly. Make sure this is intended.",
            kwargs.get("redshift"),
        )
    return mask_strong_emission_lines(*args, **kwargs)


def estimate_continuum(
    wl, flux, err, weights=None, use_log=True, knot_spacing=100.0, sigma_clip=3.0
):
    """Estimate continuum using a robust spline fit to the log-flux.

    Parameters
    ----------
    wl : ndarray
        Wavelength array (1D).
    flux : ndarray
        Flux array (1D, same length as wl).
    err : ndarray
        Uncertainty array (1D, same length as wl).
    weights : ndarray, optional
        Weights for each pixel (1D, same length as wl). If None, all pixels are equally weighted.
    knot_spacing : float, optional
        Spacing between spline knots in the same units as wl (e.g., Angstrom).
    sigma_clip : float, optional
        Sigma-clipping threshold for outlier rejection in the log-flux residuals.
    use_log : bool, optional
        Whether to fit the log of the flux (True) or the flux itself (False).
        Fitting log-flux is more robust to outliers and multiplicative features,
        but may be less accurate if there are many zero/negative flux values.

    Returns
    -------
    continuum : ndarray
        Estimated continuum flux at each wavelength.
    continuum_err : ndarray
        Estimated uncertainty of the continuum at each wavelength.
    """
    if weights is None:
        weights = np.ones_like(wl)

    knots = np.arange(wl.min() + knot_spacing / 2, wl.max(), knot_spacing)
    knots_idx = np.searchsorted(wl, knots)
    knots_idx = np.insert(knots_idx, 0, 0)
    knots_idx[-1] = len(wl) - 1

    if use_log:
        log_flux = np.where(flux > 0, np.log(flux), np.nan)
        log_flux_err = np.where(flux > 0, err / flux, np.nan)
        weights = np.where((flux > 0) & (err > 0), weights / log_flux_err**2, 0.0)
    else:
        log_flux = flux
        log_flux_err = err
        weights = weights / log_flux_err**2

    # Initial fit using weighted median in each knot region
    knot_flux = np.array(
        [
            np.nanmedian(log_flux[idx[0] : idx[1]][weights[idx[0] : idx[1]] > 0])
            if np.any(weights[idx[0] : idx[1]] > 0)
            else np.nan
            for idx in zip(knots_idx[:-1], knots_idx[1:])
        ]
    )
    knot_flux_nmad = np.array(
        [
            1.48
            * np.nanmedian(
                np.abs(
                    log_flux[idx[0] : idx[1]][weights[idx[0] : idx[1]] > 0]
                    - knot_flux[i]
                )
            )
            if np.any(weights[idx[0] : idx[1]] > 0)
            else np.nan
            for i, idx in enumerate(zip(knots_idx[:-1], knots_idx[1:]))
        ]
    )

    log_flux_model = np.interp(wl, knots, knot_flux)
    log_flux_nmad_model = np.interp(wl, knots, knot_flux_nmad)
    residuals = log_flux - log_flux_model
    mask = np.abs(residuals) > sigma_clip * log_flux_nmad_model
    weights[mask] = 0.0

    knot_flux = np.array(
        [
            np.nansum(log_flux[idx[0] : idx[1]] * weights[idx[0] : idx[1]])
            / np.nansum(weights[idx[0] : idx[1]])
            if np.any(weights[idx[0] : idx[1]] > 0)
            else np.nan
            for idx in zip(knots_idx[:-1], knots_idx[1:])
        ]
    )

    knot_flux_var = np.array(
        [
            np.nansum(
                (log_flux[idx[0] : idx[1]] - knot_flux[i]) ** 2
                * weights[idx[0] : idx[1]]
            )
            / np.nansum(weights[idx[0] : idx[1]])
            if np.any(weights[idx[0] : idx[1]] > 0)
            else np.nan
            for i, idx in enumerate(zip(knots_idx[:-1], knots_idx[1:]))
        ]
    )

    knot_weights = np.array(
        [
            np.nansum(weights[idx[0] : idx[1]])
            for idx in zip(knots_idx[:-1], knots_idx[1:])
        ]
    )

    good_knots = knot_weights > 0
    if good_knots.sum() < 2:
        logger.warning(
            "Not enough good knots (%d) to fit continuum; returning median flux as flat continuum.",
            good_knots.sum(),
        )
        median_flux = np.nanmedian(flux[weights > 0]) if np.any(weights > 0) else 1.0
        return np.full_like(wl, median_flux), np.full_like(wl, median_flux)
    elif good_knots.sum() < 4:
        logger.warning(
            "Only %d good knots to fit continuum; using linear interpolation between knots.",
            good_knots.sum(),
        )

        pol_c = np.polyfit(
            knots[good_knots], knot_flux[good_knots], w=knot_weights[good_knots], deg=1
        )
        pol_continuum = np.poly1d(pol_c)
        continuum = pol_continuum(wl)
        if use_log:
            continuum = np.exp(continuum)
            continuum_var = np.exp(2 * pol_continuum(wl)) * np.interp(
                wl, knots[good_knots], knot_flux_var[good_knots]
            )
        else:
            continuum_var = np.interp(wl, knots[good_knots], knot_flux_var[good_knots])
        return continuum, continuum_var**0.5

    # Sigma-clip residuals and refit
    spline = make_smoothing_spline(
        knots[good_knots], knot_flux[good_knots], w=knot_weights[good_knots]
    )
    if use_log:
        continuum = np.exp(spline(wl))
        continuum_var = np.exp(spline(wl)) ** 2 * np.interp(wl, knots, knot_flux_var)
    else:
        continuum = spline(wl)
        continuum_var = np.interp(wl, knots, knot_flux_var)
    return continuum, continuum_var**0.5


def _gaussian(wl, line_flux, center, sigma):
    return (
        line_flux
        / (sigma * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((wl - center) / sigma) ** 2)
    )


class LineSegmentationMap:
    """Container for detected emission-line segments and fitted line products.

    Parameters
    ----------
    line_segmentation : ndarray
        Integer segmentation map where 0 marks non-line pixels and positive
        values identify detected line regions.
    flux : ndarray
        Observed flux array.
    wavelength : ndarray
        Wavelength array corresponding to ``flux``.
    error : ndarray
        Flux uncertainty array.
    weights : ndarray
        Pixel weights used when fitting line profiles.
    continuum : ndarray
        Estimated continuum flux array.
    continuum_error : ndarray
        Uncertainty of the estimated continuum.

    Attributes
    ----------
    flux_cont_sub : ndarray
        Continuum-subtracted flux.
    flux_cont_sub_err : ndarray
        Uncertainty of the continuum-subtracted flux.
    nlines : int
        Number of labeled line regions in ``line_segmentation``.
    lines : EmissionLineList or None
        Fitted line measurements after calling :meth:`fit_all_lines`.
    """

    def __init__(
        self,
        line_segmentation: np.ndarray,
        flux: np.ndarray,
        wavelength: np.ndarray,
        error: np.ndarray,
        weights: np.ndarray,
        continuum: np.ndarray,
        continuum_error: np.ndarray,
    ):
        self.line_segmentation = line_segmentation
        self.flux = flux
        self.wavelength = wavelength
        self.error = error
        self.weights = weights
        self.continuum = continuum
        self.continuum_error = continuum_error

        self.flux_cont_sub = flux - continuum
        self.flux_cont_sub_err = np.sqrt(error**2 + continuum_error**2)
        self.nlines = int(line_segmentation.max())

        self.lines = None

    def get_line_mask(self, line_id: int) -> np.ndarray:
        """Return a boolean mask for the specified line_id."""
        return self.line_segmentation == line_id

    def _gaussian_fit(self, wl, flux, err, weights, line_id, line_mask):
        mask = (
            line_mask & np.isfinite(flux) & np.isfinite(err) & (err > 0) & (weights > 0)
        )

        n_valid = np.count_nonzero(mask)
        if not n_valid:
            logger.warning(
                "No valid pixels to fit line_id=%s; returning NaN parameters.", line_id
            )
            return dict(
                id=line_id,
                line_flux=0.0,
                line_flux_err=0.0,
                center=np.nan,
                sigma=np.nan,
                npixels=0,
                flag=1,
            )
        elif n_valid < 3:
            logger.warning(
                "Not enough valid pixels to fit line_id=%s; using peak flux and width instead.",
                line_id,
            )
            peak = np.argmax(flux[mask])
            wl_peak = wl[mask][peak]
            sigma = np.clip(wl[mask].ptp() / 2.355, (wl[1] - wl[0]) / 10, None)
            return dict(
                id=line_id,
                line_flux=flux[mask][peak],
                line_flux_err=0.0,
                center=wl_peak,
                sigma=sigma,
                npixels=n_valid,
                flag=2,
            )

        weighted_flux = np.sum(flux[mask] * weights[mask])
        mean = np.median(wl[mask])
        sigma = max(np.std(wl[mask]), (wl[1] - wl[0]) / 10)
        line_flux = np.nanmax(flux[mask])

        try:
            popt, pcov, infodict, messg, ier = curve_fit(
                _gaussian,
                wl[mask],
                flux[mask],
                p0=[line_flux, mean, sigma],
                bounds=(
                    [0, wl[mask].min(), (wl[1] - wl[0]) / 10],
                    [
                        10 * line_flux,
                        max(wl[mask].min() + wl[1] - wl[0], wl[mask].max()),
                        100,
                    ],
                ),
                sigma=err[mask],
                absolute_sigma=True,
                full_output=True,
                ftol=1e-9,
                maxfev=1000,
            )
            line_flux, mean, sigma = popt

        except Exception as e:
            logger.warning(
                "Gaussian fit failed (%s) for line_id=%s; using MLE estimates instead.",
                str(e),
                line_id,
            )
            return dict(
                id=line_id,
                line_flux=line_flux,
                line_flux_err=line_flux,
                center=mean,
                sigma=sigma,
                npixels=n_valid,
                flag=3,
            )

        return dict(
            id=line_id,
            line_flux=line_flux,
            line_flux_err=np.sqrt(np.diag(pcov)[0]),
            center=mean,
            sigma=sigma,
            npixels=n_valid,
            flag=0,
        )

    def fit_line(self, line_id: int) -> dict:
        """Fit a Gaussian to the specified line_id and return fit parameters."""
        mask = self.get_line_mask(line_id)
        return self._gaussian_fit(
            wl=self.wavelength,
            flux=self.flux_cont_sub,
            err=self.flux_cont_sub_err,
            weights=self.weights,
            line_id=line_id,
            line_mask=mask,  # Convert boolean mask to int for fitting
        )

    def fit_all_lines(self) -> Table:
        """Fit all lines in the segmentation map and return a table of results."""
        output_table = Table(
            names=[
                "id",
                "line_flux",
                "center",
                "line_flux_err",
                "sigma",
                "npixels",
                "flag",
            ],
            dtype=[int, float, float, float, float, int, int],
            meta={
                "description": "Fitted parameters for each emission line segment",
                "flags": "0=good fit, 1=no valid pixels, 2=not enough pixels for fit, 3=fit failed, used MLE estimates",
            },
        )
        measured_lines = []
        for line_id in range(1, self.nlines + 1):
            fit_params = self.fit_line(line_id)
            output_table.add_row(fit_params)
            measured_lines.append(
                EmissionLine(
                    name=f"line_{line_id}",
                    rest_wavelength=float(fit_params["center"])
                    if np.isfinite(fit_params["center"])
                    else np.nan,
                    default_half_width=float(fit_params["sigma"])
                    if np.isfinite(fit_params["sigma"])
                    else 10.0,
                    flux=float(fit_params["line_flux"]),
                    flux_error=float(fit_params["line_flux_err"]),
                    flag=int(fit_params["flag"]),
                    metadata={
                        "id": int(fit_params["id"]),
                        "npixels": int(fit_params["npixels"]),
                    },
                )
            )

        self.lines = EmissionLineList(measured_lines)
        # Build emission line spectra
        eline_flux = np.zeros_like(self.wavelength)
        for row in output_table:
            line = _gaussian(
                self.wavelength, row["line_flux"], row["center"], row["sigma"]
            )
            eline_flux += line if np.isfinite(line).all() else 0.0
        self.eline_flux = eline_flux
        return output_table


def _watershed_1d(
    signal: np.ndarray, markers: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """
    1D watershed segmentation for deblending overlapping emission lines.

    Starting from labeled seed regions (markers), flood outward following
    descending signal gradient until regions meet or the mask boundary is reached.

    Parameters
    ----------
    signal : ndarray
        1D array used to guide the flooding (typically SNR).
        Higher values are flooded first.
    markers : ndarray
        Integer label array with seed pixels (>0) already identified.
        0 means unlabeled; will be filled by watershed.
    mask : ndarray of bool
        Only pixels where mask is True are eligible for flooding.

    Returns
    -------
    labels : ndarray of int
        Label array after watershed. Pixels outside the mask remain 0.
    """
    from heapq import heappush, heappop

    labels = np.array(markers, dtype=int, copy=True)
    in_queue = np.zeros(len(signal), dtype=bool)

    # Priority queue: (-signal_value, pixel_index, label)
    # Negative because heapq is a min-heap; we want to process highest signal first
    heap = []

    # Seed the queue with all boundary pixels of each marker region
    for i in range(len(signal)):
        if labels[i] > 0 and mask[i]:
            for di in (-1, 1):
                nb = i + di
                if (
                    0 <= nb < len(signal)
                    and mask[nb]
                    and labels[nb] == 0
                    and not in_queue[nb]
                ):
                    heappush(heap, (-signal[nb], nb, labels[i]))
                    in_queue[nb] = True

    while heap:
        neg_val, idx, lbl = heappop(heap)

        # Skip if already labeled by a previous (higher-priority) flood
        if labels[idx] != 0:
            continue

        labels[idx] = lbl

        # Expand to unlabeled neighbours
        for di in (-1, 1):
            nb = idx + di
            if (
                0 <= nb < len(signal)
                and mask[nb]
                and labels[nb] == 0
                and not in_queue[nb]
            ):
                heappush(heap, (-signal[nb], nb, lbl))
                in_queue[nb] = True

    return labels


@_parse_lines_param_decorator
def find_emission_lines(
    wl,
    flux,
    err,
    weights=None,
    continuum=None,
    continuum_error=None,
    lines=None,
    redshift=0.0,
    to_rest_frame=False,
    snr_threshold=3.0,
    min_continuum_snr=1.0,
    cont_sigma_clip=1.5,
    knot_spacing=100.0,
    min_npixels: int = 3,
    pad_detection_pix: int = 10,
    deblend_lines: bool = True,
):
    """Find and fit emission lines in a spectrum.

    Parameters
    ----------
    wl : ndarray
        Wavelength array (1D).
    flux : ndarray
        Flux array (1D, same length as wl).
    err : ndarray
        Uncertainty array (1D, same length as wl).
    weights : ndarray, optional
        Weights for each pixel (1D, same length as wl). If None, all pixels are equally weighted.
    continuum : ndarray, optional
        Pre-computed continuum flux at each wavelength. If None, it will be estimated.
    continuum_error : ndarray, optional
        Uncertainty of the continuum at each wavelength. If None, it will be estimated.
    lines : sequence of EmissionLine, optional
        Line list. If None, uses DEFAULT_EMISSION_LINES_A (Angstrom).
    snr_threshold : float, optional
        Minimum peak (flux-continuum)/sigma to consider a line detected.
    min_continuum_snr : float, optional
        Require median(abs(continuum)/sigma) in the continuum window to be at least
        this value to consider a line detection valid; helps avoid false positives when everything is noise.
    cont_sigma_clip : float, optional
        Sigma-clipping threshold for outlier rejection in the continuum estimation.
    knot_spacing : float, optional
        Spacing between spline knots for continuum estimation in the same units as wl (e.g., Angstrom).
    min_npixels : int, optional
        Minimum number of contiguous pixels above the SNR threshold to consider a valid line detection.
    pad_detection_pix : int, optional
        Number of pixels to pad on either side of detected line regions to ensure the full line is captured, especially for broad lines.
    deblend_lines : bool, optional
        If True, apply 1D watershed deblending to separate overlapping line detections.
        Each local SNR maximum seeds its own region and regions are grown by flooding
        in descending SNR order until they meet. Default is True.
    Returns
    -------
    output_table : astropy.table.Table
        Table with columns: line_flux, line_flux_err, center, sigma, npixels, flag.
    continuum : ndarray
        Estimated continuum flux at each wavelength.
    continuum_error : ndarray
        Estimated uncertainty of the continuum at each wavelength.
    """
    if weights is None:
        weights = np.ones_like(wl)

    # Estimate continuum
    if continuum is None and continuum_error is None:
        continuum, continuum_error = estimate_continuum(
            wl,
            flux,
            err,
            weights,
            knot_spacing=knot_spacing,
            sigma_clip=cont_sigma_clip,
        )

    if lines is not None:
        logger.info("Using provided line list with %d lines.", lines.size)
        lines_mask = lines.to_mask(wl, redshift=redshift).astype(float)
    else:
        logger.info("No line list provided; using all pixels for detection.")
        lines_mask = np.ones_like(wl)

    # Continuum-free flux and SNR
    clean_flux = flux - continuum
    clean_flux *= lines_mask
    clean_flux_err = np.sqrt(err**2 + continuum_error**2)
    clean_snr = np.where(clean_flux_err > 0, clean_flux / clean_flux_err, 0.0)
    # Mask regions where continuum is not well constrained
    cont_snr = np.where(continuum_error > 0, continuum / continuum_error, 0.0)
    clean_snr[cont_snr < min_continuum_snr] = 0.0

    bright_pixels = clean_snr > snr_threshold
    line_ids, nlines = label(bright_pixels)
    slices = find_objects(line_ids)  # just to log the line regions
    logger.info(
        "Initial detection found %d candidate lines with SNR > %.1f",
        nlines,
        snr_threshold,
    )

    # Deblend lines
    if deblend_lines:
        # Find local SNR maxima within each labeled region to use as seeds
        peak_markers = np.zeros_like(line_ids)
        next_marker_id = 1
        for seg_id in range(1, nlines + 1):
            seg_indices = np.where(line_ids == seg_id)[0]
            if len(seg_indices) == 0:
                continue
            seg_snr = clean_snr[seg_indices]
            # Find local peaks within this segment
            local_peaks, _ = find_peaks(seg_snr)
            if len(local_peaks) == 0:
                # No local peak found; use the global maximum as the single seed
                local_peaks = [int(np.argmax(seg_snr))]
            for peak_pix in local_peaks:
                global_idx = seg_indices[peak_pix]
                peak_markers[global_idx] = next_marker_id
                next_marker_id += 1

        n_seeds = next_marker_id - 1
        logger.info(
            "Watershed deblending: found %d peak seeds across %d initial regions.",
            n_seeds,
            nlines,
        )

        if n_seeds > nlines:
            # Only run watershed if we actually found sub-peaks worth splitting
            line_ids = _watershed_1d(clean_snr, peak_markers, bright_pixels)
            nlines = np.unique(line_ids[line_ids > 0]).size
            slices = find_objects(line_ids)
            logger.info("After watershed deblending, found %d candidate lines.", nlines)
        else:
            logger.info("No additional peaks found; skipping watershed split.")

    # Get number of pixels in each line and filter by min_npixels
    line_pixel_counts = np.array([s[0].stop - s[0].start for s in slices])
    valid_line_ids = np.where(line_pixel_counts >= min_npixels)[0] + 1
    line_ids = np.where(np.isin(line_ids, valid_line_ids), line_ids, 0)
    nlines = len(valid_line_ids)
    logger.info(
        "%d candidate lines have at least %d pixels above the SNR threshold.",
        nlines,
        min_npixels,
    )
    # re-map line ids to 1..nlines
    unique_ids = np.unique(line_ids)
    new_id_map = {
        old_id: new_id
        for new_id, old_id in enumerate(unique_ids[unique_ids > 0], start=1)
    }
    line_ids = np.array(
        np.where(line_ids > 0, np.vectorize(new_id_map.get)(line_ids), 0), dtype=int
    )

    if pad_detection_pix > 0:
        slices = find_objects(line_ids)
        for i, s in enumerate(slices, start=1):
            if s is None:
                continue
            # avoid overlap with neighboring lines by only padding within the current line segment
            if i > 1 and s[0].start - pad_detection_pix < slices[i - 2][0].stop:
                dist_to_prev = s[0].start - slices[i - 2][0].stop
                pad_left = min(pad_detection_pix, dist_to_prev)
            else:
                pad_left = pad_detection_pix

            if i < len(slices) and s[0].stop + pad_detection_pix > slices[i][0].start:
                dist_to_next = slices[i][0].start - s[0].stop
                pad_right = min(pad_detection_pix, dist_to_next)
            else:
                pad_right = pad_detection_pix

            start = max(0, s[0].start - pad_left)
            stop = min(len(wl), s[0].stop + pad_right)
            line_ids[start:stop] = i

    logger.info(
        "Found %d candidate emission lines with SNR > %.1f", nlines, snr_threshold
    )

    line_segm_map = LineSegmentationMap(
        line_segmentation=line_ids,
        flux=flux,
        wavelength=wl,
        error=err,
        weights=weights,
        continuum=continuum,
        continuum_error=continuum_error,
    )
    output_table = line_segm_map.fit_all_lines()

    if lines is not None:
        # Match detected lines to input line list based on proximity of centers
        matched_line_names = []
        for row in output_table:
            line_center = row["center"]
            if not np.isfinite(line_center):
                matched_line_names.append("unknown")
                continue
            matched_line = lines.get_closest_line(line_center, redshift=redshift)
            tolerance = (
                max(3 * row["sigma"], matched_line.default_half_width)
                if matched_line is not None
                else 0.0
            )
            if (
                matched_line is not None
                and np.abs(matched_line.rest_wavelength * (1 + redshift) - line_center)
                < tolerance
            ):
                matched_line_names.append(matched_line.name)
            else:
                matched_line_names.append("unknown")
        output_table["line_name"] = matched_line_names
        if line_segm_map.lines is not None:
            line_segm_map.lines = line_segm_map.lines.with_names(matched_line_names)

    if to_rest_frame:
        output_table["center"] = output_table["center"] / (1 + redshift)
        output_table["sigma"] = output_table["sigma"] / (1 + redshift)
        if line_segm_map.lines is not None:
            line_segm_map.lines = EmissionLineList(
                [
                    replace(
                        line,
                        rest_wavelength=line.rest_wavelength / (1 + redshift),
                        default_half_width=line.default_half_width / (1 + redshift),
                    )
                    for line in line_segm_map.lines
                ]
            )
    return output_table, line_segm_map
