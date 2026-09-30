"""
Post-processing utilities for BESTA results.

This module provides:
- I/O for CosmoSIS-like tabular results.
- Weighted posterior summaries (MAP, mean, covariance, correlation).
- Weighted quantiles and (optionally multi-modal) HDI intervals.
- 1D PDFs (histogram-derived + optional KDE).
- 2D PDFs (KDE fallback to histogram) + HPD enclosed-fraction maps.
- A structured ResultsSummary object with FITS + JSON export helpers.

Notes
-----
- This module assumes the posterior samples are stored in an Astropy Table
  with a log-posterior column (default: "post") and parameter columns
  named like "section--name" (default delimiter/prefix: "--").
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import find_peaks

from astropy.io import fits
from astropy.table import Table, Column
from astropy import units as u

from cosmosis.postprocessing import run_cosmosis_postprocess

from besta import io
from besta.logging import get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Weighted statistics
# -----------------------------------------------------------------------------

def _as_float_array(x) -> np.ndarray:
    """Conversion to float ndarray (handles Astropy Column/Quantity)."""
    if isinstance(x, (u.Quantity, Column)):
        return x.value
    else:
        return np.asarray(x, dtype=float)

def _logsumexp_weighted(a: np.ndarray, w: np.ndarray) -> float:
    a = _as_float_array(a).ravel()
    w = _as_float_array(w).ravel()
    m = np.nanmax(a)
    return float(m + np.log(np.nansum(w * np.exp(a - m))))

def normalize_weights(weights: np.ndarray, *, allow_all_zero: bool = False) -> np.ndarray:
    """Normalize weights to sum=1 over finite entries."""
    w = _as_float_array(weights)
    w = np.where(np.isfinite(w), w, 0.0)
    s = np.sum(w)
    if s <= 0:
        if allow_all_zero:
            return w
        raise ValueError("Sum of weights must be > 0.")
    return w / s

def weighted_mean(x: np.ndarray, weights: np.ndarray, axis: int = -1) -> np.ndarray:
    """Weighted mean along `axis`."""
    x = _as_float_array(x)
    w = normalize_weights(weights)
    # Expand w to broadcast along x dimensions
    shape = [1] * x.ndim
    shape[axis] = -1
    w_ = w.reshape(shape)
    return np.sum(x * w_, axis=axis)

def weighted_covariance(
    x: np.ndarray,
    weights: np.ndarray,
    *,
    unbiased: bool = False,
) -> np.ndarray:
    """
    Weighted covariance for x with shape (D, N) and weights with shape (N,).

    If `unbiased=True`, applies the common correction for normalized weights:
        cov /= (1 - sum(w^2))
    """
    x = _as_float_array(x)
    if x.ndim != 2:
        raise ValueError("x must have shape (D, N).")
    w = normalize_weights(weights)  # sum=1

    mu = np.sum(x * w[None, :], axis=1)          # (D,)
    xm = x - mu[:, None]                         # (D, N)
    cov = np.sum(w[None, None, :] * xm[:, None, :] * xm[None, :, :], axis=2)  # (D, D)

    if unbiased:
        denom = 1.0 - np.sum(w**2)
        if denom <= 0:
            raise ValueError("Unbiased covariance undefined for degenerate weights.")
        cov /= denom
    return cov

def covariance_to_correlation(cov: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to correlation matrix."""
    cov = _as_float_array(cov)
    d = np.sqrt(np.clip(np.diag(cov), 0.0, np.inf))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / (d[:, None] * d[None, :])
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr

def weighted_quantile(x: np.ndarray, weights: np.ndarray, q: Sequence[float]) -> np.ndarray:
    """
    Weighted quantiles for 1D sample x with weights.

    Parameters
    ----------
    x : array-like, shape (N,)
    weights : array-like, shape (N,)
    q : sequence of quantiles in [0, 1]

    Returns
    -------
    quantiles : ndarray shape (len(q),)
    """
    x = _as_float_array(x).ravel()
    w = _as_float_array(weights).ravel()
    mask = np.isfinite(x) & np.isfinite(w)
    x = x[mask]
    w = w[mask]
    if x.size == 0:
        return np.full(len(q), np.nan, dtype=float)

    w = normalize_weights(w)
    idx = np.argsort(x)
    xs = x[idx]
    ws = w[idx]
    cdf = np.cumsum(ws)
    # Ensure cdf spans [0,1]
    cdf /= cdf[-1]
    return np.interp(np.asarray(q, dtype=float), cdf, xs)

def weighted_hdi(
    x: np.ndarray,
    weights: np.ndarray,
    *,
    mass: float = 0.68,
    max_intervals: int = 2,
    merge_tol: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Compute weighted highest-density interval(s) from 1D samples.

    The output approximates the smallest region containing ``mass`` probability,
    allowing multi-modality via disjoint intervals.

    Method
    ------
    1. Sort samples by ``x``.
    2. Use cumulative weighted mass.
    3. Find the narrowest interval(s) with enclosed mass >= ``mass``.
    4. Optionally repeat to recover additional disjoint intervals.

    Parameters
    ----------
    x, weights : arrays of shape (N,)
    mass : target probability mass in (0,1]
    max_intervals : maximum number of disjoint intervals to return (approximate)
    merge_tol : merge intervals whose gap is <= merge_tol (in x units)

    Returns
    -------
    intervals : list of (low, high)
    """
    x = _as_float_array(x).ravel()
    w = _as_float_array(weights).ravel()
    mask = np.isfinite(x) & np.isfinite(w)
    x = x[mask]
    w = w[mask]
    if x.size == 0:
        return [(np.nan, np.nan)]
    if not (0 < mass <= 1):
        raise ValueError("mass must be in (0, 1].")

    # Normalize and sort
    w = normalize_weights(w)
    idx = np.argsort(x)
    xs = x[idx]
    ws = w[idx]
    cdf = np.cumsum(ws)
    cdf[-1] = 1.0

    intervals: List[Tuple[float, float]] = []
    used = np.zeros(xs.size, dtype=bool)

    def _find_best_interval(available_mask: np.ndarray) -> Optional[Tuple[int, int]]:

        best = None
        best_width = np.inf
        # Identify contiguous runs
        avail = available_mask.astype(int)
        # runs: start indices where diff==1, end where diff==-1
        starts = np.where(np.diff(np.r_[0, avail]) == 1)[0]
        ends = np.where(np.diff(np.r_[avail, 0]) == -1)[0]
        for s, e in zip(starts, ends):
            sub_ws = ws[s:e]
            if sub_ws.size == 0:
                continue
            sub_cdf = np.cumsum(sub_ws)
            sub_cdf[-1] = np.sum(sub_ws)
            if sub_cdf[-1] < mass:
                continue
            sub_cdf0 = np.concatenate([[0.0], sub_cdf])
            i = 0
            for j in range(1, sub_cdf0.size):
                while (sub_cdf0[j] - sub_cdf0[i]) >= mass and i < j:
                    width = xs[s + j - 1] - xs[s + i]
                    if width < best_width:
                        best_width = width
                        best = (s + i, s + j - 1)
                    i += 1
        return best

    for _ in range(max_intervals):
        available = ~used
        best = _find_best_interval(available)
        if best is None:
            break
        i, j = best
        intervals.append((float(xs[i]), float(xs[j])))
        used[i : j + 1] = True

        # If the first interval already covers the full mass approximately, stop.
        m = np.sum(ws[(xs >= xs[i]) & (xs <= xs[j])])
        if m >= mass:
            break

    # Merge close intervals if requested
    if merge_tol > 0 and len(intervals) > 1:
        intervals = sorted(intervals, key=lambda t: t[0])
        merged: List[Tuple[float, float]] = [intervals[0]]
        for lo, hi in intervals[1:]:
            plo, phi = merged[-1]
            if lo - phi <= merge_tol:
                merged[-1] = (plo, max(phi, hi))
            else:
                merged.append((lo, hi))
        intervals = merged

    if len(intervals) == 0:
        # Fallback to full range
        intervals = [(float(xs[0]), float(xs[-1]))]
    return intervals

# -----------------------------------------------------------------------------
# PDFs and HPD maps
# -----------------------------------------------------------------------------

def enclosed_fraction_map(
    density: np.ndarray,
    *,
    xedges: Optional[np.ndarray] = None,
    yedges: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute an HPD-style enclosed fraction map from a 2D density grid.

    Returns an array ``F`` with the same shape as ``density`` where each pixel
    stores the cumulative enclosed mass at that density threshold (after sorting
    pixels by density in descending order).

    If ``xedges`` and ``yedges`` are provided, pixel areas are included when
    computing mass; otherwise constant pixel area is assumed.
    """
    dens = _as_float_array(density)
    if dens.ndim != 2:
        raise ValueError("density must be 2D.")

    if xedges is not None and yedges is not None:
        dx = np.diff(_as_float_array(xedges))  # along x/columns
        dy = np.diff(_as_float_array(yedges))  # along y/rows
        if dens.shape != (dy.size, dx.size):
            raise ValueError(
                f"density shape {dens.shape} does not match edges "
                f"(ny,nx)=({dy.size},{dx.size})."
            )
        area = dy[:, None] * dx[None, :]
        mass = dens * area
    else:
        mass = dens

    flat_d = dens.ravel()
    flat_m = mass.ravel()

    # Mask non-finite and negative masses/densities
    good = np.isfinite(flat_d) & np.isfinite(flat_m) & (flat_m >= 0)
    out = np.full_like(flat_d, np.nan, dtype=float)
    if not np.any(good):
        return out.reshape(dens.shape)

    flat_dg = flat_d[good]
    flat_mg = flat_m[good]

    order = np.argsort(flat_dg)[::-1]  # descending density
    cum = np.cumsum(flat_mg[order])
    if cum[-1] <= 0:
        return out.reshape(dens.shape)
    cum /= cum[-1]

    # Place back into original flat array
    tmp = np.empty_like(cum)
    tmp[order] = cum
    out[good] = tmp
    return out.reshape(dens.shape)

def histogram_pdf_1d(
    x: np.ndarray,
    weights: np.ndarray,
    *,
    bins: int = 100,
    range: Optional[Tuple[float, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    1D PDF from weighted histogram.

    Returns
    -------
    centers : (bins,)
    pdf : (bins,), normalized to integrate to 1 over x
    """
    x = _as_float_array(x).ravel()
    w = _as_float_array(weights).ravel()
    mask = np.isfinite(x) & np.isfinite(w)
    x = x[mask]
    w = w[mask]
    if x.size == 0:
        centers = np.linspace(0, 1, bins)
        return centers, np.full_like(centers, np.nan)

    # Normalize weights but histogram density will handle scaling; still OK.
    w = normalize_weights(w)
    hist, edges = np.histogram(x, bins=bins, range=range, weights=w, density=False)
    dx = np.diff(edges)
    # Convert probability per bin -> density
    with np.errstate(divide="ignore", invalid="ignore"):
        pdf = hist / dx
    # Ensure integrates to 1 (numerical)
    integral = np.nansum(pdf * dx)
    if integral > 0:
        pdf /= integral
    return edges, pdf

def kde_pdf_1d(
    x: np.ndarray,
    weights: np.ndarray,
    bins: np.ndarray,
) -> np.ndarray:
    """
    KDE PDF on a provided grid; returns NaNs on failure.
    """
    x = _as_float_array(x).ravel()
    w = _as_float_array(weights).ravel()
    edges = _as_float_array(bins).ravel()
    g = 0.5 * (edges[:-1] + edges[1:])
    mask = np.isfinite(x) & np.isfinite(w)
    x = x[mask]
    w = w[mask]
    if x.size == 0:
        return np.full_like(g, np.nan)

    try:
        w = normalize_weights(w)
        kde = stats.gaussian_kde(x, weights=w)
        return kde(g)
    except Exception:
        return np.full_like(g, np.nan)

def check_multimodal_pdf(x, f, *, prominence=None, relative_prominence=0.01):
    """Check if a 1D PDF is multimodal by counting local maxima.
    
    Parameters
    ----------
    x : array-like, shape (N,)
        Grid points.
    f : array-like, shape (N,)
        PDF values at the grid points.
    
    Returns
    -------
    n_maxima : int
        Number of local maxima found in the PDF.
    maxima_x : ndarray
        x-values of the local maxima.
    """
    x = _as_float_array(x).ravel()
    f = _as_float_array(f).ravel()
    if x.size != f.size:
        raise ValueError("`x` and `f` must have the same length.")

    if x.size < 3:
        return 0, np.empty(0), np.empty(0)

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(f)):
        raise ValueError("`x` and `f` must contain only finite values.")

    if prominence is None:
        f_range = np.ptp(f)
        prominence = relative_prominence * f_range

    indices, properties = find_peaks(f, prominence=prominence)
    if indices.size == 0:
        # Check if the PDF is flat or has a single peak at the edge
        if np.allclose(f, f[0]):
            return 0, np.empty(0), np.empty(0)
        indices = np.array([np.argmax(f)])

    maxima_x = x[indices]
    maxima_val = f[indices]

    order = np.argsort(maxima_val)[::-1]

    return (
        int(indices.size),
        maxima_x[order],
        maxima_val[order],
    )

def kde_or_hist_pdf_2d(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    bins: int = 80,
    range: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
    use_kde: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    2D PDF on a regular grid. Returns (x_centers, y_centers, pdf[y,x]).

    Uses KDE if possible (and use_kde), otherwise falls back to weighted histogram2d.
    """
    x = _as_float_array(x).ravel()
    y = _as_float_array(y).ravel()
    w = _as_float_array(weights).ravel()
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x = x[mask]
    y = y[mask]
    w = w[mask]
    if x.size == 0:
        xc = np.linspace(0, 1, bins)
        yc = np.linspace(0, 1, bins)
        return xc, yc, np.full((bins, bins), np.nan)

    w = normalize_weights(w)

    if range is None:
        xr = (np.min(x), np.max(x))
        yr = (np.min(y), np.max(y))
    else:
        xr, yr = range

    xedges = np.linspace(xr[0], xr[1], bins + 1)
    yedges = np.linspace(yr[0], yr[1], bins + 1)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])

    if use_kde:
        try:
            # KDE evaluated on grid
            X, Y = np.meshgrid(xc, yc, indexing="xy")  # shapes (bins,bins)
            kde = stats.gaussian_kde(np.vstack([x, y]), weights=w)
            Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape((bins, bins))
            # Normalize numerically to ensure integral ~ 1
            dx = np.diff(xedges)[0]
            dy = np.diff(yedges)[0]
            integral = np.sum(Z) * dx * dy
            if integral > 0:
                Z /= integral
            return xc, yc, Z
        except Exception:
            pass

    # Histogram fallback: numpy returns shape (bins_x, bins_y) by default; we want (bins_y, bins_x)
    H, xe, ye = np.histogram2d(x, y, bins=[xedges, yedges], weights=w, density=False)
    # Convert probability per bin -> density
    dx = np.diff(xe)[0]
    dy = np.diff(ye)[0]
    Z = H / (dx * dy)
    # Transpose to (y,x) for meshgrid(indexing="xy") conventions
    Z = Z.T
    # Normalize numerically
    integral = np.sum(Z) * dx * dy
    if integral > 0:
        Z /= integral
    return xc, yc, Z

def pit_from_pdf(x_edges, pdf, x_true):
    """
    Compute PIT value for a true value given a PDF defined by edges and values.

    Parameters
    ----------
    x_edges : array shape (N+1,) bin edges
    pdf : array shape (N,) PDF values for each bin, normalized to integrate to 1
    x_true : scalar true value

    Returns
    -------
    pit : scalar in [0,1] representing the cumulative probability up to ``x_true``
    """
    x_edges = _as_float_array(x_edges).ravel()
    pdf = _as_float_array(pdf).ravel()
    if x_edges.size != pdf.size + 1:
        raise ValueError("x_edges must have one more element than pdf.")
    if not np.isfinite(x_true):
        raise ValueError("x_true must be finite.")
    dx = np.diff(x_edges)
    cdf = np.cumsum(pdf * dx)
    if not np.isclose(cdf[-1], 1.0):
        raise ValueError("pdf must be normalized to integrate to 1.")

    return np.interp(x_true, x_edges, np.r_[0.0, cdf])

def pdf_stats(edges: np.ndarray, pdf: np.ndarray,
              quantiles=None,
              find_multimodal: bool = False) -> dict:
    """
    Compute summary stats from a discrete pdf over bin centers.

    Parameters
    ----------
    edges : ndarray, shape (K,)
        Bin edges.
    pdf : ndarray, shape (K,)
        PDF defined by the edges.
    quantiles : tuple of float, optional
        Quantiles to report (default 16, 50, 84 percent).
    find_multimodal : bool, optional
        If True, return rough modes by local-maximum search.

    Returns
    -------
    stats : dict
        Keys: mean, std, map, q, lo68, hi68, modes (optional).
    """
    # Ensure normalization
    norm = np.sum(pdf * np.diff(edges))
    pdf /= norm if norm > 0 else 1.0
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Trapecium integrals
    mean = np.sum(pdf * centers * np.diff(edges))
    var = np.sum(pdf * (centers - mean)**2 * np.diff(edges))
    std = var ** 0.5
    k_map = np.argmax(pdf * np.diff(edges))
    v_map = centers[k_map]

    cdf = np.cumsum(pdf * np.diff(edges))
    cdf = np.insert(cdf, 0, 0)

    if quantiles is not None:
        qs = np.array(quantiles, float)
        qvals = np.interp(qs, cdf, edges, left=edges[0], right=edges[-1])
    else:
        qvals = None

    median = np.interp(0.5, cdf, edges, left=edges[0], right=edges[-1])
    lo68 = np.interp(0.16, cdf, edges, left=edges[0], right=edges[-1])
    hi68 = np.interp(0.84, cdf, edges, left=edges[0], right=edges[-1])

    out = {"mean": mean, "std": std, "map": v_map, "q": qvals,
           "median": median, "lo68": lo68, "hi68": hi68}

    if find_multimodal:
        modes = []
        for i in range(1, len(pdf) - 1):
            if pdf[i] > pdf[i - 1] and pdf[i] > pdf[i + 1]:
                modes.append(centers[i])
        if not modes:
            modes = [v_map]
        out["modes"] = np.asarray(modes)
    return out

# -----------------------------------------------------------------------------
# Autocorrelation
# -----------------------------------------------------------------------------

def autocorrelation_1d(x):
    """
    Estimate the normalized autocorrelation function of a 1D series using FFT.

    Parameters
    ----------
    x : array_like, shape (n,)
        Input time series.

    Returns
    -------
    acf : ndarray, shape (n,)
        Normalized autocorrelation function, with acf[0] = 1.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)

    x = x - np.mean(x)

    # Zero-pad to 2*n for efficient non-circular correlation
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real

    # Normalize by number of overlapping pairs
    acf /= np.arange(n, 0, -1)

    # Normalize so that acf[0] = 1
    acf /= acf[0]

    return acf

def integrated_autocorrelation_time(chain, c=5.0, tol=30):
    """
    Estimate the integrated autocorrelation time of an MCMC chain.

    Parameters
    ----------
    chain : ndarray, shape (nsamples, nwalkers)
        MCMC samples for one parameter.
    c : float, optional
        Windowing parameter. Larger values give more conservative estimates.
        Default is 5.0.
    tol : float, optional
        Minimum recommended ratio nsamples / tau. If nsamples < tol * tau,
        the estimate is considered unreliable. Default is 50.

    Returns
    -------
    tau : float
        Estimated integrated autocorrelation time.
    acf_mean : ndarray
        Mean autocorrelation function averaged over walkers.
    reliable : bool
        Whether the chain is long enough according to nsamples > tol * tau.
    """
    chain = np.asarray(chain, dtype=float)

    if chain.ndim != 2:
        raise ValueError("Expected chain with shape (nsamples, nwalkers).")

    nsamples, nwalkers = chain.shape

    # Autocorrelation for each walker
    acfs = np.array([autocorrelation_1d(chain[:, i]) for i in range(nwalkers)])

    # Average over walkers
    acf_mean = np.mean(acfs, axis=0)

    # Cumulative estimate:
    # tau(t) = 1 + 2 * sum_{lag=1}^{t} rho(lag)
    taus = 1.0 + 2.0 * np.cumsum(acf_mean[1:])

    # Windowing criterion: stop when lag > c * tau(lag)
    lags = np.arange(1, len(taus) + 1)
    mask = lags < c * taus

    if np.any(~mask):
        window = np.argmax(~mask)
    else:
        window = len(taus) - 1

    tau = taus[window]

    reliable = nsamples > tol * tau

    return tau, acf_mean, reliable


def auto_burning_results(chains, c=5.0, tol=30, kappa_act=3.0):
    """
    Estimate the burning-in period for each parameter in the MCMC chains.

    Parameters
    ----------
    chains : ndarray, shape (nsamples, nwalkers, nparams)
        Array of MCMC chains for each parameter.
    c : float, optional
        Windowing parameter. Larger values give more conservative estimates.
        Default is 5.0.
    tol : float, optional
        Minimum recommended ratio nsamples / tau. If nsamples < tol * tau,
        the estimate is considered unreliable. Default is 50.
    kappa_act : float, optional
        Safety factor for the burning-in period. Default is 3.0.

    Returns
    -------
    max_burn : int
        Maximum estimated burning-in period across all parameters.
    """
    burn_reliable = []
    burn_unreliable = []

    # Loop over parameters and convert ACT to burn-in with a safety factor.
    nparams = chains.shape[2]
    for ith in range(nparams):
        tau, _, reliable = integrated_autocorrelation_time(
            chains[:, :, ith], c=c, tol=tol)

        if not np.isfinite(tau) or tau <= 0:
            logger.warning("Invalid ACT estimate for parameter index %d: %s", ith, tau)
            continue

        burn_i = int(np.ceil(kappa_act * tau))
        if reliable:
            burn_reliable.append(burn_i)
        else:
            burn_unreliable.append(burn_i)

    if burn_reliable:
        return int(np.max(burn_reliable))

    if burn_unreliable:
        logger.warning("No reliable autocorrelation time estimates found.")
        return int(np.max(burn_unreliable))

    logger.warning("No finite autocorrelation time estimates found. Using burn-in=0.")
    return 0

# -----------------------------------------------------------------------------
# I/O / manipulation helpers
# -----------------------------------------------------------------------------
def flat_chain_to_walkers(data, nwalkers, nsamples):
    """
    Reshape a flattened MCMC chain into a 3D array with shape (nsamples, nwalkers, -1).

    Parameters
    ----------
    data : ndarray, shape (nrows, ...)
        Flattened MCMC chain.
    nwalkers : int
        Number of walkers.
    nsamples : int
        Number of samples.

    Returns
    -------
    chain : ndarray, shape (nsamples, nwalkers, -1)
        Reshaped MCMC chain.
    """
    nrows = data.shape[0]
    expected = nwalkers * nsamples
    if nrows != expected:
        raise ValueError(
            f"Cannot reshape chain with {nrows} rows into (nsteps={nsamples}, nwalkers={nwalkers})."
        )
    return data.reshape((nsamples, nwalkers, -1))

def effective_sample_size(weights: np.ndarray) -> float:
    """Return the standard importance-sampling effective sample size."""
    w = normalize_weights(weights)
    return float(1.0 / np.sum(w**2))

def laplace_logz(loglike_map: float, logprior_map: float, cov: np.ndarray) -> EvidenceEstimate:
    """Estimate log-evidence with a Laplace approximation around the MAP point."""

    cov = _as_float_array(cov)
    d = cov.shape[0]
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0 or not np.isfinite(logdet):
        return EvidenceEstimate(
            method="laplace",
            logz=float("nan"),
            details={"reason": "covariance not positive definite", "sign": float(sign), "logdet": float(logdet)},
        )
    logz = loglike_map + logprior_map + 0.5 * d * np.log(2.0 * np.pi) + 0.5 * logdet
    return EvidenceEstimate(
        method="laplace",
        logz=float(logz),
        details={"d": int(d), "logdet_cov": float(logdet)},
    )

def harmonic_mean_logz(loglike: np.ndarray, weights: np.ndarray, *, trim_frac: float = 0.01) -> EvidenceEstimate:
    """
    Robust harmonic-mean estimator.

    Uses ``log Z = -log(E_posterior[exp(-logL)])``.

    This estimator is generally unstable; trimming helps, but it should still
    be treated as a sanity check only.
    """
    ll = _as_float_array(loglike).ravel()
    w = normalize_weights(weights)

    mask = np.isfinite(ll) & np.isfinite(w)
    ll = ll[mask]
    w = w[mask]
    if ll.size == 0:
        return EvidenceEstimate(method="hme_robust", logz=float("nan"), details={"reason": "no finite loglike"})

    # Trim the top tail in loglike to reduce variance
    if trim_frac > 0:
        cutoff = np.quantile(ll, 1.0 - trim_frac)
        keep = ll <= cutoff
        ll = ll[keep]
        w = normalize_weights(w[keep])

    log_term = _logsumexp_weighted(-ll, w)
    logz = -log_term
    return EvidenceEstimate(
        method="hme_robust",
        logz=float(logz),
        details={"n": int(ll.size), "ess": effective_sample_size(w), "trim_frac": float(trim_frac)},
    )

@dataclass
class EvidenceEstimate:
    """Container for a scalar evidence estimate and method-specific metadata."""

    method: str
    logz: float
    logz_err: Optional[float] = None
    details: Dict[str, Any] = None


#TODO
#@dataclass
#class Chain:
#    """TODO"""
#
#    flat_samples: np.ndarray
#    posterior: np.ndarray
#    parameters : List[str]
#    walkers: int = 1
#    samples: int = None
#
#    def __post_init__(self):

    
# -----------------------------------------------------------------------------
# ResultsSummary dataclass
# -----------------------------------------------------------------------------

@dataclass
class ResultsSummary:
    """
    Container for posterior summary products.

    Attributes
    ----------
    parameter_keys : list of full parameter keys (e.g. "sfh--tau")
    parameter_names : list of short names (e.g. "tau")
    parameter_sections : list of sections (e.g. "sfh")
    posterior_key : log-posterior column name used
    map_index : index of MAP (maximum log-posterior) sample among the *filtered* samples
    n_samples : number of samples used after filtering
    weights : normalized posterior weights, shape (n_samples,)
    logpost : log posterior values, shape (n_samples,)
    samples : array shape (n_params, n_samples)
    map : vector shape (n_params,)
    mean : vector shape (n_params,)
    covariance : matrix shape (n_params, n_params)
    correlation : matrix shape (n_params, n_params)
    percentiles : list of quantiles in [0,1]
    percentiles_values : array shape (n_params, n_percentiles)
    percentiles_logpost : array shape (n_params, n_percentiles)
    hdi_intervals_68 : dict short_name -> list of (low, high) for 68% mass
    hdi_intervals_95 : dict short_name -> list of (low, high) for 95% mass
    map_1d : dict short_name -> array of 1D mode locations from PDF peaks
    pdf_1d : dict short_name -> dict with keys: grid, hist_pdf, kde_pdf
    pdf_2d : dict (short_i, short_j) -> dict with keys: xgrid, ygrid, pdf, enclosed_fraction
    extra_info : arbitrary metadata to include in exports
    """
    parameter_keys: List[str]
    posterior_key: str = "post"

    parameter_sections: List[str] = field(default_factory=list)
    parameter_names: List[str] = field(default_factory=list)

    n_samples: int = 0
    map_index: int = -1

    weights: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    logpost: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    samples: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))

    map: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    mean: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    covariance: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    correlation: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))

    percentiles: List[float] = field(default_factory=list)
    percentiles_values: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    percentiles_logpost: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=float))

    hdi_intervals_68: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    hdi_intervals_95: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    map_1d: Dict[str, np.ndarray] = field(default_factory=dict)

    evidence: Optional[EvidenceEstimate] = None

    pdf_1d: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)
    pdf_2d: Dict[Tuple[str, str], Dict[str, np.ndarray]] = field(default_factory=dict)

    extra_info: Dict[str, Any] = field(default_factory=dict)

    def estimate_evidence_from_table(
        self,
        table: Table,
        *,
        logpost_key: Optional[str] = None,
        logprior_key: str = "prior",
        loglike_key: Optional[str] = None,
        method: str = "laplace",
        hme_trim_frac: float = 0.01,
        parameter_prefix: str = "--",
    ) -> EvidenceEstimate:
        """
        Estimate evidence logZ using information in the original results table.

        Supported methods are ``"laplace"`` (MAP loglike/logprior + covariance)
        and ``"hme_robust"`` (posterior expectation of ``1/L``, sanity-check).

        If ``loglike_key`` is not available, log-likelihood is reconstructed as
        ``loglike = post - prior``.
        """

        if logpost_key is None:
            logpost_key = self.posterior_key

        if logpost_key not in table.colnames:
            return EvidenceEstimate(method=method, logz=float("nan"), details={"reason": f"missing '{logpost_key}'"})
        if logprior_key not in table.colnames:
            return EvidenceEstimate(method=method, logz=float("nan"), details={"reason": f"missing '{logprior_key}'"})

        logpost_all = _as_float_array(table[logpost_key])
        logprior_all = _as_float_array(table[logprior_key])

        if loglike_key is not None and loglike_key in table.colnames:
            loglike_all = _as_float_array(table[loglike_key])
        else:
            # Reconstruct loglike = post - prior
            loglike_all = logpost_all - logprior_all

        # Rebuild the same “finite” mask used by summarize_results (posterior + parameters + prior/like)
        mask = np.isfinite(logpost_all) & np.isfinite(logprior_all) & np.isfinite(loglike_all)
        for k in self.parameter_keys:
            if k not in table.colnames:
                return EvidenceEstimate(method=method, logz=float("nan"), details={"reason": f"missing parameter column '{k}'"})
            mask &= np.isfinite(_as_float_array(table[k]))

        if not np.any(mask):
            return EvidenceEstimate(method=method, logz=float("nan"), details={"reason": "no finite rows after masking"})

        logpost = logpost_all[mask]
        logprior = logprior_all[mask]
        loglike = loglike_all[mask]

        # Weights from posterior
        w = normalize_weights(np.exp(logpost - np.max(logpost)))

        map_idx = int(np.nanargmax(logpost))
        ll_map = float(loglike[map_idx])
        lp_map = float(logprior[map_idx])

        if method.lower() == "laplace":
            est = laplace_logz(ll_map, lp_map, self.covariance)
        elif method.lower() in ("hme", "hme_robust", "harmonic", "harmonic_mean"):
            est = harmonic_mean_logz(loglike, w, trim_frac=hme_trim_frac)
        else:
            est = EvidenceEstimate(method=method, logz=float("nan"), details={"reason": f"unknown method '{method}'"})

        # Store on the object for export
        self.evidence = est
        return est

    def to_json_dict(self) -> Dict[str, Any]:
        """Convert summary to a JSON-serializable dictionary (numpy arrays -> lists)."""
        def _tolist(a):
            if isinstance(a, np.ndarray):
                return a.tolist()
            if isinstance(a, np.generic):
                return a.item()
            if isinstance(a, dict):
                return {k: _tolist(v) for k, v in a.items()}
            if isinstance(a, (list, tuple)):
                return [_tolist(v) for v in a]
            return a

        out: Dict[str, Any] = {
            "parameter_keys": self.parameter_keys,
            "parameter_sections": self.parameter_sections,
            "parameter_names": self.parameter_names,
            "posterior_key": self.posterior_key,
            "n_samples": int(self.n_samples),
            "map_index": int(self.map_index),
            "weights": _tolist(self.weights),
            "logpost": _tolist(self.logpost),
            "map": _tolist(self.map),
            "mean": _tolist(self.mean),
            "covariance": _tolist(self.covariance),
            "correlation": _tolist(self.correlation),
            "percentiles": _tolist(np.asarray(self.percentiles, dtype=float)),
            "percentiles_values": _tolist(self.percentiles_values),
            "percentiles_logpost": _tolist(self.percentiles_logpost),
            "hdi_intervals_68": {k: [list(iv) for iv in v] for k, v in self.hdi_intervals_68.items()},
            "hdi_intervals_95": {k: [list(iv) for iv in v] for k, v in self.hdi_intervals_95.items()},
            "map_1d": {k: _tolist(v) for k, v in self.map_1d.items()},
            "evidence": None if self.evidence is None else {
                "method": self.evidence.method,
                "logz": self.evidence.logz,
                "logz_err": self.evidence.logz_err,
                "details": self.evidence.details,
            },
            "pdf_1d": {
                k: {kk: _tolist(vv) for kk, vv in d.items()}
                for k, d in self.pdf_1d.items()
            },
            "pdf_2d": {
                f"{k0}__{k1}": {kk: _tolist(vv) for kk, vv in d.items()}
                for (k0, k1), d in self.pdf_2d.items()
            },
            "extra_info": self.extra_info,
        }
        return out

    @classmethod
    def from_json(cls):
        #TODO
        pass

    def write_json(self, path: str, *, overwrite: bool = True, indent: int = 2) -> str:
        """Write a JSON summary file."""
        if (not overwrite) and os.path.exists(path):
            raise FileExistsError(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, indent=indent)
        return path

    def to_fits(self) -> fits.HDUList:
        """
        Build a FITS HDUList with:
        - PrimaryHDU: extra_info
        - ImageHDU: COVARIANCE (+ header with means/MAP)
        - ImageHDU: CORRELATION
        - BinTableHDU: PERCENTILES
        - BinTableHDU: PDF1D
        - ImageHDUs: PDF2D_<i>__<j> and ENCFRAC_<i>__<j>
        """
        prim = fits.PrimaryHDU()
        # Store extra_info as header cards (best effort)
        for k, v in (self.extra_info or {}).items():
            try:
                prim.header[str(k)[:8]] = v
            except Exception:
                # Skip non-serializable header items
                continue

        hdr = fits.Header()
        hdr["N_SAMP"] = int(self.n_samples)
        hdr["MAP_IDX"] = int(self.map_index)
        hdr["POSTKEY"] = self.posterior_key

        for i, (full_key, sect, name) in enumerate(zip(self.parameter_keys, self.parameter_sections, self.parameter_names)):
            tag = f"P{i:03d}"
            hdr[f"{tag}NM"] = name[:68]
            hdr[f"{tag}SC"] = sect[:68]
            hdr[f"{tag}KY"] = full_key[:68]
            if i < self.mean.size:
                if np.isfinite(self.mean[i]):
                    hdr[f"{tag}MN"] = float(self.mean[i]), "mean"
                else:
                    hdr[f"{tag}MN"] = "nan", "mean not finite"
            if i < self.map.size:
                if np.isfinite(self.map[i]):
                    hdr[f"{tag}MP"] = float(self.map[i]), "MAP"
                else:
                    hdr[f"{tag}MP"] = "nan", "map not finite"

        if self.evidence is not None and np.isfinite(self.evidence.logz):
            prim.header["LOGZ"] = float(self.evidence.logz), "log-evidence estimate"
            if self.evidence.logz_err is not None and np.isfinite(self.evidence.logz_err):
                prim.header["LOGZERR"] = float(self.evidence.logz_err), "log-evidence error"
            prim.header["LOGZMET"] = self.evidence.method[:20], "evidence estimation method"

        hdus: List[fits.hdu.base.ExtensionHDU] = [prim]
        hdus.append(fits.ImageHDU(data=_as_float_array(self.covariance), header=hdr, name="COVARIANCE"))
        hdus.append(fits.ImageHDU(data=_as_float_array(self.correlation), name="CORRELATION"))

        # Percentiles table
        t_pct = Table()
        t_pct["percentile"] = np.asarray(self.percentiles, dtype=float)
        for i, (name, section) in enumerate(zip(self.parameter_names, self.parameter_sections)):
            k = ".".join([section, name])
            t_pct[f"{k}_val"] = _as_float_array(self.percentiles_values[i, :]) if self.percentiles_values.size else np.full(len(self.percentiles), np.nan)
            t_pct[f"{k}_logp"] = _as_float_array(self.percentiles_logpost[i, :]) if self.percentiles_logpost.size else np.full(len(self.percentiles), np.nan)

        # Add HDI intervals as header cards on the percentiles HDU (compact)
        pct_hdr = fits.Header()
        # Store 68% and 95% HDI intervals explicitly.
        for name, ivs in self.hdi_intervals_68.items():
            for j, (lo, hi) in enumerate(ivs[:2]):
                pct_hdr[f"HIERARCH {name}_68L{j}"] = float(lo) if np.isfinite(lo) else "nan", "68% lower limit"
                pct_hdr[f"HIERARCH {name}_68U{j}"] = float(hi) if np.isfinite(hi) else "nan", "68% upper limit"
        for name, ivs in self.hdi_intervals_95.items():
            for j, (lo, hi) in enumerate(ivs[:2]):
                pct_hdr[f"HIERARCH {name}_95L{j}"] = float(lo) if np.isfinite(lo) else "nan", "95% lower limit"
                pct_hdr[f"HIERARCH {name}_95U{j}"] = float(hi) if np.isfinite(hi) else "nan", "95% upper limit"

        hdus.append(fits.BinTableHDU(t_pct, name="PERCENTILES", header=pct_hdr))

        # PDF1D table: store edges/pdf/kde per parameter as separate columns
        t_pdf1 = Table()
        t_pdf1_edges = Table()
        for name, d in self.pdf_1d.items():
            edges = _as_float_array(d["edges"])
            centers = _as_float_array(d.get("grid", 0.5 * (edges[:-1] + edges[1:])))
            hist_pdf = _as_float_array(d["hist_pdf"])
            kde_pdf = _as_float_array(d.get("kde_pdf", np.full_like(hist_pdf, np.nan)))

            # FITS bin table columns must have consistent lengths across rows.
            # Store centers/PDF/KDE in PDF1D (all length = n_bins).
            t_pdf1[f"{name}_x"] = centers
            t_pdf1[f"{name}_pdf"] = hist_pdf
            t_pdf1[f"{name}_kde"] = kde_pdf
            # Store raw histogram edges separately (length = n_bins + 1).
            t_pdf1_edges[f"{name}_edges"] = edges

        if len(t_pdf1.colnames) > 0:
            hdus.append(fits.BinTableHDU(t_pdf1, name="PDF1D"))
        if len(t_pdf1_edges.colnames) > 0:
            hdus.append(fits.BinTableHDU(t_pdf1_edges, name="PDF1D_EDGES"))

        # PDF2D images
        for (n0, n1), d in self.pdf_2d.items():
            # pdf stored as (ny, nx) matching ygrid,xgrid
            hdr2 = fits.Header()
            hdr2["AX0"] = n0
            hdr2["AX1"] = n1
            hdus.append(fits.ImageHDU(data=_as_float_array(d["pdf"]), header=hdr2, name=f"PDF2D_{n0}__{n1}"[:68]))
            if "enclosed_fraction" in d:
                hdus.append(fits.ImageHDU(data=_as_float_array(d["enclosed_fraction"]), header=hdr2, name=f"ENCFRAC_{n0}__{n1}"[:68]))

        return fits.HDUList(hdus)

    def write_fits(self, path: str, *, overwrite: bool = True) -> str:
        """Write FITS summary file."""
        hdul = self.to_fits()
        hdul.writeto(path, overwrite=overwrite)
        return path

    @classmethod
    def from_fits(cls):
        #TODO
        pass

    def plot_1d_pdfs(
        self,
        outdir: str,
        *,
        show: bool = False,
        dpi: int = 200,
    ) -> List[str]:
        """Save 1D PDF plots for all parameters."""
        os.makedirs(outdir, exist_ok=True)
        paths: List[str] = []
        for i, name in enumerate(self.parameter_names):
            if name not in self.pdf_1d:
                continue
            d = self.pdf_1d[name]
            fig, ax = plt.subplots()
            ax.plot(d["grid"], d["hist_pdf"], label="hist")
            if "kde_pdf" in d and np.any(np.isfinite(d["kde_pdf"])):
                ax.plot(d["grid"], d["kde_pdf"], label="kde")
            if i < self.mean.size:
                ax.axvline(self.mean[i], label="mean")
            if i < self.map.size:
                ax.axvline(self.map[i], label="MAP")
            # Percentiles
            if self.percentiles_values.size:
                for p, v in zip(self.percentiles, self.percentiles_values[i, :]):
                    ax.axvline(v, alpha=0.5)
            # HDI 95% then 68% to keep narrower interval visible on top.
            if name in self.hdi_intervals_95:
                for lo, hi in self.hdi_intervals_95[name]:
                    ax.axvspan(lo, hi, alpha=0.10, color="C0", label="HDI 95%")
            if name in self.hdi_intervals_68:
                for lo, hi in self.hdi_intervals_68[name]:
                    ax.axvspan(lo, hi, alpha=0.20, color="C0", label="HDI 68%")
            ax.set_title(name)
            ax.set_xlabel(name)
            ax.set_ylabel("PDF")
            handles, labels = ax.get_legend_handles_labels()
            uniq = dict(zip(labels, handles))
            ax.legend(uniq.values(), uniq.keys())

            fp = os.path.join(outdir, f"pdf1d_{name}.png")
            fig.savefig(fp, dpi=dpi, bbox_inches="tight")
            paths.append(fp)
            if show:
                plt.show()
            else:
                plt.close(fig)
        return paths

    def plot_2d_pdfs(
        self,
        outdir: str,
        *,
        levels: Sequence[float] = (0.68, 0.95),
        show: bool = False,
        dpi: int = 200,
    ) -> List[str]:
        """Save 2D PDF plots with HPD enclosed-fraction contours."""
        os.makedirs(outdir, exist_ok=True)
        paths: List[str] = []
        for (n0, n1), d in self.pdf_2d.items():
            xg = _as_float_array(d["xgrid"])
            yg = _as_float_array(d["ygrid"])
            pdf = _as_float_array(d["pdf"])  # (ny, nx)
            frac = _as_float_array(d.get("enclosed_fraction", np.full_like(pdf, np.nan)))

            fig, ax = plt.subplots()
            X, Y = np.meshgrid(xg, yg, indexing="xy")
            im = ax.pcolormesh(X, Y, pdf, shading="auto", cmap="Greys")
            fig.colorbar(im, ax=ax, label="PDF")

            if np.any(np.isfinite(frac)):
                cs = ax.contour(X, Y, frac, levels=list(levels), linewidths=1.0)
                ax.clabel(cs, inline=True, fontsize=8, fmt=lambda v: f"{v:.2f}")

            ax.set_xlabel(n0)
            ax.set_ylabel(n1)
            ax.set_title(f"{n0} vs {n1}")

            fp = os.path.join(outdir, f"pdf2d_{n0}__{n1}.png")
            fig.savefig(fp, dpi=dpi, bbox_inches="tight")
            paths.append(fp)
            if show:
                plt.show()
            else:
                plt.close(fig)
        return paths

    def corner_plot(
        self,
        outpath: str,
        *,
        bins: int = 50,
        max_points: int = 20000,
        dpi: int = 200,
        show: bool = False,
    ) -> str:
        """Build a corner plot"""
        npar = len(self.parameter_names)
        if npar == 0 or self.samples.size == 0:
            raise ValueError("No samples available to plot.")

        # Subsample points for scatter
        N = self.n_samples
        if N > max_points:
            # sample indices proportional to weights
            idx = np.random.choice(np.arange(N), size=max_points, replace=False, p=self.weights)
        else:
            idx = np.arange(N)

        S = self.samples[:, idx]
        w = self.weights[idx]

        fig, axes = plt.subplots(
            npar, npar, figsize=(2.2 * npar, 2.2 * npar),
            sharex="col",
            constrained_layout=True)

        for i in range(npar):
            for j in range(npar):
                ax = axes[i, j]
                if i == j:
                    x = self.samples[i, :]
                    edges, pdf = histogram_pdf_1d(x, self.weights, bins=bins)
                    xc = 0.5 * (edges[:-1] + edges[1:])
                    ax.plot(xc, pdf, lw=1.2, color="black")
                    # mark mean/MAP
                    ax.axvline(self.mean[i], lw=1.0, alpha=0.8, color="r")
                    ax.axvline(self.map[i], lw=1.0, alpha=0.8, color="b")
                    ax.set_yticks([])
                elif i > j:
                    H, xedges, yedges = np.histogram2d(S[j, :], S[i, :], weights=w, bins=bins, density=True)
                    
                    fraction = enclosed_fraction_map(H, xedges=xedges, yedges=yedges)
                    xbins = 0.5 * (xedges[:-1] + xedges[1:])
                    ybins = 0.5 * (yedges[:-1] + yedges[1:])
                    ax.contourf(xbins, ybins, fraction.T, levels=[0.0, 0.68, 0.95],
                                cmap="Spectral")
                else:
                    ax.axis("off")

                if i == npar - 1:
                    ax.set_xlabel(self.parameter_names[j])
                elif i < npar - 1:
                    pass
                    # ax.set_xticks([])
                if j == 0 and i >= j:
                    ax.set_ylabel(self.parameter_names[i])
                else:
                    ax.set_yticks([])

        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
        return outpath


# -----------------------------------------------------------------------------
# Latent -> physical SFH parameters
# -----------------------------------------------------------------------------
#
# Non-parametric SFH models sampled with ``use_transforms = T`` store latent
# (unit-cube) variables in the results. The transforms used by BESTA map the
# uniform latent prior onto a uniform prior over the physical ordered region,
# so their Jacobian is constant: transformed samples are physical posterior
# samples with unchanged weights, and the ``post`` column remains a valid
# (unnormalised) physical log-posterior.

_SFH_SPACE_META_KEY = "besta_sfh_space"


def _get_option(section: Mapping[str, Any], name: str, default=None):
    """Case-insensitive lookup of an option in an ini section."""
    lowered = {str(key).lower(): value for key, value in section.items()}
    return lowered.get(name.lower(), default)


def find_sfh_modules(ini: Mapping[str, Any]) -> List[str]:
    """Return the pipeline modules whose configuration defines an SFH model.

    Parameters
    ----------
    ini : dict
        Parsed CosmoSIS configuration (e.g. :attr:`besta.io.Reader.ini`).

    Returns
    -------
    list of str
        Module (section) names with an ``SFHModel`` option, in pipeline order.
    """
    modules = ini["pipeline"]["modules"]
    if isinstance(modules, str):
        modules = modules.replace(",", " ").split()
    return [
        module for module in modules
        if module in ini and _get_option(ini[module], "SFHModel") is not None
    ]


def build_sfh_model(ini: Mapping[str, Any], module_name: Optional[str] = None):
    """Rebuild the SFH model of a run from its configuration.

    Uses :func:`besta.sfh.build_sfh_from_options`, the same builder as the
    pipeline modules, so the model (bins, ``use_transforms``, age of the
    Universe at the source redshift, ...) matches the one used during
    sampling. No SSP models or data are loaded.

    Parameters
    ----------
    ini : dict
        Parsed CosmoSIS configuration (e.g. :attr:`besta.io.Reader.ini`).
    module_name : str, optional
        Module section defining the SFH. Defaults to the first module with an
        ``SFHModel`` option.

    Returns
    -------
    :class:`besta.sfh.SFHBase`
        The configured SFH model.
    """
    from besta.sfh import build_sfh_from_options

    candidates = find_sfh_modules(ini)
    if module_name is None:
        if not candidates:
            raise ValueError(
                "No pipeline module with an 'SFHModel' option was found.")
        module_name = candidates[0]
        if len(candidates) > 1:
            logger.warning(
                "Several modules define an SFH model (%s); using '%s'.",
                ", ".join(candidates), module_name)
    elif module_name not in candidates:
        raise ValueError(
            f"Module '{module_name}' does not define an 'SFHModel' option.")

    return build_sfh_from_options(ini[module_name])


def sfh_parameter_columns(
    table: Table, sfh_model, *, parameter_prefix: str = "--"
) -> List[str]:
    """Table columns holding the SFH bin parameters, ordered as ``sfh_bin_keys``.

    Returns an empty list for models without bin parameters (parametric SFHs).
    Matching is case-insensitive (results files store lower-case names).
    """
    bin_keys = getattr(sfh_model, "sfh_bin_keys", None)
    if not bin_keys:
        return []
    lookup = {name.lower(): name for name in table.colnames}
    columns = []
    for key in bin_keys:
        wanted = f"{sfh_model.sect_name}{parameter_prefix}{key}".lower()
        if wanted not in lookup:
            raise KeyError(
                f"Column '{wanted}' for SFH parameter '{key}' not found in the table.")
        columns.append(lookup[wanted])
    return columns


def latent_to_physical_table(
    table: Table,
    sfh_model,
    *,
    parameter_prefix: str = "--",
    keep_latent: bool = False,
    latent_prefix: str = "latent_",
) -> Table:
    """Convert the latent SFH columns of a results table to physical values.

    Parameters
    ----------
    table : astropy.table.Table
        Results table (e.g. :attr:`besta.io.Reader.results_table`).
    sfh_model : :class:`besta.sfh.SFHBase`
        SFH model used in the run (see :func:`build_sfh_model`).
    parameter_prefix : str
        Section/name delimiter in the column names (default ``"--"``).
    keep_latent : bool
        If ``True``, keep copies of the latent columns named
        ``latent_prefix + column``. Note that these still contain
        ``parameter_prefix`` and are therefore picked up by
        :func:`summarize_results` unless ``parameter_keys`` is given.
    latent_prefix : str
        Prefix for the latent copies.

    Returns
    -------
    astropy.table.Table
        A copy of ``table`` whose SFH columns hold physical values (same
        column names). Rows that cannot be mapped are NaN (and are dropped by
        :func:`summarize_results`). ``meta["besta_sfh_space"]`` is set to
        ``"physical"``. The input table is not modified.

    Notes
    -----
    The weights and the ``post`` column are unchanged: the SFH transforms
    have a constant Jacobian (see the section comment above).
    """
    out = table.copy()
    if table.meta.get(_SFH_SPACE_META_KEY) == "physical":
        logger.warning("Table is already in physical SFH space; not converting again.")
        return out
    out.meta[_SFH_SPACE_META_KEY] = "physical"
    out.meta["besta_sfh_model"] = type(sfh_model).__name__
    if not getattr(sfh_model, "use_transforms", False):
        logger.info("The SFH model does not use latent transforms; "
                    "the table is already in physical space.")
        return out

    columns = sfh_parameter_columns(
        table, sfh_model, parameter_prefix=parameter_prefix)
    if not columns:
        return out

    latent = np.column_stack([_as_float_array(table[col]) for col in columns])
    physical = sfh_model.to_physical_batch(latent)

    n_unmapped = int(np.sum(
        np.all(np.isfinite(latent), axis=1) & ~np.all(np.isfinite(physical), axis=1)))
    if n_unmapped:
        logger.warning(
            "%d of %d samples lie outside the latent support and were set to NaN.",
            n_unmapped, len(table))

    for index, col in enumerate(columns):
        if keep_latent:
            out[latent_prefix + col] = table[col].copy()
        out[col] = physical[:, index]
    logger.info("Converted %d SFH parameters of %d samples to physical space.",
                len(columns), len(table))
    return out


def to_physical_table(reader, *, module_name: Optional[str] = None, **kwargs) -> Table:
    """Results of a run with the SFH parameters in physical space.

    Parameters
    ----------
    reader : :class:`besta.io.Reader`
        Reader of the run. Its results are loaded if needed.
    module_name : str, optional
        Only convert the SFH of this module (default: every module defining an
        ``SFHModel`` that uses latent transforms).
    **kwargs
        Passed to :func:`latent_to_physical_table`.

    Returns
    -------
    astropy.table.Table

    Examples
    --------
    >>> reader = Reader.from_results_file("results.txt")
    >>> summary = summarize_results(to_physical_table(reader))
    """
    table = getattr(reader, "_results_table", None)
    if table is None:
        reader.load_results()
        table = reader.results_table
    return _table_to_physical(table, reader.ini, module_name, **kwargs)


def _table_to_physical(table: Table, ini: Mapping[str, Any],
                       module_name: Optional[str] = None, **kwargs) -> Table:
    """Convert the latent SFH columns of every SFH module in ``ini``."""
    if table.meta.get(_SFH_SPACE_META_KEY) == "physical":
        return table
    for module, model in _unique_sfh_models(ini, module_name).items():
        if model.use_transforms:
            logger.info("Converting the latent SFH parameters of module '%s' "
                        "to physical space.", module)
            table = latent_to_physical_table(table, model, **kwargs)
            table.meta.pop(_SFH_SPACE_META_KEY, None)
    table = table.copy(copy_data=False)
    table.meta[_SFH_SPACE_META_KEY] = "physical"
    return table


# CosmoSIS text results in physical SFH space
#
# A converted results file keeps the CosmoSIS text layout (column header,
# run metadata, embedded ini blocks, samples and trailer), so it can be read
# with :class:`besta.io.Reader` and :func:`besta.io.read_results_file` like
# the original. The embedded configuration is edited so that it describes
# the physical table:
#
# - ``use_transforms = F`` in every module defining an SFH, so the model
#   rebuilt from the file expects physical parameters;
# - physical prior ranges for the SFH parameters in the values block;
# - ``[output] filename`` pointing to the converted file.
#
# Trailer lines ``#besta_sfh_space=physical`` (read into ``table.meta``)
# prevent converting the same table twice.

PHYSICAL_RESULTS_SUFFIX = "_physical"

_PARAMS_INI_BLOCK = ("## START_OF_PARAMS_INI", "## END_OF_PARAMS_INI")
_VALUES_INI_BLOCK = ("## START_OF_VALUES_INI", "## END_OF_VALUES_INI")


def physical_results_path(path: str) -> str:
    """Default path of the physical-space copy of a results file.

    ``results.txt`` -> ``results_physical.txt`` (``.txt`` is appended if the
    path has no extension, as CosmoSIS does).
    """
    root, ext = os.path.splitext(path)
    return f"{root}{PHYSICAL_RESULTS_SUFFIX}{ext or '.txt'}"


def _results_file_path(ini: Mapping[str, Any]) -> str:
    """Results file written by CosmoSIS for a run (same rule as ``Reader``)."""
    path = str(ini["output"]["filename"])
    return path if ".txt" in path else path + ".txt"


def _split_results_text(lines: Sequence[str]):
    """Split a CosmoSIS text results file into header / data / trailer lines."""
    if not lines or not lines[0].startswith("#"):
        raise ValueError("Expected a CosmoSIS text results file (first line '#...').")
    start = 1
    while start < len(lines) and lines[start].startswith("#"):
        start += 1
    end = len(lines)
    while end > start and lines[end - 1].startswith("#"):
        end -= 1
    return list(lines[:start]), list(lines[end:])


def _format_ini_value(value) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        return " ".join(repr(float(v)) if isinstance(v, (float, np.floating))
                        else str(v) for v in value)
    return str(value)


def _set_ini_block_options(lines, block, updates):
    """Set options inside an embedded ``## [section]`` ini block.

    ``updates`` maps ``(section, key)`` to the new value (as text). Existing
    options are replaced (keys are matched case-insensitively); missing ones
    are added at the end of their section. Sections are matched exactly.
    """
    begin_tag, end_tag = block
    try:
        begin = next(i for i, line in enumerate(lines) if line.startswith(begin_tag))
        end = next(i for i, line in enumerate(lines) if line.startswith(end_tag))
    except StopIteration as exc:
        raise ValueError(f"Block {begin_tag} not found in the results file.") from exc

    pending = {(sect, key.lower()): value for (sect, key), value in updates.items()}
    out = list(lines[:begin + 1])
    section = None

    def flush(section_name):
        # Add the options of the section that were not present in the file.
        for (sect, key), value in list(pending.items()):
            if sect == section_name:
                out.append(f"## {key} = {value}\n")
                del pending[(sect, key)]

    for line in lines[begin + 1:end]:
        body = line[2:].strip() if line.startswith("##") else line.strip()
        if body.startswith("[") and body.endswith("]"):
            flush(section)
            section = body[1:-1]
        elif "=" in body and section is not None:
            key = body.split("=", 1)[0].strip().lower()
            if (section, key) in pending:
                line = f"## {key} = {pending.pop((section, key))}\n"
        elif not body:
            # A blank "## " line closes the section in CosmoSIS output.
            flush(section)
        out.append(line)
    flush(section)
    for (sect, key), value in pending.items():
        out.extend([f"## [{sect}]\n", f"## {key} = {value}\n", "## \n"])
    out.extend(lines[end:])
    return out


def _unique_sfh_models(ini: Mapping[str, Any], module_name: Optional[str] = None):
    """``{module: sfh_model}`` for the SFH modules to convert (one per section)."""
    modules = [module_name] if module_name is not None else find_sfh_modules(ini)
    models, sections = {}, {}
    for module in modules:
        model = build_sfh_model(ini, module)
        if model.sect_name in sections:
            logger.warning(
                "Modules '%s' and '%s' share the SFH section '%s'; "
                "converting it once with the SFH of '%s'.",
                sections[model.sect_name], module, model.sect_name,
                sections[model.sect_name])
            continue
        sections[model.sect_name] = module
        models[module] = model
    return models


def _physical_sfh_model(ini: Mapping[str, Any], module: str):
    """SFH model of ``module`` built with ``use_transforms = False``."""
    from besta.sfh import build_sfh_from_options

    options = dict(ini[module])
    options["use_transforms"] = False
    model = build_sfh_from_options(options)
    if model.use_transforms:
        raise ValueError(
            f"Could not disable use_transforms for module '{module}' "
            "(is it set in SFHArgs?).")
    return model


def convert_results_file(
    results_path: str,
    output_path: Optional[str] = None,
    *,
    module_name: Optional[str] = None,
    overwrite: bool = True,
) -> Optional[str]:
    """Write a copy of a CosmoSIS text results file in physical SFH space.

    Parameters
    ----------
    results_path : str
        CosmoSIS text results file (with the embedded ini blocks).
    output_path : str, optional
        Output file. Defaults to :func:`physical_results_path`.
    module_name : str, optional
        Only convert the SFH of this module (default: all modules defining an
        ``SFHModel``).
    overwrite : bool
        Overwrite ``output_path`` if it exists.

    Returns
    -------
    str or None
        Path of the file written, or ``None`` if there was nothing to convert
        (no SFH module using ``use_transforms``, or a table that is already
        in physical space).
    """
    results_path = os.path.expandvars(results_path)
    output_path = os.path.expandvars(
        output_path or physical_results_path(results_path))
    if os.path.abspath(output_path) == os.path.abspath(results_path):
        raise ValueError("The output file must differ from the input file.")
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"{output_path} already exists.")

    table = io.read_results_file(results_path)
    if table.meta.get(_SFH_SPACE_META_KEY) == "physical":
        logger.info("%s is already in physical SFH space; nothing to convert.",
                    results_path)
        return None

    ini = io.Reader.read_ini_file_from_results(results_path)
    models = {module: model for module, model in
              _unique_sfh_models(ini, module_name).items() if model.use_transforms}
    if not models:
        logger.info("No SFH model with use_transforms in %s; nothing to convert.",
                    results_path)
        return None

    converted = table
    for module, model in models.items():
        logger.info("Converting the latent '%s' parameters of module '%s' (%s) "
                    "to physical space.", model.sect_name, module,
                    type(model).__name__)
        converted = latent_to_physical_table(converted, model)
        converted.meta.pop(_SFH_SPACE_META_KEY, None)   # next model: convert too

    # Physical versions of the models, for the edited configuration.
    physical_models = {module: _physical_sfh_model(ini, module) for module in models}

    with open(results_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    header, trailer = _split_results_text(lines)

    params_updates = {("output", "filename"): output_path}
    values_updates = {}
    for module, model in physical_models.items():
        params_updates[(module, "use_transforms")] = "F"
        for key in model.sfh_bin_keys:
            values_updates[(model.sect_name, key)] = _format_ini_value(
                model.free_params[key])
    header = _set_ini_block_options(header, _PARAMS_INI_BLOCK, params_updates)
    if any(line.startswith(_VALUES_INI_BLOCK[0]) for line in header):
        header = _set_ini_block_options(header, _VALUES_INI_BLOCK, values_updates)

    trailer = trailer + [
        f"#{_SFH_SPACE_META_KEY}=physical\n",
        "#besta_sfh_model=" + ",".join(
            type(m).__name__ for m in models.values()) + "\n",
        f"#besta_converted_from={results_path}\n",
    ]

    data = np.column_stack([_as_float_array(converted[name])
                            for name in table.colnames])
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        file.writelines(header)
        np.savetxt(file, data, delimiter="\t", fmt="%.17g")
        file.writelines(trailer)
    logger.info("Physical-space results written to %s", output_path)
    return output_path


def write_physical_results(ini: Mapping[str, Any], **kwargs) -> Optional[str]:
    """Convert the results file of a run described by ``ini`` (see
    :func:`convert_results_file`). Returns the output path or ``None``."""
    output = ini.get("output", {})
    fmt = str(_get_option(output, "format", "text")).lower()
    if fmt != "text":
        logger.info("Output format '%s' is not 'text'; no physical-space table "
                    "is written.", fmt)
        return None
    return convert_results_file(_results_file_path(ini), **kwargs)


def load_physical_results(results_path: str, *, module_name: Optional[str] = None
                          ) -> Table:
    """Read a results file with the SFH parameters in physical space.

    Latent SFH parameters are converted on the fly (and the conversion is
    logged); tables that are already physical are returned as read.
    """
    table = io.read_results_file(results_path)
    if table.meta.get(_SFH_SPACE_META_KEY) == "physical":
        return table
    ini = io.Reader.read_ini_file_from_results(results_path)
    return _table_to_physical(table, ini, module_name)


# -----------------------------------------------------------------------------
# Utilities for SFH reconstruction
# -----------------------------------------------------------------------------
#
# Each posterior sample is loaded into the SFH model of the run (built with
# ``use_transforms = F``, as the samples are converted to physical space
# first) and PST evaluates the cumulative mass formed on a grid of lookback
# times in the same fashion as in the fit.
#
# Quantities (relative to the stellar mass formed up to the observation):
# - ``mass_fraction``: M(t_obs - lookback) / M(t_obs) at the bin edges;
# - ``ssfr``: mass formed in each lookback bin / (M(t_obs) * bin width) [1/yr];
# - ``metallicity``: ISM metallicity at the bin edges (models with an
#   enrichment history only);
# - ``ssfr_tau``: sSFR averaged over the last ``tau`` [1/yr].


def _percentile_label(q: float) -> str:
    """0.16 -> 'p16', 0.025 -> 'p2_5'."""
    return "p" + f"{round(100 * q, 3):g}".replace(".", "_")


def default_lookback_edges(today_gyr: float, n_bins: int = 40,
                           min_lookback: float = 1e-3) -> np.ndarray:
    """Lookback-time bin edges in Gyr, starting in 0, then log-spaced between
    ``min_lookback`` and ``today``."""
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")
    if not 0 < min_lookback < today_gyr:
        raise ValueError("min_lookback must lie between 0 and the age of the "
                         f"Universe at the source ({today_gyr:.4g} Gyr).")
    return np.concatenate(([0.0], np.geomspace(min_lookback, today_gyr, n_bins)))


def _read_embedded_values(results_path: str) -> Dict[str, Dict[str, Any]]:
    """Parameter values block (priors) stored in a CosmoSIS results file."""
    with open(results_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
    try:
        begin = next(i for i, line in enumerate(lines)
                     if line.startswith(_VALUES_INI_BLOCK[0]))
        end = next(i for i, line in enumerate(lines)
                   if line.startswith(_VALUES_INI_BLOCK[1]))
    except StopIteration:
        return {}
    content = "".join(line[3:] if line.startswith("## ") else line.lstrip("#")
                      for line in lines[begin + 1:end])
    return io._ini_string_to_dict(content)


def _is_fixed_value(value) -> bool:
    return np.ndim(value) == 0 and isinstance(value, (int, float, np.number, bool))


@dataclass
class SFHReconstruction:
    """Posterior SFHs evaluated on a grid of lookback times.

    Attributes
    ----------
    lookback_edges : ndarray, shape (n_bins + 1,)
        Lookback-time bin edges [Gyr].
    percentiles : tuple of float
        Quantiles (in [0, 1]) stored in the ``*_percentiles`` arrays.
    mass_fraction, ssfr, metallicity, ssfr_tau : ndarray
        Per-sample values, shapes (n_samples, n_bins + 1), (n_samples, n_bins),
        (n_samples, n_bins + 1) and (n_samples, n_tau). ``metallicity`` is
        ``None`` for models without an enrichment history.
    taus : ndarray
        Timescales [Gyr] of ``ssfr_tau``.
    weights : ndarray, shape (n_samples,)
        Sample weights used for the percentiles.
    sample_table : astropy.table.Table
        Per-sample input values (SFH parameters, ``post``, weights, ...).
    meta : dict
        Model name, redshift, age of the Universe, source, ...
    """

    lookback_edges: np.ndarray
    percentiles: Tuple[float, ...]
    mass_fraction: np.ndarray
    ssfr: np.ndarray
    ssfr_tau: np.ndarray
    taus: np.ndarray
    weights: np.ndarray
    metallicity: Optional[np.ndarray] = None
    sample_table: Optional[Table] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    _percentile_cache: Dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    # --- Derived grids --------------------------------------------------------
    @property
    def lookback_low(self) -> np.ndarray:
        return self.lookback_edges[:-1]

    @property
    def lookback_high(self) -> np.ndarray:
        return self.lookback_edges[1:]

    @property
    def lookback_centres(self) -> np.ndarray:
        return 0.5 * (self.lookback_edges[:-1] + self.lookback_edges[1:])

    @property
    def n_samples(self) -> int:
        return int(self.weights.size)

    # --- Percentiles ----------------------------------------------------------
    def quantity(self, name: str) -> Optional[np.ndarray]:
        """Per-sample array of ``mass_fraction``, ``ssfr``, ``metallicity`` or ``ssfr_tau``."""
        if name not in ("mass_fraction", "ssfr", "metallicity", "ssfr_tau"):
            raise KeyError(f"Unknown SFH quantity '{name}'.")
        return getattr(self, name)

    def percentile_values(self, name: str) -> Optional[np.ndarray]:
        """Weighted percentiles of a quantity, shape (n_percentiles, n_points)."""
        if name in self._percentile_cache:
            return self._percentile_cache[name]
        values = self.quantity(name)
        if values is None:
            return None
        if values.shape[0] == 0:
            result = np.full((len(self.percentiles), values.shape[1]), np.nan)
        else:
            result = np.column_stack([
                weighted_quantile(values[:, j], self.weights, self.percentiles)
                for j in range(values.shape[1])])
        self._percentile_cache[name] = result
        return result

    def _plot_percentiles(self, name: str, ax=None, *, color="C0", alpha=0.25,
                          **kwargs):
        """Plot the percentile bands of a quantity vs lookback time.

        Parameters
        ----------
        name : str
            ``"mass_fraction"`` or ``"metallicity"`` (values at the bin
            edges, drawn as lines) or ``"ssfr"`` (bin averages, drawn as
            steps).
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If ``None``, a new figure and axes are created.
        color : str
            Colour of the bands and the median.
        alpha : float
            Opacity of each band (nested bands add up).
        **kwargs
            Passed to :func:`matplotlib.axes.Axes.fill_between`.
        """
        if name == "ssfr_tau":
            raise ValueError("'ssfr_tau' is not a function of lookback time; "
                             "use percentile_values('ssfr_tau') and taus.")
        values = self.percentile_values(name)
        if values is None:
            raise ValueError(f"No values for '{name}' (e.g. no enrichment history).")
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))

        step = name == "ssfr"                 # bin averages -> steps over the edges
        x = self.lookback_edges

        def curve(v):
            return np.append(v, v[-1]) if step else v

        quantiles = np.asarray(self.percentiles, dtype=float)
        order = np.argsort(quantiles)
        n_q = order.size
        for i in range(n_q // 2):
            low, high = order[i], order[-(i + 1)]
            ax.fill_between(
                x, curve(values[low]), curve(values[high]),
                step="post" if step else None, color=color, alpha=alpha, lw=0,
                label=f"{100 * quantiles[low]:g}-{100 * quantiles[high]:g}%",
                **kwargs)
        if n_q % 2 == 1:
            mid = order[n_q // 2]
            ax.plot(x, curve(values[mid]), color=color, lw=1.2,
                    drawstyle="steps-post" if step else "default",
                    label=f"{100 * quantiles[mid]:g}%")

        ax.set_xlabel("Lookback time [Gyr]")
        ax.set_ylabel(name.replace("_", " ").capitalize())
        ax.legend(frameon=False, fontsize="small")
        return ax

    def make_figure(self, figsize=(6, 6), **kwargs):
        """Make a figure with the percentiles of all quantities.

        Parameters
        ----------
        figsize : tuple of float
            Figure size in inches.
        **kwargs
            Passed to :meth:`_plot_percentiles`.
        """
        n_rows = 3 if self.metallicity is not None else 2
        fig, axes = plt.subplots(n_rows, 1, figsize=figsize, sharex=True)
        ax = axes[0]
        # symlog: the first bin starts at a lookback time of 0
        ax.set_xscale("symlog", linthresh=1e-3, linscale=0.5)
        self._plot_percentiles("mass_fraction", ax=ax, **kwargs)
        ax.set_ylabel("Mass fraction")

        ax = axes[1]
        self._plot_percentiles("ssfr", ax=ax, **kwargs)
        ax.set_yscale("log")
        ylim = list(ax.get_ylim())
        ylim[0] = max(ylim[0], 1e-15)
        ylim[1] = min(ylim[1], 1e-7)
        ax.set_ylim(*ylim)
        ax.set_ylabel("sSFR [1/yr]")

        if self.metallicity is not None:
            ax = axes[2]
            self._plot_percentiles("metallicity", ax=ax, **kwargs)
            ax.set_yscale("log")
            ylims = list(ax.get_ylim())
            ylims[0] = max(ylims[0], 1e-4)
            ylims[1] = min(ylims[1], 1e-1)
            ax.set_ylim(*ylims)
            ax.set_ylabel("Metallicity [Z]")

        # Shared x axis and bands: one x label and one legend
        for ax in axes[:-1]:
            ax.set_xlabel("")
        for ax in axes[1:]:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
        axes[-1].set_xlabel("Lookback time [Gyr]")
        fig.tight_layout()
        return fig, axes

    # --- I/O -----------------------------------------------------------------
    def _percentile_table(self, name: str, prefix: str) -> Dict[str, np.ndarray]:
        values = self.percentile_values(name)
        if values is None:
            return {}
        return {f"{prefix}_{_percentile_label(q)}": values[i]
                for i, q in enumerate(self.percentiles)}

    def to_fits(self, include_samples: bool = False) -> fits.HDUList:
        """FITS representation.

        HDUs: ``PRIMARY`` (metadata), ``SFH_BINS`` (sSFR percentiles per
        lookback bin), ``SFH_EDGES`` (mass fraction and metallicity
        percentiles at the bin edges), ``SSFR_TAU`` (sSFR averaged over the
        last ``tau``). With ``include_samples``: ``SAMPLES`` (per-sample
        table) and the image HDUs ``SAMPLES_SSFR``, ``SAMPLES_MFRAC`` and
        ``SAMPLES_Z`` (rows = samples).
        """
        primary = fits.PrimaryHDU()
        header = primary.header
        header["BESTASFH"] = (1, "BESTA SFH reconstruction format version")
        header["NSAMPLES"] = (self.n_samples, "Posterior samples used")
        header["ESS"] = (float(effective_sample_size(self.weights))
                         if self.n_samples else 0.0, "Effective sample size")
        header["PCTILES"] = (",".join(f"{q:g}" for q in self.percentiles),
                             "Quantiles of the *_pNN columns")
        header["TIMEUNIT"] = ("Gyr", "Unit of lookback times and tau")
        header["SSFRUNIT"] = ("1/yr", "Unit of sSFR columns")
        for key, value in self.meta.items():
            if value is None:
                continue
            card = key.upper()[:8]
            header[card] = value if isinstance(value, (int, float, bool)) else str(value)

        bins = {"lookback_low": self.lookback_low,
                "lookback_high": self.lookback_high,
                "lookback_centre": self.lookback_centres}
        bins.update(self._percentile_table("ssfr", "ssfr"))
        edges = {"lookback": self.lookback_edges}
        edges.update(self._percentile_table("mass_fraction", "mass_fraction"))
        edges.update(self._percentile_table("metallicity", "metallicity"))
        taus = {"tau": self.taus}
        taus.update(self._percentile_table("ssfr_tau", "ssfr"))

        hdus = [primary,
                fits.BinTableHDU(Table(bins), name="SFH_BINS"),
                fits.BinTableHDU(Table(edges), name="SFH_EDGES"),
                fits.BinTableHDU(Table(taus), name="SSFR_TAU")]
        if include_samples:
            samples = Table() if self.sample_table is None else self.sample_table.copy()
            samples["weight"] = self.weights
            for j, tau in enumerate(self.taus):
                samples[f"ssfr_tau_{tau:g}"] = self.ssfr_tau[:, j]
            hdus.append(fits.BinTableHDU(samples, name="SAMPLES"))
            hdus.append(fits.ImageHDU(self.ssfr, name="SAMPLES_SSFR"))
            hdus.append(fits.ImageHDU(self.mass_fraction, name="SAMPLES_MFRAC"))
            if self.metallicity is not None:
                hdus.append(fits.ImageHDU(self.metallicity, name="SAMPLES_Z"))
        return fits.HDUList(hdus)

    def write_fits(self, path: str, *, include_samples: bool = False,
                   overwrite: bool = True) -> str:
        """Write :meth:`to_fits` to ``path`` and return the path."""
        path = os.path.expandvars(path)
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.to_fits(include_samples=include_samples).writeto(path, overwrite=overwrite)
        logger.info("SFH reconstruction written to %s", path)
        return path

    @classmethod
    def from_fits(cls, path: str) -> "SFHReconstruction":
        """Read a file written by :meth:`write_fits`.

        Without the sample HDUs, only the percentiles are available (the
        per-sample arrays are empty and :meth:`percentile_values` returns the
        stored percentiles).
        """
        with fits.open(os.path.expandvars(path)) as hdul:
            header = hdul[0].header
            percentiles = tuple(float(q) for q in str(header["PCTILES"]).split(","))
            bins = Table(hdul["SFH_BINS"].data)
            edges = Table(hdul["SFH_EDGES"].data)
            taus_tab = Table(hdul["SSFR_TAU"].data)
            has_samples = "SAMPLES" in hdul
            lookback_edges = np.asarray(edges["lookback"], dtype=float)
            taus = np.asarray(taus_tab["tau"], dtype=float)
            has_z = any(c.startswith("metallicity_") for c in edges.colnames)
            if has_samples:
                samples = Table(hdul["SAMPLES"].data)
                weights = np.asarray(samples["weight"], dtype=float)
                ssfr = np.asarray(hdul["SAMPLES_SSFR"].data, dtype=float)
                mfrac = np.asarray(hdul["SAMPLES_MFRAC"].data, dtype=float)
                z = (np.asarray(hdul["SAMPLES_Z"].data, dtype=float)
                     if "SAMPLES_Z" in hdul else None)
                ssfr_tau = np.column_stack(
                    [np.asarray(samples[f"ssfr_tau_{tau:g}"], dtype=float) for tau in taus]
                ) if taus.size else np.empty((weights.size, 0))
            else:
                samples, weights = None, np.empty(0)
                ssfr = np.empty((0, lookback_edges.size - 1))
                mfrac = np.empty((0, lookback_edges.size))
                z = np.empty((0, lookback_edges.size)) if has_z else None
                ssfr_tau = np.empty((0, taus.size))
            meta = {key.lower(): header[key] for key in
                    ("SFHMODEL", "MODULE", "REDSHIFT", "TODAY", "SOURCE", "SFHSPACE")
                    if key in header}
            rec = cls(lookback_edges=lookback_edges, percentiles=percentiles,
                      mass_fraction=mfrac, ssfr=ssfr, ssfr_tau=ssfr_tau, taus=taus,
                      weights=weights, metallicity=z, sample_table=samples, meta=meta)
            if not has_samples:
                def stored(table, prefix):
                    return np.vstack([np.asarray(table[f"{prefix}_{_percentile_label(q)}"],
                                                 dtype=float) for q in percentiles])
                rec._percentile_cache["ssfr"] = stored(bins, "ssfr")
                rec._percentile_cache["mass_fraction"] = stored(edges, "mass_fraction")
                rec._percentile_cache["ssfr_tau"] = stored(taus_tab, "ssfr")
                if has_z:
                    rec._percentile_cache["metallicity"] = stored(edges, "metallicity")
        return rec


def reconstruct_sfh(
    table: Table,
    ini: Mapping[str, Any],
    values: Optional[Mapping[str, Mapping[str, Any]]] = None,
    *,
    module_name: Optional[str] = None,
    lookback_edges: Optional[Sequence[float]] = None,
    n_bins: int = 40,
    min_lookback: float = 1e-3,
    taus: Sequence[float] = (0.01, 0.1, 1.0),
    percentiles: Sequence[float] = (0.05, 0.16, 0.5, 0.84, 0.95),
    weight_key: str = "weight",
    posterior_key: str = "post",
    burn_in: int = 0,
    nwalkers: int = 1,
    max_samples: Optional[int] = None,
    seed: Optional[int] = 0,
    parameter_prefix: str = "--",
) -> SFHReconstruction:
    """Evaluate the posterior SFHs of a run on a grid of lookback times.

    Parameters
    ----------
    table : astropy.table.Table
        Results table. Latent SFH parameters are converted to physical space
        first (logged).
    ini : dict
        Run configuration (e.g. :attr:`besta.io.Reader.ini`).
    values : dict, optional
        Parameter values (priors) of the run, used for the fixed SFH
        parameters that are not columns of ``table`` (e.g.
        :attr:`besta.io.Reader.ini_values`).
    module_name : str, optional
        Module defining the SFH (default: the first one).
    lookback_edges : sequence of float, optional
        Lookback-time bin edges in Gyr (increasing, starting at 0). Default:
        :func:`default_lookback_edges` with ``n_bins`` and ``min_lookback``.
    taus : sequence of float
        Timescales [Gyr] for the sSFR averaged over the last ``tau``.
    percentiles : sequence of float
        Quantiles in [0, 1].
    weight_key : str
        Column with sample weights (nested samplers); uniform weights are used
        if it is missing.
    burn_in, nwalkers : int
        Discard the first ``burn_in`` samples of each walker.
    max_samples : int, optional
        Use a random subset of at most this many samples (faster). With a
        weight column, the subset is drawn in proportion to the weights (with
        replacement) and then equally weighted.
    seed : int, optional
        Seed of the random subset.

    Returns
    -------
    SFHReconstruction
    """
    candidates = find_sfh_modules(ini)
    if module_name is None:
        if not candidates:
            raise ValueError("No pipeline module with an 'SFHModel' option was found.")
        module_name = candidates[0]
    model = _physical_sfh_model(ini, module_name)
    table = _table_to_physical(table, ini, module_name)
    if burn_in > 0:
        table = io.burn_table(table, nwalkers=nwalkers, burn_in=burn_in)

    today = float(model.today.to_value("Gyr"))
    if lookback_edges is None:
        edges = default_lookback_edges(today, n_bins=n_bins, min_lookback=min_lookback)
    else:
        edges = np.asarray(lookback_edges, dtype=float)
        if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError("lookback_edges must be a strictly increasing 1D array.")
        if edges[0] != 0 or edges[-1] > today:
            raise ValueError(f"lookback_edges must start at 0 and end before "
                             f"{today:.4g} Gyr (age of the Universe at the source).")
    taus = np.atleast_1d(np.asarray(taus, dtype=float))
    if np.any((taus <= 0) | (taus > today)):
        raise ValueError("taus must lie in (0, today].")

    # Columns of the SFH section and fixed values from the priors
    sect = model.sect_name
    prefix = f"{sect}{parameter_prefix}".lower()
    columns = {name[len(prefix):]: name for name in table.colnames
               if name.lower().startswith(prefix)}
    fixed = {}
    for section, params in (values or {}).items():
        if section.lower() != sect.lower():
            continue
        for key, value in params.items():
            if key.lower() in columns:
                continue
            if not _is_fixed_value(value):
                raise KeyError(f"Free parameter '{sect}--{key}' is not in the table.")
            fixed[key.lower()] = value

    n_rows = len(table)
    if weight_key in table.colnames:
        weights_all = _as_float_array(table[weight_key])
        weight_source = weight_key
    else:
        weights_all = np.ones(n_rows)
        weight_source = "uniform"
    rows = np.arange(n_rows)
    if max_samples is not None and n_rows > max_samples:
        rng = np.random.default_rng(seed)
        if weight_source == "uniform":
            rows = np.sort(rng.choice(n_rows, size=int(max_samples), replace=False))
            logger.info("Using a random subset of %d of %d samples.", rows.size, n_rows)
        else:
            # Resample in proportion to the weights (with replacement) and
            # treat the draws as equally weighted: re-using the weights would
            # count them twice.
            p = np.nan_to_num(weights_all, nan=0.0, posinf=0.0).clip(min=0.0)
            if p.sum() <= 0:
                raise ValueError(f"Column '{weight_key}' has no positive weights.")
            rows = np.sort(rng.choice(n_rows, size=int(max_samples), replace=True,
                                      p=p / p.sum()))
            weights_all = np.ones(n_rows)
            weight_source = f"resampled:{weight_key}"
            logger.info("Resampled %d of %d samples in proportion to '%s'.",
                        rows.size, n_rows, weight_key)

    arrays = {key: _as_float_array(table[col]) for key, col in columns.items()}
    times = today - edges[::-1]                      # increasing cosmic time
    # One PST call per sample: bin edges followed by the tau limits
    eval_times = np.concatenate((times, today - taus))
    n_edges = edges.size
    has_z = hasattr(model.model, "ism_metallicity")

    def as_values(x):
        return np.asarray(getattr(x, "value", x), dtype=float)

    mfrac, ssfr, zhist, ssfr_tau, used = [], [], [], [], []
    for row in rows:
        params = dict(fixed)
        params.update({key: float(values_[row]) for key, values_ in arrays.items()})
        if not all(np.isfinite(v) for v in params.values() if isinstance(v, float)):
            continue
        status, _ = model.parse_free_params(params)
        if not status:
            continue
        mass = as_values(model.model.stellar_mass_formed(eval_times))
        mass_today = mass[n_edges - 1]
        if not np.isfinite(mass_today) or mass_today <= 0:
            continue
        frac = mass[:n_edges][::-1] / mass_today       # at edges, lookback order
        mfrac.append(frac)
        ssfr.append((frac[:-1] - frac[1:]) / (np.diff(edges) * 1e9))
        ssfr_tau.append((1.0 - mass[n_edges:] / mass_today) / (taus * 1e9))
        if has_z:
            zhist.append(as_values(model.model.ism_metallicity(times))[::-1])
        used.append(row)

    used = np.asarray(used, dtype=int)
    if used.size < rows.size:
        logger.warning("%d of %d samples could not be evaluated and were skipped.",
                       rows.size - used.size, rows.size)
    sample_table = Table()
    for key, col in columns.items():
        sample_table[col] = _as_float_array(table[col])[used]
    for col in (posterior_key, "extra--stellar_mass"):
        if col in table.colnames:
            sample_table[col] = _as_float_array(table[col])[used]

    meta = {"sfhmodel": type(model).__name__, "module": module_name,
            "redshift": float(getattr(model, "redshift", 0.0) or 0.0),
            "today": today, "weights": weight_source, "sfhspace": "physical"}
    rec = SFHReconstruction(
        lookback_edges=edges,
        percentiles=tuple(float(q) for q in percentiles),
        mass_fraction=np.array(mfrac).reshape(-1, n_edges),
        ssfr=np.array(ssfr).reshape(-1, n_edges - 1),
        ssfr_tau=np.array(ssfr_tau).reshape(-1, taus.size),
        taus=taus,
        weights=weights_all[used],
        metallicity=np.array(zhist).reshape(-1, n_edges) if has_z else None,
        sample_table=sample_table,
        meta=meta,
    )
    logger.info("Reconstructed %d SFHs of model %s on %d lookback bins.",
                rec.n_samples, meta["sfhmodel"], n_edges - 1)
    return rec


def reconstruct_sfh_from_reader(reader, **kwargs) -> SFHReconstruction:
    """:func:`reconstruct_sfh` for a :class:`besta.io.Reader`."""
    table = getattr(reader, "_results_table", None)
    if table is None:
        reader.load_results()
        table = reader.results_table
    kwargs.setdefault("values", getattr(reader, "ini_values", None))
    rec = reconstruct_sfh(table, reader.ini, **kwargs)
    rec.meta.setdefault("source", getattr(reader, "results_file", None))
    return rec


def reconstruct_sfh_from_file(results_path: str, **kwargs) -> SFHReconstruction:
    """:func:`reconstruct_sfh` for a CosmoSIS text results file.

    The configuration and the parameter values are read from the file itself.
    """
    table = io.read_results_file(results_path)
    ini = io.Reader.read_ini_file_from_results(results_path)
    kwargs.setdefault("values", _read_embedded_values(results_path))
    rec = reconstruct_sfh(table, ini, **kwargs)
    rec.meta["source"] = results_path
    return rec


# -----------------------------------------------------------------------------
# Main summarization function
# -----------------------------------------------------------------------------

def summarize_results(
    table: Table,
    *,
    output_fits: Optional[str] = None,
    output_json: Optional[str] = None,
    nwalkers: Optional[int] = 1,
    burn_in: int = 0,
    parameter_prefix: str = "--",
    posterior_key: str = "post",
    use_posterior_weights: bool = False,
    parameter_keys: Optional[Sequence[str]] = None,
    percentiles: Sequence[float] = (0.05, 0.16, 0.5, 0.84, 0.95),
    compute_1d: bool = True,
    compute_2d: bool = False,
    parameter_key_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    pdf_bins_1d: int = 500,
    pdf_bins_2d: int = 80,
    estimate_evidence: bool = False,
    evidence_method: str = "laplace",
    logprior_key: str = "prior",
    loglike_key: Optional[str] = None,
    hme_trim_frac: float = 0.01,
    kde_1d: bool = True,
    kde_2d: bool = True,
    extra_info: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    sfh_model: Optional[Any] = None,
) -> ResultsSummary:
    """
    Summarize posterior results from a samples table into a ResultsSummary.

    Parameters
    ----------
    table : astropy.table.Table
        Input results table.
    output_fits : str, optional
        If provided, writes a FITS file with summary products.
    output_json : str, optional
        If provided, writes a JSON file with summary products.
    parameter_prefix : str
        Delimiter/prefix used in parameter names (default: "--").
    posterior_key : str
        Name of the log-posterior column (default: "post").
    parameter_keys : list of str, optional
        Explicit list of parameter columns to use. If None, selects columns
        containing parameter_prefix.
    percentiles : sequence of float
        Quantiles in [0,1].
    HDI intervals are computed at fixed masses of 68% and 95%.
    compute_1d / compute_2d : bool
        Enable 1D/2D PDF products.
    parameter_key_pairs : list of (key1, key2), required if compute_2d=True
    pdf_bins_1d / pdf_bins_2d : int
        Number of bins/grid size for PDFs.
    kde_1d / kde_2d : bool
        Prefer KDE; fallback to histogram if KDE fails.
    extra_info : dict
        Arbitrary metadata included in exports.
    verbose : bool
        Print progress.
    sfh_model : :class:`besta.sfh.SFHBase`, optional
        If given, latent SFH parameters are converted to physical values
        before summarizing (see :func:`latent_to_physical_table` and
        :func:`build_sfh_model`). No effect if the model does not use
        transforms or the table is already in physical space.

    Returns
    -------
    ResultsSummary
    """
    if posterior_key not in table.colnames:
        raise KeyError(f"posterior_key='{posterior_key}' not in table.")

    if sfh_model is not None:
        table = latent_to_physical_table(
            table, sfh_model, parameter_prefix=parameter_prefix)

    keys = io._select_parameter_keys(
        table, parameter_prefix=parameter_prefix, parameter_keys=parameter_keys)

    if burn_in == "auto":
        if nwalkers is None:
            raise ValueError("nwalkers must be provided for auto burn-in estimation.")
        # Reshape table into (nsamples, nwalkers, nparams)
        logger.info("Estimating burn-in automatically from MCMC chains with nwalkers=%d.", nwalkers)
        nsamples_guess = len(table) // nwalkers
        if nsamples_guess * nwalkers != len(table):
            raise ValueError(f"Unrecognized number of samples: {len(table)} is not divisible by nwalkers={nwalkers}.")
        # Convert the table into MCMC chains
        flat_chains = np.vstack([_as_float_array(table[k]) for k in keys]).T  # shape (nrows, nparams)
        chains = flat_chain_to_walkers(flat_chains, nwalkers=nwalkers, nsamples=nsamples_guess)
        burn_in = auto_burning_results(chains)
        logger.info("Auto burn-in estimated as %d samples.", burn_in)
    if burn_in > 0:
        logger.info("Burning-in first %d samples from each walker.", burn_in)
        table = io.burn_table(table, nwalkers=nwalkers, burn_in=burn_in)

    extra_info = extra_info or {}
    extra_info["burn_in"] = burn_in
    if _SFH_SPACE_META_KEY in table.meta:
        extra_info["sfhspace"] = table.meta[_SFH_SPACE_META_KEY]

    # Extract and filter samples
    logpost_all = _as_float_array(table[posterior_key])
    # Finite mask across posterior and all selected parameters
    mask = np.isfinite(logpost_all)

    if not np.any(mask):
        raise ValueError("No finite samples after masking posterior/parameters.")

    logpost = logpost_all[mask]
    # Stabilized weights from log-posterior
    max_lp = np.max(logpost)
    if use_posterior_weights:
        w = np.exp(logpost - max_lp)
        w = normalize_weights(w)
    else:
        w = np.ones_like(logpost, dtype=float)
    # Effective sample size
    extra_info["ess"] = effective_sample_size(w)
    # Samples matrix (D, N)
    samples = np.vstack([_as_float_array(table[k])[mask] for k in keys])
    # Filter NaN in samples
    finite_mask = np.isfinite(samples).all(axis=0)
    logpost = logpost[finite_mask]
    w = w[finite_mask]
    samples = samples[:, finite_mask]
    
    npar, nsamp = samples.shape

    map_idx = np.argmax(logpost)
    map_vec = samples[:, map_idx]

    mean_vec = weighted_mean(samples, w, axis=1)
    cov = weighted_covariance(samples, w, unbiased=False)
    corr = covariance_to_correlation(cov)

    # Names/sections
    sections = []
    names = []
    for k in keys:
        sect, nm = io._split_param_key(k, prefix=parameter_prefix)
        sections.append(sect)
        names.append(nm)

    if verbose:
        logger.info("Summarizing: %s parameters, %s samples (filtered).", npar, nsamp)
        logger.info("Max logpost = %.6g", max_lp)

    # Percentiles per parameter + interpolate logpost at those quantiles (via weighted CDF)
    pct = np.asarray(percentiles, dtype=float)
    if np.any((pct < 0) | (pct > 1)):
        raise ValueError("percentiles must be in [0,1].")

    pct_vals = np.full((npar, pct.size), np.nan, dtype=float)
    pct_lp = np.full((npar, pct.size), np.nan, dtype=float)
    hdi_68 = {}
    hdi_95 = {}
    map_1d = {}
    pdf1d: Dict[str, Dict[str, np.ndarray]] = {}

    for i, (name, sect) in enumerate(zip(names, sections)):
        nm = ".".join([sect, name])
        x = samples[i, :]
        # Weighted quantiles
        pct_vals[i, :] = weighted_quantile(x, w, pct)
        idx = np.argsort(x)
        ws = w[idx]
        cdf = np.cumsum(ws)
        cdf /= cdf[-1]
        lps = logpost[idx]
        pct_lp[i, :] = np.interp(pct, cdf, lps)
        # Highest density intervals (HDI) for 68% and 95%
        hdi_68[nm] = weighted_hdi(samples[i, :], w, mass=0.68, max_intervals=2)
        hdi_95[nm] = weighted_hdi(samples[i, :], w, mass=0.95, max_intervals=2)
        # 1D PDFs
        if compute_1d:
            edges, hist_pdf = histogram_pdf_1d(samples[i, :], w, bins=pdf_bins_1d)
            centers = 0.5 * (edges[:-1] + edges[1:])
            d = {"grid": centers, "edges": edges, "hist_pdf": hist_pdf}
            if kde_1d:
                d["kde_pdf"] = kde_pdf_1d(samples[i, :], w, edges)
                n_maxima, maxima_x, maxima_val = check_multimodal_pdf(centers, d["kde_pdf"])
            else:
                bins = (edges[:-1] + edges[1:]) / 2
                n_maxima, maxima_x, maxima_val = check_multimodal_pdf(bins, hist_pdf)
            d["n_maxima"] = n_maxima
            d["map"] = maxima_x
            d["map_values"] = maxima_val
            map_1d[nm] = maxima_x
            pdf1d[nm] = d

    # 2D PDFs
    pdf2d: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    if compute_2d:
        if parameter_key_pairs is None:
            raise ValueError("compute_2d=True requires parameter_key_pairs.")
        # Convert full keys to indices
        # TODO: account for section names
        key_to_idx = {k: i for i, k in enumerate(keys)}
        for k0, k1 in parameter_key_pairs:
            if k0 not in key_to_idx or k1 not in key_to_idx:
                raise KeyError(f"Pair ({k0}, {k1}) not in selected parameter keys:", key_to_idx)
            i0 = key_to_idx[k0]
            i1 = key_to_idx[k1]
            n0 = names[i0]
            n1 = names[i1]

            xg, yg, pdf = kde_or_hist_pdf_2d(
                samples[i0, :],
                samples[i1, :],
                w,
                bins=pdf_bins_2d,
                use_kde=kde_2d,
            )
            # pdf returned as (ny, nx) corresponding to yg, xg
            frac = enclosed_fraction_map(pdf)
            pdf2d[(n0, n1)] = {
                "xgrid": xg,
                "ygrid": yg,
                "pdf": pdf,
                "enclosed_fraction": frac,
            }

    summary = ResultsSummary(
        parameter_keys=list(keys),
        posterior_key=posterior_key,
        parameter_sections=sections,
        parameter_names=names,
        n_samples=int(nsamp),
        map_index=int(map_idx),
        weights=w,
        logpost=logpost,
        samples=samples,
        map=map_vec,
        mean=mean_vec,
        covariance=cov,
        correlation=corr,
        percentiles=list(map(float, pct.tolist())),
        percentiles_values=pct_vals,
        percentiles_logpost=pct_lp,
        hdi_intervals_68=hdi_68,
        hdi_intervals_95=hdi_95,
        map_1d=map_1d,
        pdf_1d=pdf1d,
        pdf_2d=pdf2d,
        extra_info=dict(extra_info) if extra_info is not None else {},
    )

    if estimate_evidence:
        summary.estimate_evidence_from_table(
            table,
            logprior_key=logprior_key,
            loglike_key=loglike_key,   # None -> will reconstruct from post-prior
            method=evidence_method,
            hme_trim_frac=hme_trim_frac,
            parameter_prefix=parameter_prefix,
        )

    if output_fits is not None:
        summary.write_fits(output_fits, overwrite=True)
    if output_json is not None:
        summary.write_json(output_json, overwrite=True)

    return summary


# -----------------------------------------------------------------------------
# Convenience wrappers for file-based workflows
# -----------------------------------------------------------------------------

def summarize_results_file(
    results_path: str,
    *,
    output_fits: Optional[str] = None,
    output_json: Optional[str] = None,
    delimiter: str = "\t",
    physical: bool = True,
    **kwargs,
) -> ResultsSummary:
    """Read a results file and summarize it (passes kwargs to summarize_results).

    Latent SFH parameters are converted to physical space first (see
    :func:`load_physical_results`) unless ``physical=False``.
    """
    if physical:
        tab = load_physical_results(results_path)
    else:
        tab = io.read_results_file(results_path, delimiter=delimiter)
    return summarize_results(
        tab, output_fits=output_fits, output_json=output_json, **kwargs)

def summarize_results_cosmosis(
    results_path: str,
    burn_in: int = 0,
    output_fits: Optional[str] = None
    ):

    processor = run_cosmosis_postprocess([results_path], no_plots=True, burn=burn_in)

    tables = {}
    for key, output in processor.outputs.items():
        if hasattr(output, "value") and key != "citations":
            try:
                tables[key] = output.value.to_astropy()
            except Exception as e:
                logger.warning("Failed to convert output '%s' to astropy table.", key)
                logger.debug("Error: %s", e)
                continue

    # Save all tables in a FITS
    if output_fits is not None:
        hdus = [fits.PrimaryHDU()]
        for k, t in tables.items():
            hdus.append(fits.BinTableHDU(t, name=f"{k}"))
        fits.HDUList(hdus).writeto(output_fits, overwrite=True)

    return tables

def compute_chain_percentiles(
    chain_results: Mapping[str, Any],
    *,
    pct: Sequence[float] = (0.16, 0.5, 0.84),
    weight_key: str = "weight",
    parameter_prefix: str = "--",
) -> Dict[str, np.ndarray]:
    """
    Compute weighted quantiles for chain-like results dict.

    Expects chain_results[param] arrays and chain_results[weight_key] weights.
    """
    if weight_key not in chain_results:
        raise KeyError(f"'{weight_key}' not present in chain_results.")

    w = _as_float_array(chain_results[weight_key]).ravel()
    out: Dict[str, np.ndarray] = {}
    for par, vals in chain_results.items():
        if par == weight_key:
            continue
        if parameter_prefix not in par:
            continue
        x = _as_float_array(vals).ravel()
        out[par] = weighted_quantile(x, w, pct)
    return out


def make_plot_chains(
    chain_results: Mapping[str, Any],
    *,
    truth_values: Optional[Mapping[str, float]] = None,
    weight_key: str = "weight",
    parameter_prefix: str = "--",
    outdir: Optional[str] = None,
    show: bool = False,
    dpi: int = 200,
) -> List[str]:
    """
    Plot simple trace + weighted histogram per parameter.

    Parameters
    ----------
    chain_results : dict-like
        Contains arrays for parameters and `weight_key`.
    truth_values : dict, optional
        Mapping parameter name -> truth value.
    outdir : str, optional
        If provided, saves PNGs to outdir and returns file paths.
        If None, returns empty list and only shows/creates figures.
    """
    if weight_key not in chain_results:
        raise KeyError(f"'{weight_key}' not present in chain_results.")

    w = _as_float_array(chain_results[weight_key]).ravel()
    paths: List[str] = []
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    for par, vals in chain_results.items():
        if par == weight_key:
            continue
        if parameter_prefix not in par:
            continue

        x = _as_float_array(vals).ravel()
        truth = np.nan
        if truth_values is not None and par in truth_values:
            truth = float(truth_values[par])

        fig = plt.figure(constrained_layout=True, figsize=(7, 3))
        ax = fig.add_subplot(111)
        ax.plot(x, ",", c="k", alpha=0.6)
        if np.isfinite(truth):
            ax.axhline(truth, c="r", lw=1.0)

        inax = ax.inset_axes((1.02, 0.0, 0.45, 1.0))
        inax.hist(x[np.isfinite(x)], weights=normalize_weights(w[np.isfinite(x)]), bins=60)
        if np.isfinite(truth):
            inax.axvline(truth, c="r", lw=1.0)
        inax.set_yticks([])

        ax.set_title(par)
        ax.set_xlabel("step")
        ax.set_ylabel(par)

        if outdir is not None:
            safe = par.replace("/", "_").replace(" ", "_").replace(":", "_")
            fp = os.path.join(outdir, f"chain_{safe}.png")
            fig.savefig(fp, dpi=dpi, bbox_inches="tight")
            paths.append(fp)

def photoz_metrics(z_true: np.ndarray, z_est: np.ndarray) -> dict:
    """
    Standard photo-z metrics using delta z over 1+z.

    Returns
    -------
    out : dict
        Keys: bias, nmad, outlier, rmse.
    """
    d = (z_est - z_true) / (1.0 + z_true)
    med = np.nanmedian(d)
    nmad = 1.48 * np.nanmedian(np.abs(d - med))
    outlier = float(np.mean(np.abs(d) > 0.15))
    rmse = float(np.sqrt(np.nanmean(d ** 2)))
    return {"bias": float(med), "nmad": float(nmad),
            "outlier": outlier, "rmse": rmse}

def specz_posterior(path: str, pct_val=[0.16, 0.5, 0.84]) -> dict:
    """Load spectral redshift posterior from a file.

    Parameters
    ----------
    path : str
        Path to the posterior file.
    pct_val : list of float
        Percentiles to compute (default: 16, 50, 84).
   
    Returns
    -------
    results : dict
        Keys: pct, mean, var, modes, mode_loglike, mode_log_amplitude.
    """
    z, loglike = np.loadtxt(path, dtype=np.float64, skiprows=1, unpack=True)

    # sort by z
    idx = np.argsort(z)
    z = z[idx]
    loglike = loglike[idx]
    # Renormalize to get a proper PDF
    loglike -= np.nanmax(loglike)
    like = np.exp(loglike)
    pdf = like / np.trapz(like, z)

    pct = weighted_quantile(z, like, pct_val)
    mean = weighted_mean(z, like)
    var = weighted_covariance(z[None, :], like, unbiased=False)[0, 0]
    # Analyze multimodality (simple local maxima)
    modes = []
    modes_loglike = []
    modes_idx = []  # list of list of indices corresponding to modes
    modes_pct = []
    modes_mean = []
    modes_var = []
    modes_log_amplitude = []
    modes_evidence = []
    idx_cont = []  # to track indices contributing to modes for continuum estimation

    # Step 1: find local minima
    for i in range(1, len(z) - 1):
        if loglike[i] < loglike[i - 1] and loglike[i] < loglike[i + 1]:
            idx_cont.append(i)

    # Step 2: characterise local maxima and assign mode indices
    start = 0
    for i in range(len(idx_cont) + 1):  # add end index to capture last segment
        if i < len(idx_cont):
            segment_idx = range(start, idx_cont[i])
        else:
            segment_idx = range(start, len(z))
        if len(segment_idx) == 0:
            continue
        # Find local maximum in this segment
        seg_loglike = loglike[segment_idx]
        max_idx_in_seg = np.argmax(seg_loglike)
        global_idx = segment_idx[max_idx_in_seg]
        modes.append(z[global_idx])
        modes_loglike.append(loglike[global_idx])
        modes_idx.append(list(segment_idx))

        # mode quantities
        mode_like = np.exp(seg_loglike - loglike[global_idx])
        mode_mean = weighted_mean(z[segment_idx], mode_like)
        mode_var = weighted_covariance(
            z[segment_idx][None, :], mode_like, unbiased=False
        )[0, 0]
        mode_pct = weighted_quantile(z[segment_idx], mode_like, pct_val)
        modes_mean.append(mode_mean)
        modes_var.append(mode_var)
        modes_pct.append(mode_pct)
        # mode loglike amplitude above local continuum
        left_cont = loglike[segment_idx[0]] if segment_idx[0] > 0 else loglike[0]
        right_cont = loglike[segment_idx[-1]] if segment_idx[-1] < len(z) - 1 else loglike[-1]

        cont_loglike = np.interp(z[global_idx], [z[segment_idx[0]], z[segment_idx[-1]]], [left_cont, right_cont])
        mode_log_amplitude = loglike[global_idx] - cont_loglike
        modes_log_amplitude.append(mode_log_amplitude)

        # mode evidence
        mode_evidence = np.trapz(pdf[segment_idx], z[segment_idx])
        modes_evidence.append(mode_evidence)

        if i < len(idx_cont):
            start = idx_cont[i] + 1

    if modes:
        modes = np.array(modes)
        modes_loglike = np.array(modes_loglike)
        modes_log_amplitude = np.array(modes_log_amplitude)
        modes_evidence = np.array(modes_evidence)
        modes_pct = np.array(modes_pct)
        modes_mean = np.array(modes_mean)
        modes_var = np.array(modes_var)

        # from matplotlib import pyplot as plt
        # plt.figure()
        # plt.plot(z, loglike, label="loglike")
        # plt.scatter(modes, modes_loglike, c=np.log(modes_evidence), label="modes")
        # plt.colorbar()
        # plt.legend()
    else:
        modes = np.array([z[np.argmax(loglike)]])
        modes_loglike = np.array([np.max(loglike)])
        modes_log_amplitude = np.array([0.0])
        modes_evidence = np.array([np.trapz(np.exp(loglike), z)])
        modes_evidence_contsub = np.array([0.0])
        modes_pct = np.array([0.0])
        modes_mean = np.array([0.0])
        modes_var = np.array([0.0])
        modes_idx = [list(range(len(z)))]

    results = {
        "pct": pct,
        "mean": mean,
        "var": var,
        "modes": modes,
        "mode_loglike": modes_loglike,
        "mode_log_amplitude": modes_log_amplitude,
        "mode_evidence": modes_evidence,
        "mode_pct": modes_pct,
        "mode_mean": modes_mean,
        "mode_var": modes_var,
        "mode_indices": modes_idx,
    }
    return results


def plot_chains(table, truth_values=None, output_dir=None, posterior_key="post"):
    """Make trace plots from an astropy Table containing chain results.

    Parameters
    ----------
    table : Table
        Astropy Table containing the chain results.
    truth_values : list of float, optional
        True values for the parameters (used for plotting).
    output_dir : str, optional
        Directory to save the output plots.
    posterior_key : str, optional
        Key to use for the posterior samples in the table.

    Returns
    -------
    all_figs : list of Figure
        List of all generated figures.
    """
    parameters = [par for par in table.colnames if "parameters" in par]
    if truth_values is None:
        truth_values = [np.nan] * len(parameters)
    all_figs = []
    #TODO: user-configurable posterior key limits
    if posterior_key is not None and posterior_key not in table.colnames:
        raise ValueError(f"Posterior key '{posterior_key}' not found in table columns.")
    else:
        logger.info("Using posterior key: %s", posterior_key)
        maxpost = np.nanmax(table[posterior_key])
        vmin = max(np.nanmin(table[posterior_key]), maxpost - 3.4)
        norm=plt.Normalize(vmin=vmin, vmax=maxpost)

    x_values = np.arange(len(table[parameters[0]]))
    for par, truth in zip(parameters, truth_values):
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(111)
        mappable = ax.scatter(x_values, table[par], s=0.5,
                              c=table[posterior_key], cmap="viridis",
                              norm=norm)
        cbar = fig.colorbar(mappable, ax=ax)
        cbar.set_label("log-posterior")
        ax.set_xlabel("Sample index")
        ax.set_ylabel(par.replace("parameters--", ""))
        if truth is not None and np.isfinite(truth):
            ax.axhline(truth, c="r")

        if output_dir is not None:
            fig.savefig(
                os.path.join(
                    output_dir,
                    f"chain_plot_{par.replace('parameters--', '')}.png",
                ),
                dpi=200,
                bbox_inches="tight",
            )
        all_figs.append(fig)
    return all_figs
