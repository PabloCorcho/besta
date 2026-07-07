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

from astropy.io import fits
from astropy.table import Table, Column
from astropy import units as u

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
    cdf[-1] = 1.0
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
        # Work on contiguous segments of availability. This keeps interval definition meaningful.
        best = None
        best_width = np.inf

        # Identify contiguous runs
        avail = available_mask.astype(int)
        # runs: start indices where diff==1, end where diff==-1
        starts = np.where(np.diff(np.r_[0, avail]) == 1)[0]
        ends = np.where(np.diff(np.r_[avail, 0]) == -1)[0]

        for s, e in zip(starts, ends):
            # Consider sub-array xs[s:e], ws[s:e]
            sub_ws = ws[s:e]
            if sub_ws.size == 0:
                continue
            sub_cdf = np.cumsum(sub_ws)
            sub_cdf[-1] = np.sum(sub_ws)
            # Normalize to segment mass; but we want absolute mass, so compare to `mass` directly.
            # Since total mass across all samples is 1, segment mass might be < mass; skip then.
            if sub_cdf[-1] < mass:
                continue

            sub_cdf0 = np.concatenate([[0.0], sub_cdf])
            # Two-pointer to find minimal width interval >= mass in this segment
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
        # (We approximate by computing mass inside that interval.)
        m = np.sum(ws[(xs >= xs[i]) & (xs <= xs[j])])
        if m >= mass:
            # Good enough — returning single interval is typical.
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

def integrated_autocorrelation_time(chain, c=5.0, tol=50):
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


def auto_burning_results(chains, c=5.0, tol=50, kappa_act=3.0):
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
    burn = []
    burn_rel = []
    # loop over parameters
    nparams = chains.shape[2]
    for ith in range(nparams):
        tau, acf_mean, reliable = integrated_autocorrelation_time(
            chains[:, :, ith], c=c, tol=tol)
        if reliable and np.isfinite(tau):
            burn_rel.append(reliable)
        else:
            tau = np.ceil(kappa_act * tau)
            burn.append(int(tau) if np.isfinite(tau) else 0)

    if burn_rel:
        max_burn = np.nanmax(burn_rel) if burn_rel else 0
    else:
        logger.warning("No reliable autocorrelation time estimates found.")
        max_burn = np.nanmax(burn) if burn else 0
    return max_burn

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
    hdi_mass : mass used for HDI
    hdi_intervals : dict short_name -> list of (low, high)
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

    hdi_mass: float = 0.68
    hdi_intervals: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)

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
            "hdi_mass": float(self.hdi_mass),
            "hdi_intervals": {k: [list(iv) for iv in v] for k, v in self.hdi_intervals.items()},
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

        # Per-parameter header cards (best effort: keep short names)
        for i, (full_key, sect, name) in enumerate(zip(self.parameter_keys, self.parameter_sections, self.parameter_names)):
            # FITS keyword length limits: use e.g. P000NM, P000SC, P000MN, P000MP
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
            prim.header["LOGZ"] = float(self.evidence.logz)
            prim.header["LOGZMET"] = self.evidence.method[:20]

        hdus: List[fits.hdu.base.ExtensionHDU] = [prim]
        hdus.append(fits.ImageHDU(data=_as_float_array(self.covariance), header=hdr, name="COVARIANCE"))
        hdus.append(fits.ImageHDU(data=_as_float_array(self.correlation), name="CORRELATION"))

        # Percentiles table
        t_pct = Table()
        t_pct["percentile"] = np.asarray(self.percentiles, dtype=float)
        for i, name in enumerate(self.parameter_names):
            t_pct[f"{name}_val"] = _as_float_array(self.percentiles_values[i, :]) if self.percentiles_values.size else np.full(len(self.percentiles), np.nan)
            t_pct[f"{name}_logp"] = _as_float_array(self.percentiles_logpost[i, :]) if self.percentiles_logpost.size else np.full(len(self.percentiles), np.nan)

        # Add HDI intervals as header cards on the percentiles HDU (compact)
        pct_hdr = fits.Header()
        pct_hdr["HDIMASS"] = float(self.hdi_mass) if np.isfinite(self.hdi_mass) else "nan", "HDI mass"
        for name, ivs in self.hdi_intervals.items():
            # store up to 2 intervals by default
            for j, (lo, hi) in enumerate(ivs[:2]):
                pct_hdr[f"{name[:6]}L{j}"] = float(lo) if np.isfinite(lo) else "nan", "lower limit"
                pct_hdr[f"{name[:6]}H{j}"] = float(hi) if np.isfinite(hi) else "nan", "upper limit"

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
            # HDI
            if name in self.hdi_intervals:
                for lo, hi in self.hdi_intervals[name]:
                    ax.axvspan(lo, hi, alpha=0.15)
            ax.set_title(name)
            ax.set_xlabel(name)
            ax.set_ylabel("PDF")
            ax.legend()

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
    parameter_keys: Optional[Sequence[str]] = None,
    percentiles: Sequence[float] = (0.05, 0.16, 0.5, 0.84, 0.95),
    hdi_mass: float = 0.68,
    compute_1d: bool = True,
    compute_2d: bool = False,
    parameter_key_pairs: Optional[Sequence[Tuple[str, str]]] = None,
    pdf_bins_1d: int = 100,
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
    hdi_mass : float
        Target mass for HDI intervals.
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

    Returns
    -------
    ResultsSummary
    """
    if posterior_key not in table.colnames:
        raise KeyError(f"posterior_key='{posterior_key}' not in table.")

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

    # Extract and filter samples
    logpost_all = _as_float_array(table[posterior_key])
    # Finite mask across posterior and all selected parameters
    mask = np.isfinite(logpost_all)

    if not np.any(mask):
        raise ValueError("No finite samples after masking posterior/parameters.")

    logpost = logpost_all[mask]
    # Stabilized weights from log-posterior
    max_lp = np.max(logpost)
    w = np.exp(logpost - max_lp)
    w = normalize_weights(w)

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

    for i in range(npar):
        x = samples[i, :]
        # Weighted quantiles
        pct_vals[i, :] = weighted_quantile(x, w, pct)

        # For logpost at those percentiles: sort by x, build weighted cdf, and interp logpost along that ordering
        idx = np.argsort(x)
        xs = x[idx]
        ws = w[idx]
        cdf = np.cumsum(ws)
        cdf[-1] = 1.0
        # logpost along sorted x
        lps = logpost[idx]
        pct_lp[i, :] = np.interp(pct, cdf, lps)

    # HDI intervals
    hdi: Dict[str, List[Tuple[float, float]]] = {}
    for i, nm in enumerate(names):
        hdi[nm] = weighted_hdi(samples[i, :], w, mass=hdi_mass, max_intervals=2)

    # 1D PDFs
    pdf1d: Dict[str, Dict[str, np.ndarray]] = {}
    if compute_1d:
        for i, nm in enumerate(names):
            edges, hist_pdf = histogram_pdf_1d(samples[i, :], w, bins=pdf_bins_1d)
            centers = 0.5 * (edges[:-1] + edges[1:])
            d = {"grid": centers, "edges": edges, "hist_pdf": hist_pdf}
            if kde_1d:
                d["kde_pdf"] = kde_pdf_1d(samples[i, :], w, edges)
            pdf1d[nm] = d

    # 2D PDFs
    pdf2d: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    if compute_2d:
        if parameter_key_pairs is None:
            raise ValueError("compute_2d=True requires parameter_key_pairs.")
        # Convert full keys to indices
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
        hdi_mass=float(hdi_mass),
        hdi_intervals=hdi,
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
    **kwargs,
) -> ResultsSummary:
    """Read a results file and summarize it (passes kwargs to summarize_results)."""
    tab = io.read_results_file(results_path, delimiter=delimiter)
    return summarize_results(
        tab, output_fits=output_fits, output_json=output_json, **kwargs)


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

def weighted_quantiles(x: np.ndarray, w: np.ndarray,
                        qs: Sequence[float]) -> np.ndarray:
    """Compute weighted quantiles for a one-dimensional sample."""

    x = np.asarray(x); w = np.asarray(w)
    m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    if not m.any():
        return np.array([np.nan] * len(qs))
    x, w = x[m], w[m]
    order = np.argsort(x)
    x, w = x[order], w[order]
    cdf = np.cumsum(w)
    cdf = cdf / cdf[-1]
    return np.interp(qs, cdf, x)

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
