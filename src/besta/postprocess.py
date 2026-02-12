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
from astropy.io import fits
from astropy.table import Table, Column
from astropy import units as u
from matplotlib import pyplot as plt
from scipy import stats
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
    Weighted highest-density interval(s) for a 1D distribution, from samples.

    This returns one or more intervals whose union approximates the smallest
    region(s) containing `mass` of the probability, allowing multi-modality.

    Approach:
    - Sort samples by x.
    - Use the cumulative weights CDF.
    - Find the narrowest interval(s) [i, j] such that CDF[j]-CDF[i] >= mass.
    - Optionally return multiple intervals by iteratively masking out the best
      interval and repeating (approximate, but useful).

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
    Compute an HPD-style enclosed fraction map from a 2D density.

    Returns a map F with same shape as density such that:
    - Sorting pixels by density descending, F at a pixel is the cumulative
      probability mass enclosed at that density threshold.

    If edges are provided, uses pixel area dy*dx to compute mass; otherwise
    assumes constant pixel area.
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
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Ensure integrates to 1 (numerical)
    integral = np.nansum(pdf * dx)
    if integral > 0:
        pdf /= integral
    return centers, pdf

def kde_pdf_1d(
    x: np.ndarray,
    weights: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """
    KDE PDF on a provided grid; returns NaNs on failure.
    """
    x = _as_float_array(x).ravel()
    w = _as_float_array(weights).ravel()
    g = _as_float_array(grid).ravel()
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

# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def read_results_file(path: str, *, delimiter: str = "\t") -> Table:
    """
    Read a CosmoSIS-style text results file:
    - First line is a commented header starting with '#'
    - Columns are delimiter-separated
    - Remaining lines numeric

    Returns an Astropy Table with lowercase column names.
    """
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
    if not header.startswith("#"):
        raise ValueError("Expected first line header starting with '#'.")

    columns = header.strip("# \n").split(delimiter)
    matrix = np.atleast_2d(np.loadtxt(path))
    tab = Table()
    if matrix.size <= 1:
        return tab
    if matrix.shape[1] != len(columns):
        raise ValueError(
            f"Data has {matrix.shape[1]} columns but header lists {len(columns)}."
        )
    for i, c in enumerate(columns):
        tab[c.strip().lower()] = matrix[:, i]
    return tab

def _select_parameter_keys(
    table: Table,
    *,
    parameter_prefix: str = "--",
    parameter_keys: Optional[Sequence[str]] = None,
) -> List[str]:
    if parameter_keys is not None:
        keys = list(parameter_keys)
    else:
        keys = [k for k in table.colnames if parameter_prefix in k]
    if len(keys) == 0:
        raise ValueError("No parameter keys found/selected.")
    return keys

def _split_param_key(key: str, prefix: str = "--") -> Tuple[str, str]:
    """
    Split a parameter key into (section, name) using the delimiter/prefix.

    If the key cannot be split, returns ("", key).
    """
    if prefix in key:
        sect, name = key.split(prefix, 1)
        return sect, name
    return "", key

def _logsumexp_weighted(a: np.ndarray, w: np.ndarray) -> float:
    a = _as_float_array(a).ravel()
    w = _as_float_array(w).ravel()
    m = np.nanmax(a)
    return float(m + np.log(np.nansum(w * np.exp(a - m))))

def effective_sample_size(weights: np.ndarray) -> float:
    w = normalize_weights(weights)
    return float(1.0 / np.sum(w**2))

def laplace_logz(loglike_map: float, logprior_map: float, cov: np.ndarray) -> EvidenceEstimate:
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
    Robust harmonic-mean estimator:
      log Z = - log E_posterior[exp(-logL)].
    This is generally unstable; trimming helps but it remains a sanity-check only.
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
    method: str
    logz: float
    logz_err: Optional[float] = None
    details: Dict[str, Any] = None

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

        Supported:
        - method="laplace": uses MAP loglike/logprior + covariance
        - method="hme_robust": uses posterior expectation of 1/L (unstable; sanity-check)

        Inputs
        ------
        You said you have:
          post = loglike + logprior
          prior = logprior
        So if loglike_key is not present, we reconstruct loglike = post - prior.
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
                hdr[f"{tag}MN"] = float(self.mean[i])
            if i < self.map.size:
                hdr[f"{tag}MP"] = float(self.map[i])

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
        pct_hdr["HDIMASS"] = float(self.hdi_mass)
        for name, ivs in self.hdi_intervals.items():
            # store up to 2 intervals by default
            for j, (lo, hi) in enumerate(ivs[:2]):
                pct_hdr[f"{name[:6]}L{j}"] = float(lo)
                pct_hdr[f"{name[:6]}H{j}"] = float(hi)

        hdus.append(fits.BinTableHDU(t_pct, name="PERCENTILES", header=pct_hdr))

        # PDF1D table: store grid/pdf/kde per parameter as separate columns
        t_pdf1 = Table()
        for name, d in self.pdf_1d.items():
            t_pdf1[f"{name}_x"] = _as_float_array(d["grid"])
            t_pdf1[f"{name}_pdf"] = _as_float_array(d["hist_pdf"])
            t_pdf1[f"{name}_kde"] = _as_float_array(d.get("kde_pdf", np.full_like(d["grid"], np.nan)))

        if len(t_pdf1.colnames) > 0:
            hdus.append(fits.BinTableHDU(t_pdf1, name="PDF1D"))

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

    # -------------------------------------------------------------------------
    # Plot helpers (optional)
    # -------------------------------------------------------------------------

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
        """
        Lightweight corner plot without external dependencies.

        Diagonal: 1D hist PDFs (weighted)
        Off-diagonal: scatter of (subsampled) points colored by weight rank (simple)

        For serious usage, consider adding an optional dependency later (corner/arviz),
        but this is a decent built-in baseline.
        """
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

        fig, axes = plt.subplots(npar, npar, figsize=(2.2 * npar, 2.2 * npar), constrained_layout=True)

        for i in range(npar):
            for j in range(npar):
                ax = axes[i, j]
                if i == j:
                    x = self.samples[i, :]
                    xc, pdf = histogram_pdf_1d(x, self.weights, bins=bins)
                    ax.plot(xc, pdf, lw=1.2)
                    # mark mean/MAP
                    ax.axvline(self.mean[i], lw=1.0, alpha=0.8)
                    ax.axvline(self.map[i], lw=1.0, alpha=0.8)
                    ax.set_yticks([])
                elif i > j:
                    ax.scatter(S[j, :], S[i, :], s=2, alpha=0.25)
                else:
                    ax.axis("off")

                if i == npar - 1 and j <= i:
                    ax.set_xlabel(self.parameter_names[j])
                else:
                    ax.set_xticks([])
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

    keys = _select_parameter_keys(table, parameter_prefix=parameter_prefix, parameter_keys=parameter_keys)
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
    npar, nsamp = samples.shape

    map_idx = int(np.argmax(logpost))  # index within filtered arrays
    map_vec = samples[:, map_idx]

    mean_vec = weighted_mean(samples, w, axis=1)
    cov = weighted_covariance(samples, w, unbiased=False)
    corr = covariance_to_correlation(cov)

    # Names/sections
    sections = []
    names = []
    for k in keys:
        sect, nm = _split_param_key(k, prefix=parameter_prefix)
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
            grid, hist_pdf = histogram_pdf_1d(samples[i, :], w, bins=pdf_bins_1d)
            d = {"grid": grid, "hist_pdf": hist_pdf}
            if kde_1d:
                d["kde_pdf"] = kde_pdf_1d(samples[i, :], w, grid)
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
                raise KeyError(f"Pair ({k0}, {k1}) not in selected parameter keys.")
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
    tab = read_results_file(results_path, delimiter=delimiter)
    return summarize_results(tab, output_fits=output_fits, output_json=output_json, **kwargs)


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

        if show:
            plt.show()
        else:
            plt.close(fig)

    return paths
