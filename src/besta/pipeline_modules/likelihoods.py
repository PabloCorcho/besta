"""Likelihood helpers for BESTA pipeline modules.

The module exposes fast, backend-selectable likelihood functions for spectra and
photometry. Spectral modules use an inverse-variance Gaussian likelihood with no
limit handling. Photometry modules use the more general likelihood that can
handle optional upper/lower limits, with a fast no-limits path when requested.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.stats import norm

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency guard
    NUMBA_AVAILABLE = False
    njit = None  # type: ignore[assignment]


_LOG_2PI = math.log(2.0 * math.pi)


def _validate_spectra_inputs(data, model, ivar):
    if data.shape != model.shape or data.shape != ivar.shape:
        raise ValueError("data, model, ivar must have the same shape (ivar is per-datum inverse variance).")
    if np.any(ivar < 0):
        raise ValueError("All ivar entries must be >= 0 (inverse variance).")


def _validate_photometry_inputs(data, model, var, weights, is_upper, is_lower):
    if data.shape != model.shape or data.shape != var.shape:
        raise ValueError("data, model, var must have the same shape (var is per-datum variance).")
    if np.any(var <= 0):
        raise ValueError("All var entries must be > 0 (variance).")

    if is_upper is None:
        is_upper = np.zeros_like(data, dtype=bool)
    else:
        is_upper = np.asarray(is_upper, dtype=bool)

    if is_lower is None:
        is_lower = np.zeros_like(data, dtype=bool)
    else:
        is_lower = np.asarray(is_lower, dtype=bool)

    if is_upper.shape != data.shape or is_lower.shape != data.shape:
        raise ValueError("is_upper and is_lower must have the same shape as data/model.")
    if np.any(is_upper & is_lower):
        raise ValueError("A data point cannot be both an upper and a lower limit.")

    if weights is None:
        weights = np.ones_like(data, dtype=float)
        normalize = False
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != data.shape:
            raise ValueError("weights must have the same shape as data/model.")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")
        normalize = True

    return weights, is_upper, is_lower, normalize


def spectra_loglike_numpy(data, model, ivar):
    """Inverse-variance Gaussian log-likelihood for spectra."""
    _validate_spectra_inputs(data, model, ivar)
    chi2 = np.sum(ivar * (data - model) ** 2)
    return -0.5 * chi2


if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=False)
    def spectra_loglike_numba(data, model, ivar):
        chi2 = 0.0
        for i in range(data.size):
            residual = data[i] - model[i]
            chi2 += ivar[i] * residual * residual
        return -0.5 * chi2


    @njit(cache=True, fastmath=False)
    def photometry_loglike_numba_no_limits(data, model, var, weights, normalize, include_norm):
        total = 0.0
        wsum = 0.0
        for i in range(data.size):
            sigma = math.sqrt(var[i])
            z = (data[i] - model[i]) / sigma
            logp = -0.5 * z * z
            if include_norm:
                logp += -0.5 * (_LOG_2PI + math.log(var[i]))
            w = weights[i]
            total += logp * w
            if normalize:
                wsum += w
        if normalize:
            return total / wsum
        return total

else:  # pragma: no cover - exercised only when numba is missing

    def spectra_loglike_numba(*args, **kwargs):
        raise ImportError("numba is not available")

    def photometry_loglike_numba_no_limits(*args, **kwargs):
        raise ImportError("numba is not available")


def photometry_loglike_numpy(
    data,
    model,
    var,
    weights=None,
    is_upper=None,
    is_lower=None,
    include_norm=True,
):
    """Gaussian photometry likelihood with optional upper/lower limits."""
    weights, is_upper, is_lower, normalize = _validate_photometry_inputs(
        data, model, var, weights, is_upper, is_lower
    )

    if not np.any(is_upper | is_lower):
        sigma = np.sqrt(var)
        logp = -0.5 * ((data - model) / sigma) ** 2
        if include_norm:
            logp = logp - 0.5 * (np.log(2.0 * np.pi) + np.log(var))
        if normalize:
            wsum = np.sum(weights)
            if wsum <= 0:
                raise ValueError("Sum of weights must be > 0 for normalization.")
            return np.sum(logp * weights) / wsum
        return np.sum(logp * weights)

    sigma = np.sqrt(var)
    logp = np.empty_like(data, dtype=float)
    det = ~(is_upper | is_lower)

    if np.any(det):
        if include_norm:
            logp[det] = norm.logpdf(data[det], loc=model[det], scale=sigma[det])
        else:
            z = (data[det] - model[det]) / sigma[det]
            logp[det] = -0.5 * z**2

    if np.any(is_upper):
        z_u = (data[is_upper] - model[is_upper]) / sigma[is_upper]
        logp[is_upper] = norm.logcdf(z_u)

    if np.any(is_lower):
        z_l = (data[is_lower] - model[is_lower]) / sigma[is_lower]
        logp[is_lower] = norm.logsf(z_l)

    if normalize:
        wsum = np.sum(weights)
        if wsum <= 0:
            raise ValueError("Sum of weights must be > 0 for normalization.")
        return np.sum(logp * weights) / wsum

    return np.sum(logp * weights)


def make_spectra_loglike(method: str = "auto") -> Callable:
    """Return a spectra likelihood implementation for the requested backend."""
    method = (method or "auto").strip().lower()
    if method in {"auto", "numba"} and NUMBA_AVAILABLE:
        return spectra_loglike_numba
    if method in {"auto", "numpy"}:
        return spectra_loglike_numpy
    raise ValueError("Valid spectra likelihood methods are: auto, numpy, numba")


def make_photometry_loglike(method: str = "auto") -> Callable:
    """Return a photometry likelihood implementation for the requested backend."""
    method = (method or "auto").strip().lower()

    if method in {"auto", "numba"} and NUMBA_AVAILABLE:

        def _numba_or_numpy(
            data,
            model,
            var,
            weights=None,
            is_upper=None,
            is_lower=None,
            include_norm=True,
        ):
            if is_upper is None and is_lower is None:
                weights_arr = np.ones_like(data, dtype=float) if weights is None else np.asarray(weights, dtype=float)
                normalize = weights is not None
                if normalize and np.any(weights_arr < 0):
                    raise ValueError("weights must be non-negative.")
                return photometry_loglike_numba_no_limits(
                    np.asarray(data, dtype=float),
                    np.asarray(model, dtype=float),
                    np.asarray(var, dtype=float),
                    weights_arr,
                    normalize,
                    include_norm,
                )
            return photometry_loglike_numpy(
                data,
                model,
                var,
                weights=weights,
                is_upper=is_upper,
                is_lower=is_lower,
                include_norm=include_norm,
            )

        return _numba_or_numpy

    if method in {"auto", "numpy"}:
        return photometry_loglike_numpy

    raise ValueError("Valid photometry likelihood methods are: auto, numpy, numba")
