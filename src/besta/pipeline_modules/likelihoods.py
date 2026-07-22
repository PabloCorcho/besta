"""Likelihood helpers for BESTA pipeline modules.

The module contains likelihood functions for spectra and
photometry observables. Spectral modules use an inverse-variance Gaussian likelihood with no
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
    print("WARNING: numba is not available: falling back to numpy-based likelihoods."
    )
    njit = None  # type: ignore[assignment]

_LOG_2PI = math.log(2.0 * math.pi)


def _validate_spectra_inputs(data, model, ivar):
    if data.shape != model.shape or data.shape != ivar.shape:
        raise ValueError(
            "data, model, and ivar must have the same shape "
            "(ivar is per-datum inverse variance)."
        )

    if np.any(~np.isfinite(data)):
        raise ValueError("All spectral data entries must be finite.")

    if np.any(~np.isfinite(model)):
        raise ValueError("All spectral model entries must be finite.")

    if np.any(~np.isfinite(ivar)) or np.any(ivar < 0):
        raise ValueError(
            "All ivar entries must be finite and >= 0."
        )

    if not np.any(ivar > 0):
        raise ValueError(
            "At least one spectral datum must have positive inverse variance."
        )

def _validate_photometry_inputs(data, model, var, weights, is_upper, is_lower):
    data = np.asarray(data, dtype=float)
    model = np.asarray(model, dtype=float)
    var = np.asarray(var, dtype=float)

    if data.shape != model.shape or data.shape != var.shape:
        raise ValueError(
            "data, model, and var must have the same shape "
            "(var is per-datum variance)."
        )

    if np.any(~np.isfinite(data)):
        raise ValueError("All data entries must be finite.")
    if np.any(~np.isfinite(model)):
        raise ValueError("All model entries must be finite.")
    if np.any(~np.isfinite(var)) or np.any(var <= 0):
        raise ValueError("All var entries must be finite and > 0.")

    if is_upper is None:
        is_upper = np.zeros_like(data, dtype=bool)
    else:
        is_upper = np.asarray(is_upper, dtype=bool)

    if is_lower is None:
        is_lower = np.zeros_like(data, dtype=bool)
    else:
        is_lower = np.asarray(is_lower, dtype=bool)

    if is_upper.shape != data.shape or is_lower.shape != data.shape:
        raise ValueError(
            "is_upper and is_lower must have the same shape as data."
        )

    if np.any(is_upper & is_lower):
        raise ValueError(
            "A data point cannot be both an upper and a lower limit."
        )

    if weights is None:
        weights = np.ones_like(data, dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

        if weights.shape != data.shape:
            raise ValueError(
                "weights must have the same shape as data/model."
            )

        if np.any(~np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError(
                "weights must be finite and non-negative."
            )

    return data, model, var, weights, is_upper, is_lower


def spectra_loglike_numpy(
    data,
    model,
    ivar,
    include_norm=True,
):
    """Inverse-variance Gaussian log-likelihood for spectra.
    
    Parameters
    ----------
    data : array_like
        Observed spectral data.
    model : array_like
        Model spectral data.
    ivar : array_like
        Inverse variance of the observed data.
    include_norm : bool, optional
        Whether to include the normalization term in the log-likelihood.
        If True, the log-likelihood will include the term 0.5 * sum(log(ivar) - log(2*pi)),
        necessary for evidence estimation.
    
    Returns
    -------
    loglike : float
        The computed log-likelihood value.
    """
    # Ensure inputs are arrays of floats
    # data = np.asarray(data, dtype=float)
    # model = np.asarray(model, dtype=float)
    # ivar = np.asarray(ivar, dtype=float)

    # Validate format
    # _validate_spectra_inputs(data, model, ivar)

    valid = ivar > 0
    residual = data[valid] - model[valid]

    loglike = -0.5 * np.sum(
        ivar[valid] * residual**2
    )

    # This needs to be included if the evidence is estimated
    if include_norm:
        loglike += 0.5 * np.sum(
            np.log(ivar[valid]) - _LOG_2PI
        )

    return loglike

if NUMBA_AVAILABLE:
    # Internal method
    @njit(cache=True, fastmath=False)
    def _spectra_loglike_numba_kernel(
        data,
        model,
        ivar,
        include_norm,
    ):
        total = 0.0

        for i in range(data.size):
            if ivar[i] <= 0.0:
                continue

            residual = data[i] - model[i]
            total -= 0.5 * ivar[i] * residual * residual

            if include_norm:
                total += 0.5 * (
                    math.log(ivar[i]) - _LOG_2PI
                )

        return total

    # API method
    def spectra_loglike_numba(
        data,
        model,
        ivar,
        include_norm=True,
    ):
        """Fast numba-compiled inverse-variance Gaussian log-likelihood for spectra.

        Parameters
        ----------
        data : array_like
            Observed spectral data.
        model : array_like
            Model spectral data.
        ivar : array_like
            Inverse variance of the observed data.
        include_norm : bool, optional
            Whether to include the normalization term in the log-likelihood.
            If True, the log-likelihood will include the term 0.5 * sum(log(ivar) - log(2*pi)),
            necessary for evidence estimation.

        Returns
        -------
        loglike : float
            The computed log-likelihood value.
        
        See Also
        --------
        spectra_loglike_numpy : Numpy-based implementation of the same likelihood.
        """
        # data = np.asarray(data, dtype=float)
        # model = np.asarray(model, dtype=float)
        # ivar = np.asarray(ivar, dtype=float)

        # _validate_spectra_inputs(data, model, ivar)

        return _spectra_loglike_numba_kernel(
            data.ravel(),
            model.ravel(),
            ivar.ravel(),
            include_norm,
        )

    @njit(cache=True, fastmath=False)
    def _photometry_loglike_numba_no_limits_kernel(
        data,
        model,
        var,
        weights,
        include_norm,
    ):
        total = 0.0

        for i in range(data.size):
            residual2_over_var = (data[i] - model[i]) ** 2 / var[i]
            logp = -0.5 * residual2_over_var

            if include_norm:
                logp -= 0.5 * (math.log(var[i]) + _LOG_2PI)

            total += weights[i] * logp

        return total

    def photometry_loglike_numba_no_limits(
        data,
        model,
        var,
        weights,
        include_norm,
    ):
        """Fast numba-compiled photometry likelihood when no limits are present.
        
        Parameters
        ----------
        data : array_like
            Observed photometric data.
        model : array_like
            Model photometric data.
        var : array_like
            Variance of the observed data.
        weights : array_like
            Weights for each datum's log-likelihood contribution.
        include_norm : bool, optional
            Whether to include the normalization term in the log-likelihood.
        
        Returns
        -------
        loglike : float
            The computed log-likelihood value.
        """
        return _photometry_loglike_numba_no_limits_kernel(
            data,
            model,
            var,
            weights,
            include_norm,
        )

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
    """Inverse-variance Gaussian log-likelihood for photometry, with optional upper/lower limits.

    Parameters
    ----------
    data : array_like
        Observed photometric data.
    model : array_like
        Model photometric data.
    var : array_like
        Variance of the observed data.
    weights : array_like, optional
        Weights for each datum's log-likelihood contribution. If None, all weights are set to 1.
    is_upper : array_like of bool, optional
        Boolean array indicating which data points are upper limits. If None, all are treated as detected.
    is_lower : array_like of bool, optional
        Boolean array indicating which data points are lower limits. If None, all are treated as detected.
    include_norm : bool, optional
        Whether to include the normalization term in the log-likelihood.
    
    Returns
    -------
    loglike : float
        The computed log-likelihood value.
    """
    if weights is None:
        weights = np.ones_like(data, dtype=float)

    # Fast path: no limit masks provided.
    if is_upper is None and is_lower is None:
        residual2_over_var = (data - model) ** 2 / var
        logp = -0.5 * residual2_over_var

        if include_norm:
            logp -= 0.5 * (_LOG_2PI + np.log(var))

        return np.sum(weights * logp)

    if is_upper is None:
        is_upper = np.zeros_like(data, dtype=bool)

    if is_lower is None:
        is_lower = np.zeros_like(data, dtype=bool)

    sigma = np.sqrt(var)
    logp = np.empty_like(data, dtype=float)

    detected = ~(is_upper | is_lower)

    if np.any(detected):
        residual2_over_var = (
            data[detected] - model[detected]
        ) ** 2 / var[detected]

        logp[detected] = -0.5 * residual2_over_var

        if include_norm:
            logp[detected] -= 0.5 * (
                _LOG_2PI + np.log(var[detected])
            )

    if np.any(is_upper):
        z_upper = (
            data[is_upper] - model[is_upper]
        ) / sigma[is_upper]
        logp[is_upper] = norm.logcdf(z_upper)

    if np.any(is_lower):
        z_lower = (
            data[is_lower] - model[is_lower]
        ) / sigma[is_lower]
        logp[is_lower] = norm.logsf(z_lower)

    return np.sum(weights * logp)


def make_spectra_loglike(method: str = "auto") -> Callable:
    """Return a spectra likelihood implementation for the requested backend.
    
    Parameters
    ----------
    method : str, optional
        The backend to use for the likelihood computation. Options are:
        - "auto": Use numba if available, otherwise fall back to numpy.
        - "numba": Use the numba-compiled implementation (requires numba).
        - "numpy": Use the numpy-based implementation.
    
    Returns
    -------
    Callable
        A function that computes the spectra log-likelihood with the specified backend.
    """
    method = (method or "auto").strip().lower()
    if method in {"auto", "numba"} and NUMBA_AVAILABLE:
        return spectra_loglike_numba
    if method in {"auto", "numpy"}:
        return spectra_loglike_numpy
    raise ValueError("Valid spectra likelihood methods are: auto, numpy, numba")


def make_photometry_loglike(method: str = "auto") -> Callable:
    """Return a photometry likelihood implementation for the requested backend.
    
    Parameters
    ----------
    method : str, optional
        The backend to use for the likelihood computation. Options are:
        - "auto": Use numba if available, otherwise fall back to numpy.
        - "numba": Use the numba-compiled implementation (requires numba).
        - "numpy": Use the numpy-based implementation.

    Returns
    -------
    Callable
        A function that computes the photometry log-likelihood with the specified backend.
    """
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
            if weights is None:
                weights = np.ones_like(data, dtype=float)

            # Fast path: no limit masks provided.
            if is_upper is None and is_lower is None:
                return photometry_loglike_numba_no_limits(
                    data.ravel(),
                    model.ravel(),
                    var.ravel(),
                    weights.ravel(),
                    include_norm,
                )

            if is_upper is None:
                is_upper = np.zeros_like(data, dtype=bool)

            if is_lower is None:
                is_lower = np.zeros_like(data, dtype=bool)

            if not np.any(is_upper | is_lower):
                return photometry_loglike_numba_no_limits(
                    data.ravel(),
                    model.ravel(),
                    var.ravel(),
                    weights.ravel(),
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
