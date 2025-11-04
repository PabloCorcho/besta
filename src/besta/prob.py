#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 06:18:51 2025

@author: pcorchoc
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List

import numpy as np


# ------------------------------- utilities -------------------------------

def _logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    """
    Stable logsumexp.

    Parameters
    ----------
    a : ndarray
        Input array.
    axis : int or None
        Axis over which to reduce. If None, reduces over all elements.

    Returns
    -------
    out : ndarray
        log(sum(exp(a))) along the given axis.
    """
    m = np.nanmax(a, axis=axis, keepdims=True)
    m[np.isneginf(m)] = 0.0
    s = np.sum(np.exp(a - m), axis=axis, keepdims=True)
    out = np.log(s) + m
    if axis is None:
        return out.reshape(())
    return np.squeeze(out, axis=axis)


def _normal_logpdf(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """
    Univariate normal log pdf per element.

    Parameters
    ----------
    x, mu, sigma : ndarray
        Broadcastable arrays. sigma must be positive.

    Returns
    -------
    logp : ndarray
        Log density values.
    """
    var = sigma * sigma
    return -0.5 * (np.log(2.0 * np.pi * var) + (x - mu) ** 2 / var)


def _std_norm_cdf(x: np.ndarray) -> np.ndarray:
    """
    Standard normal CDF using erf.

    Parameters
    ----------
    x : ndarray

    Returns
    -------
    cdf : ndarray
    """
    from math import sqrt
    from scipy.special import erf  # if you prefer not to depend on SciPy, replace with a rational approx
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# ================================ Priors ================================

class Prior(ABC):
    """
    Abstract prior interface over model targets.

    A prior returns log p(theta) for each model row. It may depend on
    specific target columns (e.g., redshift) and optionally on other
    model metadata.

    Methods
    -------
    log_prob_for_models(targets)
        Return log prior probability per model row.
    """

    @abstractmethod
    def log_prob_for_models(self, targets: np.ndarray) -> np.ndarray:
        """
        Evaluate log prior for each model row.

        Parameters
        ----------
        targets : ndarray, shape (N, Q)
            Model targets array from ModelGrid.

        Returns
        -------
        logp : ndarray, shape (N,)
            Log prior probability per model.
        """
        raise NotImplementedError


@dataclass
class FlatPrior(Prior):
    """
    Flat prior over all models (constant log probability).
    """

    def log_prob_for_models(self, targets: np.ndarray) -> np.ndarray:
        return np.zeros(targets.shape[0], dtype=float)


@dataclass
class EmpiricalHistogramPrior1D(Prior):
    """
    Empirical 1-D histogram prior over a single target column.

    This prior estimates p(t) from the model grid itself and assigns the
    same log prior to all models that fall in the same bin.

    Parameters
    ----------
    target_col : int
        Index of the target column to build the prior on (e.g., redshift).
    edges : ndarray, shape (K+1,)
        Histogram bin edges. Must cover the support of the target.
    density_floor : float, optional
        Minimum probability mass per bin to avoid -inf (default 1e-12).
    """

    target_col: int
    edges: np.ndarray
    density_floor: float = 1e-12

    def fit_from_targets(self, targets: np.ndarray, weights: Optional[np.ndarray] = None) -> "EmpiricalHistogramPrior1D":
        """
        Fit histogram from targets.

        Parameters
        ----------
        targets : ndarray, shape (N, Q)
        weights : ndarray, shape (N,), optional

        Returns
        -------
        self : EmpiricalHistogramPrior1D
        """
        t = targets[:, self.target_col]
        hist, _ = np.histogram(t, bins=self.edges, weights=weights, density=False)
        mass = hist.astype(float)
        mass = mass / np.sum(mass) if np.sum(mass) > 0 else np.full_like(mass, 1.0 / mass.size)
        mass = np.clip(mass, self.density_floor, None)
        self._logp_per_bin = np.log(mass)
        return self

    def log_prob_for_models(self, targets: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_logp_per_bin"):
            raise RuntimeError("Prior not fitted. Call fit_from_targets first.")
        t = targets[:, self.target_col]
        j = np.digitize(t, self.edges) - 1
        j = np.clip(j, 0, self._logp_per_bin.size - 1)
        return self._logp_per_bin[j]


class ObservableDependentPrior(Prior):
    """TODO"""
    def fit_from_grid(self):
        raise NotImplementedError()

@dataclass
class MagDependentRedshiftPrior(ObservableDependentPrior):
    """
    Magnitude-dependent redshift prior p(z | m) from a 2-D histogram.

    Parameters
    ----------
    z_col : int
        Index of redshift in targets.
    mag_observable_index : int
        Index of magnitude in observables (e.g., VIS magnitude column).
    z_edges : ndarray
        Bin edges in redshift.
    m_edges : ndarray
        Bin edges in magnitude.
    density_floor : float, optional
        Minimum conditional probability per (z bin) to avoid zeros.

    Notes
    -----
    After calling fit_from_grid, the conditional log prior is defined by
    log p(z_k | m_bin) for each magnitude bin. For a given model row, its
    magnitude chooses a column in the 2-D histogram and the redshift of
    that model chooses the row.
    """

    z_col: int
    mag_observable_index: int
    z_edges: np.ndarray
    m_edges: np.ndarray
    density_floor: float = 1e-12

    def fit_from_grid(self, observables: np.ndarray, targets: np.ndarray, weights: Optional[np.ndarray] = None) -> "MagDependentRedshiftPrior":
        """
        Fit conditional histogram from the model grid.

        Parameters
        ----------
        observables : ndarray, shape (N, P)
        targets : ndarray, shape (N, Q)
        weights : ndarray, shape (N,), optional

        Returns
        -------
        self : MagDependentRedshiftPrior
        """
        z = targets[:, self.z_col]
        m = observables[:, self.mag_observable_index]
        H, z_edges, m_edges = np.histogram2d(z, m, bins=[self.z_edges, self.m_edges], weights=weights)
        # normalise each magnitude column to sum 1 over z
        colsum = H.sum(axis=0, keepdims=True)
        colsum[colsum == 0] = 1.0
        P = H / colsum
        P = np.clip(P, self.density_floor, None)
        self._logP_z_given_m = np.log(P)  # shape (Kz, Km)
        return self

    def log_prob_for_models(self, targets: np.ndarray, observables: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Evaluate log p(z | m) per model row.

        Parameters
        ----------
        targets : ndarray, shape (N, Q)
        observables : ndarray, shape (N, P), optional
            Required for the magnitude column. If None, raises.

        Returns
        -------
        logp : ndarray, shape (N,)
        """
        if not hasattr(self, "_logP_z_given_m"):
            raise RuntimeError("Prior not fitted. Call fit_from_grid first.")
        if observables is None:
            raise ValueError("observables must be provided to evaluate p(z|m)")
        z = targets[:, self.z_col]
        m = observables[:, self.mag_observable_index]
        iz = np.clip(np.digitize(z, self.z_edges) - 1, 0, self._logP_z_given_m.shape[0] - 1)
        im = np.clip(np.digitize(m, self.m_edges) - 1, 0, self._logP_z_given_m.shape[1] - 1)
        return self._logP_z_given_m[iz, im]


class HierarchicalPrior(Prior):
    def __init__(self, hyperparams: dict):
        self.hyperparams = hyperparams

    def update_hyperparams(self, new_values: dict) -> None:
        self.hyperparams.update(new_values)

    @abstractmethod
    def log_prob_for_models(self, targets: np.ndarray, **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def fit_from_data(self, targets: np.ndarray, observables: np.ndarray, weights: np.ndarray | None = None) -> None:
        pass

# class MainSequencePrior(HierarchicalPrior):

# ============================== Likelihoods ==============================

class Likelihood(ABC):
    """
    Abstract likelihood interface p(x | model).

    Methods
    -------
    log_likelihood(x_native, sigma_native, X_models)
        Return log likelihood per candidate model row.
    """

    @abstractmethod
    def log_likelihood(self,
                       x_native: np.ndarray,
                       sigma_native: np.ndarray,
                       X_models: np.ndarray) -> np.ndarray:
        """
        Evaluate log likelihood for each model.

        Parameters
        ----------
        x_native : ndarray, shape (P,)
            Query observables.
        sigma_native : ndarray, shape (P,)
            Measurement uncertainties for the query.
        X_models : ndarray, shape (Nc, P)
            Candidate model observables.

        Returns
        -------
        logL : ndarray, shape (Nc,)
        """
        raise NotImplementedError


@dataclass
class GaussianProductLikelihood(Likelihood):
    """
    Independent per-dimension Gaussian product likelihood.

    The likelihood is proportional to the product over j of
    N(x_j | X_ij, h_j^2), where h_j is derived from sigma_native with
    an optional floor.

    Parameters
    ----------
    bandwidth_floor : float, optional
        Minimum bandwidth per dimension in native units. Default 0.0.
    scale : float, optional
        Multiplicative scale applied to sigma_native. Default 1.0.
    """

    bandwidth_floor: float = 0.0
    scale: float = 1.0

    def log_likelihood(self,
                       x_native: np.ndarray,
                       sigma_native: np.ndarray,
                       X_models: np.ndarray) -> np.ndarray:
        h = np.maximum(self.scale * sigma_native, self.bandwidth_floor)
        # broadcast to (Nc, P)
        diff = (X_models - x_native[None, :])
        var = h[None, :] ** 2
        # sum of 1-D logpdfs
        logL = -0.5 * (np.sum(np.log(2.0 * np.pi * var), axis=1) + np.sum(diff ** 2 / var, axis=1))
        return logL


@dataclass
class CensoredSizeLikelihood(Likelihood):
    """
    Photometry-only Gaussian product with a left-censored size factor.

    This is useful when the apparent size is below a reliability floor
    (e.g., PSF or measurement threshold). The photometric part is a
    Gaussian product over selected photometry indices. The size part
    adds a log CDF factor log Phi((s_min - s_model) / h_s), where s is
    log10(Re) and h_s is derived from sigma_native[size_index].

    Parameters
    ----------
    phot_indices : Sequence[int]
        Indices of observable columns to include in the Gaussian product
        (typically colours and anchor magnitude).
    size_index : int
        Index of the size observable column (e.g., log10(Re)).
    s_min : float
        Left-censoring threshold in the same units as the size observable.
    bandwidth_floor : float, optional
        Minimum bandwidth per dimension in native units. Default 0.0.
    scale : float, optional
        Multiplicative scale applied to sigma_native. Default 1.0.
    """

    phot_indices: Sequence[int]
    size_index: int
    s_min: float
    bandwidth_floor: float = 0.0
    scale: float = 1.0

    def log_likelihood(self,
                       x_native: np.ndarray,
                       sigma_native: np.ndarray,
                       X_models: np.ndarray) -> np.ndarray:
        # Photometry part
        phot_idx = np.asarray(self.phot_indices, dtype=int)
        x_ph = x_native[phot_idx]
        sig_ph = np.maximum(self.scale * sigma_native[phot_idx], self.bandwidth_floor)
        Xm_ph = X_models[:, phot_idx]
        diff = (Xm_ph - x_ph[None, :])
        var = sig_ph[None, :] ** 2
        logL_ph = -0.5 * (np.sum(np.log(2.0 * np.pi * var), axis=1) + np.sum(diff ** 2 / var, axis=1))

        # Censored size factor: log Phi((s_min - s_model)/h_s)
        h_s = max(self.scale * sigma_native[self.size_index], self.bandwidth_floor)
        s_model = X_models[:, self.size_index]
        z = (self.s_min - s_model) / h_s
        # avoid log(0)
        cdf = np.clip(_std_norm_cdf(z), 1e-300, 1.0)
        logL_sz = np.log(cdf)

        return logL_ph + logL_sz


@dataclass
class CompositeLikelihood(Likelihood):
    """
    Sum of multiple likelihood terms (log-likelihoods add).

    Parameters
    ----------
    terms : list of Likelihood
        Components to add together.
    """

    terms: List[Likelihood]

    def log_likelihood(self,
                       x_native: np.ndarray,
                       sigma_native: np.ndarray,
                       X_models: np.ndarray) -> np.ndarray:
        total = None
        for lk in self.terms:
            lp = lk.log_likelihood(x_native, sigma_native, X_models)
            total = lp if total is None else (total + lp)
        return total if total is not None else np.zeros(X_models.shape[0], dtype=float)


# =========================== posterior helper ===========================

def posterior_over_models(x_native: np.ndarray,
                          sigma_native: np.ndarray,
                          X_models: np.ndarray,
                          targets_models: np.ndarray,
                          likelihood: Likelihood,
                          prior: Prior,
                          model_weights: Optional[np.ndarray] = None,
                          prior_needs_observables: bool = False,
                          observables_models: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convenience function: compute normalised posterior weights over models.

    Parameters
    ----------
    x_native : ndarray, shape (P,)
        Query observables.
    sigma_native : ndarray, shape (P,)
        Measurement uncertainties for the query.
    X_models : ndarray, shape (Nc, P)
        Candidate model observables.
    targets_models : ndarray, shape (Nc, Q)
        Candidate model targets.
    likelihood : Likelihood
        Likelihood instance to evaluate log p(x | model).
    prior : Prior
        Prior instance to evaluate log p(model).
    model_weights : ndarray, shape (Nc,), optional
        Optional sampling weights for models; multiplies the posterior.
    prior_needs_observables : bool, optional
        If True, the prior expects observables to evaluate (for p(z|m)).
    observables_models : ndarray, shape (Nc, P), optional
        Candidate observables to pass to the prior if needed.

    Returns
    -------
    w : ndarray, shape (Nc,)
        Normalised posterior weights over candidate models.
    """
    logL = likelihood.log_likelihood(x_native, sigma_native, X_models)

    if prior_needs_observables:
        logP = prior.log_prob_for_models(targets_models, observables=observables_models if observables_models is not None else X_models)  # type: ignore
    else:
        logP = prior.log_prob_for_models(targets_models)

    logw = logL + logP
    if model_weights is not None:
        # multiply in linear space -> add in log space
        with np.errstate(divide="ignore"):
            logw = logw + np.log(np.clip(model_weights, 1e-300, np.inf))

    # normalise safely
    logZ = _logsumexp(logw)
    w = np.exp(logw - logZ)
    return w
