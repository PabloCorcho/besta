"""Probability, likelihood, and prior models for grid-based inference."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List
import warnings
from numba import njit, prange

import numpy as np
from besta.logging import get_logger

logger = get_logger(__name__)

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
    from scipy.special import (
        erf,
    )  # if you prefer not to depend on SciPy, replace with a rational approx

    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


# ================================ Priors ================================


class Prior(ABC):
    """
    Prior base class.

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
class GaussianPrior1D(Prior):
    """
    1-D Gaussian prior over a single target column.

    Parameters
    ----------
    target_col : int
        Index of the target column to build the prior on (e.g., redshift).
    mu : float
        Mean of the Gaussian prior.
    sigma : float
        Standard deviation of the Gaussian prior.
    """

    target_col: int
    mu: float
    sigma: float

    def log_prob_for_models(self, targets: np.ndarray) -> np.ndarray:
        t = targets[:, self.target_col]
        return _normal_logpdf(t, self.mu, self.sigma)


@dataclass
class UniformPrior1D(Prior):
    """
    1-D uniform prior over a single target column.

    Parameters
    ----------
    target_col : int
        Index of the target column to build the prior on (e.g., redshift).
    low : float
        Lower bound of the uniform prior.
    high : float
        Upper bound of the uniform prior.
    """

    target_col: int
    low: float
    high: float

    def log_prob_for_models(self, targets: np.ndarray) -> np.ndarray:
        t = targets[:, self.target_col]
        logp = np.where(
            (t >= self.low) & (t <= self.high), -np.log(self.high - self.low), -np.inf
        )
        return logp


@dataclass
class DeltaPrior1D(Prior):
    """
    1-D delta-function prior over a single target column.

    Parameters
    ----------
    target_col : int
        Index of the target column to build the prior on (e.g., redshift).
    value : float
        Location of the delta prior.
    tolerance : float, optional
        Width around `value` to assign finite log-probability (default 1e-6).
    """

    target_col: int
    value: float
    tolerance: float = 1e-6

    def log_prob_for_models(self, targets: np.ndarray) -> np.ndarray:
        t = targets[:, self.target_col]
        logp = np.where(
            np.abs(t - self.value) <= self.tolerance,
            -np.log(2.0 * self.tolerance),
            -np.inf,
        )
        return logp


@dataclass
class ExponentialPrior1D(Prior):
    """
    1-D exponential prior over a single target column.

    Parameters
    ----------
    target_col : int
        Index of the target column to build the prior on (e.g., redshift).
    scale : float
        Scale parameter (mean) of the exponential prior.
    """

    target_col: int
    scale: float

    def log_prob_for_models(self, targets: np.ndarray) -> np.ndarray:
        t = targets[:, self.target_col]
        logp = np.where(t >= 0, -t / self.scale - np.log(self.scale), -np.inf)
        return logp


@dataclass
class ExponentialTruncatedPrior1D(Prior):
    """
    1-D truncated exponential prior over a single target column.

    Parameters
    ----------
    target_col : int
        Index of the target column to build the prior on (e.g., redshift).
    scale : float
        Scale parameter (mean) of the exponential prior.
    t_max : float
        Maximum value of the target (support upper bound).
    """

    target_col: int
    scale: float
    t_max: float

    def log_prob_for_models(self, targets: np.ndarray) -> np.ndarray:
        t = targets[:, self.target_col]
        norm = 1.0 - np.exp(-self.t_max / self.scale)
        logp = np.where(
            (t >= 0) & (t <= self.t_max),
            -t / self.scale - np.log(self.scale * norm),
            -np.inf,
        )
        return logp


@dataclass
class PowerLawPrior1D(Prior):
    """
    1-D power-law prior over a single target column.

    Parameters
    ----------
    target_col : int
        Index of the target column to build the prior on (e.g., redshift).
    alpha : float
        Power-law index (p(t) ~ t^alpha).
    t_min : float
        Minimum value of the target (support lower bound).
    t_max : float
        Maximum value of the target (support upper bound).
    """

    target_col: int
    alpha: float
    t_min: float
    t_max: float

    def log_prob_for_models(self, targets: np.ndarray) -> np.ndarray:
        t = targets[:, self.target_col]
        norm = (self.t_max ** (self.alpha + 1) - self.t_min ** (self.alpha + 1)) / (
            self.alpha + 1
        )
        logp = np.where(
            (t >= self.t_min) & (t <= self.t_max),
            self.alpha * np.log(t) - np.log(norm),
            -np.inf,
        )
        return logp


@dataclass
class EmpiricalHistogramPrior1D(Prior):
    """
    Empirical 1-D histogram prior over a single target column.

    This prior estimates p(t) from the model grid itself and assigns the
    same log prior to all models that fall in the same bin.

    Parameters
    ----------
    target_col : int
        Index of the target column to build the prior.
    edges : ndarray, shape (K+1,)
        Histogram bin edges. Must cover the support of the target.
    density_floor : float, optional
        Minimum probability mass per bin to avoid -inf (default 1e-12).
    """

    target_col: int
    edges: np.ndarray
    density_floor: float = 1e-12

    def fit_from_targets(
        self, targets: np.ndarray, weights: Optional[np.ndarray] = None
    ) -> "EmpiricalHistogramPrior1D":
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

        logger.debug("Fitting EmpiricalHistogramPrior 1D")
        # Select the target column
        t = targets[:, self.target_col]
        # Compute the histogram
        hist, _ = np.histogram(t, bins=self.edges, weights=weights, density=False)
        # Normalise histogram
        norm = np.sum(mass)
        if norm > 0:
            mass /= norm
        else:
            mass = np.full_like(mass, 1.0 / mass.size)
        mass = np.clip(mass, self.density_floor, None)
        self._logp_per_bin = np.log(mass)
        return self

    def log_prob_for_models(self, targets: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_logp_per_bin"):
            raise RuntimeError("Prior not fitted. Call fit_from_targets first.")
        t = targets[:, self.target_col]
        # Bin targets using the pre-defined bins
        j = np.digitize(t, self.edges) - 1
        j = np.clip(j, 0, self._logp_per_bin.size - 1)
        return self._logp_per_bin[j]


@dataclass
class EmpiricalFlatteningPriorND(Prior):
    """
    Empirical flattening prior over arbitrary target columns.

    This prior uses the model grid itself to estimate the (possibly
    non-uniform) distribution of a set of parameters and builds a prior
    that counteracts those inhomogeneities.

    Two modes are provided:

    - 'factorised': build 1-D histograms for each column separately and
      form a product prior over dimensions. This approximately flattens
      the marginal distributions of those parameters.

    - 'joint': build a joint N-D histogram over all selected columns and
      assign prior mass proportional to 1 / N_k for each occupied
      N-D bin. This flattens the distribution over the joint grid of
      bins, but is more memory hungry and can be sparse.

    Parameters
    ----------
    target_cols : sequence of int
        Indices of the target columns to build the prior on.
    edges_list : sequence of ndarray
        List of bin edges for each column. Must have the same length as
        `target_cols`. Each element is an array of shape (K_d + 1,)
        defining the bin edges along dimension d.
    mode : {'factorised', 'joint'}, optional
        Flattening strategy (default 'factorised').
    density_floor : float, optional
        Minimum probability mass per bin (or joint cell) to avoid
        -inf log-probabilities (default 1e-12). Only relevant for
        'joint' mode.
    count_floor : float, optional
        Minimum effective count per bin to avoid infinite weights
        (default 1e-3). Used to clip empty or nearly empty bins.

    Notes
    -----
    The prior is defined *over models*, not over a continuous parameter
    space. It is intended to compensate for non-uniform sampling of the
    grid, so that the effective prior over the chosen parameters is
    closer to flat.
    """

    target_cols: Sequence[int]
    edges_list: Sequence[np.ndarray]
    mode: str = "factorised"
    density_floor: float = 1e-12
    count_floor: float = 1e-3

    def __post_init__(self):
        if len(self.target_cols) != len(self.edges_list):
            raise ValueError("target_cols and edges_list must have the same length.")
        if self.mode not in ("factorised", "joint"):
            raise ValueError("mode must be 'factorised' or 'joint'.")

    def fit_from_targets(
        self,
        targets: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> "EmpiricalFlatteningPriorND":
        """
        Fit flattening prior from the model grid targets.

        Parameters
        ----------
        targets : ndarray, shape (N, Q)
            Model grid targets.
        weights : ndarray, shape (N,), optional
            Optional model weights (e.g. importance weights) used to
            define the effective occupancy per bin.

        Returns
        -------
        self : EmpiricalFlatteningPriorND
        """
        logger.debug("Fitting EmpiricalHistogramPriorND")
        t = targets[:, self.target_cols]  # (N, D)
        # Grid dimensions
        D = t.shape[1]

        if weights is not None and weights.shape[0] != t.shape[0]:
            raise ValueError("weights must have shape (N,) if provided.")

        if self.mode == "factorised":
            logger.debug("Using 'factorised' mode (per-dim prior)")
            # One histogram per dimension, store log inverse-mass per bin
            log_inv_mass_list = []
            for d in range(D):
                edges = self.edges_list[d]
                # get all parameter values
                td = t[:, d]
                # compute histogram
                counts, _ = np.histogram(td, bins=edges, weights=weights, density=False)
                total = np.sum(counts)
                if total <= 0:
                    raise RuntimeError(
                        f"No models in any user-provided bin for dimension {d}; cannot fit prior."
                    )
                # Clip prior to prevent zero division
                counts = np.clip(counts, self.count_floor, None)

                # Define per-bin prior mass proportional to 1 / counts
                inv_counts = 1.0 / counts
                inv_counts /= np.sum(inv_counts)
                log_inv_mass_list.append(np.log(inv_counts))

            self._log_inv_mass_list = log_inv_mass_list

        else:  # mode == 'joint'
            logger.debug("Using 'joint' mode (multi-dim prior)")
            # Build joint N-D histogram
            bin_indices = []
            bin_sizes = []
            for d in range(D):
                edges = self.edges_list[d]
                td = t[:, d]
                j = np.digitize(td, edges) - 1
                # Clip into valid range
                j = np.clip(j, 0, edges.size - 2)
                bin_indices.append(j)
                bin_sizes.append(edges.size - 1)

            bin_indices = np.stack(bin_indices, axis=0)

            # Flatten to 1-D indices for bincount
            linear_indices = np.ravel_multi_index(
                bin_indices, dims=tuple(bin_sizes)
            )

            counts_flat = np.bincount(
                linear_indices,
                weights=weights,
                minlength=int(np.prod(bin_sizes)),
            ).astype(float)

            total = np.sum(counts_flat)
            if total <= 0:
                raise RuntimeError("No models in any joint bin; cannot fit prior.")

            counts_flat = np.clip(counts_flat, self.count_floor, None)
            self.counts_flat = counts_flat
            # Per-cell prior mass proportional to 1 / counts
            inv_counts_flat = 1.0 / counts_flat
            inv_counts_flat /= np.sum(inv_counts_flat)

            inv_counts_flat = np.clip(inv_counts_flat, self.density_floor, None)
            self._log_mass_per_cell = np.log(inv_counts_flat)
            self._bin_sizes = tuple(bin_sizes)

        return self

    def log_prob_for_models(self, targets: np.ndarray) -> np.ndarray:
        """
        Compute log prior for each model in the given targets array.

        Parameters
        ----------
        targets : ndarray, shape (N, Q)
            Model grid targets.

        Returns
        -------
        logp : ndarray, shape (N,)
            Log prior for each target row.
        """

        t = targets[:, self.target_cols]  # (N, D)
        N, D = t.shape
        logp = np.zeros(N, dtype=float)

        if self.mode == "factorised":
            if not hasattr(self, "_log_inv_mass_list"):
                raise RuntimeError("Prior not fitted. Call fit_from_targets first.")

            for d in range(D):
                edges = self.edges_list[d]
                td = t[:, d]
                j = np.digitize(td, edges) - 1
                j = np.clip(j, 0, edges.size - 2)

                log_inv_mass_d = self._log_inv_mass_list[d]
                logp += log_inv_mass_d[j]

        elif self.mode == "joint":
            if not hasattr(self, "_log_mass_per_cell"):
                raise RuntimeError("Prior not fitted. Call fit_from_targets first.")

            bin_indices = []
            for d in range(D):
                edges = self.edges_list[d]
                td = t[:, d]
                j = np.digitize(td, edges) - 1
                # Clip into valid range
                j = np.clip(j, 0, edges.size - 2)
                bin_indices.append(j)

            linear_indices = np.ravel_multi_index(bin_indices, dims=self._bin_sizes)
            logp += self._log_mass_per_cell[linear_indices]

        return logp


class ObservableDependentPrior(Prior):
    """Base class for priors that depend on observables."""

    def fit_from_grid(self):
        raise NotImplementedError()


@dataclass
class MagDependentRedshiftPrior(ObservableDependentPrior):
    """
    Magnitude-dependent redshift prior, i.e. likelihood
    of an object with a magnitude ``m`` being detected at
    redshift ``z``.

    Parameters
    ----------
    z_col : int
        Index of redshift in targets.
    mag_observable_index : int
        Index of magnitude in observables.
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
    # TODO: allow for optional user-provided prior

    def fit_from_grid(
        self,
        observables: np.ndarray,
        targets: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> "MagDependentRedshiftPrior":
        """
        Fit conditional histogram from input dataset.

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
        H, z_edges, m_edges = np.histogram2d(
            z, m, bins=[self.z_edges, self.m_edges], weights=weights
        )
        # compute the conditional distribution
        colsum = H.sum(axis=0, keepdims=True)
        colsum[colsum == 0] = 1.0
        p_z_given_m = H / colsum
        p_z_given_m = np.clip(p_z_given_m, self.density_floor, None)
        self._logP_z_given_m = np.log(p_z_given_m)
        return self

    def log_prob_for_models(
        self, targets: np.ndarray, observables: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Evaluate :math:`log p(z | m)` per model row.

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
        # Interpolate input magnitudes
        iz = np.clip(
            np.digitize(z, self.z_edges) - 1, 0, self._logP_z_given_m.shape[0] - 1
        )
        im = np.clip(
            np.digitize(m, self.m_edges) - 1, 0, self._logP_z_given_m.shape[1] - 1
        )
        return self._logP_z_given_m[iz, im]


@dataclass
class CompositePrior(Prior):
    """
    Composite prior combining multiple target-only Priors.

    The total log prior is defined as a weighted sum of component
    log priors:

        :math:`\log p_{total}(model) = sum_i w_i * \log p_i(model)`

    where each p_i is a Prior that does not depend on observables.

    Parameters
    ----------
    priors : sequence of Prior
        List or tuple of prior instances to combine. Each must implement
        `log_prob_for_models(targets)` and must NOT require observables.
    weights : sequence of float, optional
        Per-component multiplicative weights for the log-probabilities.
        If None, all weights are taken as 1.0. A zero weight effectively
        disables a component. Negative weights are allowed but should be
        used with care, as they correspond to dividing by p_i(model) in
        linear space.
    """

    priors: Sequence[Prior]
    weights: Optional[Sequence[float]] = None

    def __post_init__(self):
        self.priors = list(self.priors)
        logger.debug(f"Setting up CompositePrior with {len(self.priors)} priors")
        if not self.priors:
            raise ValueError("CompositePrior requires at least one component prior.")

        for i, pr in enumerate(self.priors):
            if isinstance(pr, ObservableDependentPrior):
                raise TypeError(
                    f"CompositePrior component {i} is observable-dependent "
                    "(ObservableDependentPrior); use ObservableCompositePrior instead."
                )

        if self.weights is not None:
            logger.debug("Using user-provided relative prior weights")
            if len(self.weights) != len(self.priors):
                raise ValueError(
                    "weights must have the same length as priors "
                    f"({len(self.weights)} != {len(self.priors)})"
                )
            self._weights = np.asarray(self.weights, dtype=float)
        else:
            self._weights = None

    def log_prob_for_models(self, targets: np.ndarray) -> np.ndarray:
        """
        Evaluate the composite log prior for each model row.

        Parameters
        ----------
        targets : ndarray, shape (N, Q)
            Model targets array.

        Returns
        -------
        logp : ndarray, shape (N,)
            Composite log prior per model.
        """
        N = targets.shape[0]
        logp_total: Optional[np.ndarray] = None

        for i, pr in enumerate(self.priors):
            w_i = 1.0 if self._weights is None else float(self._weights[i])
            if w_i == 0.0:
                continue
            lp_i = pr.log_prob_for_models(targets)
            if lp_i.shape != (N,):
                raise ValueError(
                    f"Component prior {i} returned shape {lp_i.shape}, expected ({N},)"
                )

            if w_i != 1.0:
                lp_i = w_i * lp_i

            logp_total = lp_i if logp_total is None else (logp_total + lp_i)

        if logp_total is None:
            warnings.warn(
                "All weights were zero in CompositePrior; returning flat prior."
            )
            logger.warning(
                "All weights were zero in CompositePrior; returning flat prior."
            )
            return np.zeros(N, dtype=float)
        return logp_total


@dataclass
class ObservableCompositePrior(ObservableDependentPrior):
    """
    Same as :class:`CompositePrior`, but including :class:`ObservableDependentPrior`.

    The total log prior is defined as a weighted sum of component
    log priors:

        :math:`\log p_{total}(model) = sum_i w_i * \log p_i(model)`

    Components can be:
      - Plain Prior (target-only), evaluated as
            p_i.log_prob_for_models(targets)
      - ObservableDependentPrior, evaluated as
            p_i.log_prob_for_models(targets, observables=...)

    Parameters
    ----------
    priors : sequence of Prior
        List or tuple of prior instances to combine. Each must implement
        `log_prob_for_models(...)` following the Prior or
        ObservableDependentPrior API.
    weights : sequence of float, optional
        Per-component multiplicative weights for the log-probabilities.
        If None, all weights are taken as 1.0. A zero weight effectively
        disables a component. Negative weights are allowed but should be
        used with care, as they correspond to dividing by p_i(model) in
        linear space.

    Notes
    -----
    - Because this class inherits from ObservableDependentPrior, external
      code (e.g. GridFitter / posterior_over_models) should treat it as
      requiring observables and pass them when available.
    - If the composite contains any ObservableDependentPrior and
      `observables` is None, a ValueError is raised.
    """

    priors: Sequence[Prior]
    weights: Optional[Sequence[float]] = None

    def __post_init__(self):
        self.priors = list(self.priors)
        logger.debug(f"Setting up CompositePrior with {len(self.priors)} priors")

        if not self.priors:
            raise ValueError(
                "ObservableCompositePrior requires at least one component prior."
            )

        if self.weights is not None:
            logger.debug("Using user-provided relative prior weights")
            if len(self.weights) != len(self.priors):
                raise ValueError(
                    "weights must have the same length as priors "
                    f"({len(self.weights)} != {len(self.priors)})"
                )
            self._weights = np.asarray(self.weights, dtype=float)
        else:
            self._weights = None

    def log_prob_for_models(
        self,
        targets: np.ndarray,
        observables: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Evaluate the composite log prior for each model row.

        Parameters
        ----------
        targets : ndarray, shape (N, Q)
            Model targets array.
        observables : ndarray, shape (N, P), optional
            Model observables. Required if any component is an
            ObservableDependentPrior. If such a prior is present and
            observables is None, a ValueError is raised.

        Returns
        -------
        logp : ndarray, shape (N,)
            Composite log prior per model.
        """
        N = targets.shape[0]
        logp_total: Optional[np.ndarray] = None
        need_obs = any(isinstance(pr, ObservableDependentPrior) for pr in self.priors)

        if need_obs and observables is None:
            raise ValueError(
                "ObservableCompositePrior includes observable-dependent components "
                "but 'observables' was not provided."
            )

        for i, pr in enumerate(self.priors):
            w_i = 1.0 if self._weights is None else float(self._weights[i])
            if w_i == 0.0:
                continue  # skip null-weight component

            # Evaluate component prior
            if isinstance(pr, ObservableDependentPrior):
                lp_i = pr.log_prob_for_models(targets, observables=observables)
            else:
                lp_i = pr.log_prob_for_models(targets)

            if lp_i.shape != (N,):
                raise ValueError(
                    f"Component prior {i} returned shape {lp_i.shape}, expected ({N},)"
                )

            if w_i != 1.0:
                lp_i = w_i * lp_i

            logp_total = lp_i if logp_total is None else (logp_total + lp_i)

        if logp_total is None:
            # All weights were zero; fall back to flat
            return np.zeros(N, dtype=float)

        return logp_total


# ============================== Likelihoods ==============================


class Likelihood(ABC):
    """
    Base likelihood class representing :math:`p(x | model)`.

    Methods
    -------
    log_likelihood(x_native, sigma_native, X_models)
        Return log likelihood per candidate model row.
    """

    @abstractmethod
    def log_likelihood(
        self, x_native: np.ndarray, sigma_native: np.ndarray, X_models: np.ndarray
    ) -> np.ndarray:
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

    Parameters
    ----------
    bandwidth_floor : float, optional
        Minimum bandwidth per dimension in native units. Default 0.0.
    scale : float, optional
        Multiplicative scale applied to sigma_native. Default 1.0.
    """

    bandwidth_floor: float = 0.0
    scale: float = 1.0

    def log_likelihood(
        self, x_native: np.ndarray, sigma_native: np.ndarray, X_models: np.ndarray
    ) -> np.ndarray:
        # truncate prior
        h = np.maximum(self.scale * sigma_native, self.bandwidth_floor)
        diff = X_models - x_native[None, :]
        var = h[None, :] ** 2
        # sum of 1-D logpdfs
        logL = -0.5 * (
            np.sum(np.log(2.0 * np.pi * var), axis=1) + np.sum(diff**2 / var, axis=1)
        )
        return logL


@dataclass
class SplitGaussianProductLikelihood(Likelihood):
    """Independent per-dimension split Gaussian likelihood

    The likelihood along each dimension is given by

    .. math:

        \mathcal{L} = \mathcal{N}(x | \mu, \sigma_L),\, if x\leq\mu\\
        \mathcal{L} = \mathcal{N}(x | \mu, \sigma_R),\, if x>\mu

    and the total likelihood is the product along all dimensions.
    """
    bandwith_floor: float = 0.0
    scale_left: float = 1.0
    scale_right: float = 1.0

    def log_likelihood(
        self, x_native: np.ndarray, sigma_native: np.ndarray, X_models: np.ndarray
    ) -> np.ndarray:
        # truncate prior
        raise NotImplementedError("Class not implemented")


@dataclass
class CompositeLikelihood(Likelihood):
    """
    Sum of multiple likelihood terms.

    Parameters
    ----------
    terms : list of Likelihood
        Components to add together.
    """

    terms: List[Likelihood]

    def log_likelihood(
        self, x_native: np.ndarray, sigma_native: np.ndarray, X_models: np.ndarray
    ) -> np.ndarray:
        total = None
        for lk in self.terms:
            lp = lk.log_likelihood(x_native, sigma_native, X_models)
            total = lp if total is None else (total + lp)
        return total if total is not None else np.zeros(X_models.shape[0], dtype=float)


# =========================== posterior helper ===========================


def posterior_over_models(
    x_native: np.ndarray,
    sigma_native: np.ndarray,
    X_models: np.ndarray,
    targets_models: np.ndarray,
    likelihood: Likelihood,
    prior: Prior,
    model_weights: Optional[np.ndarray] = None,
    prior_needs_observables: bool = False,
    observables_models: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute normalised posterior weights over models.

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
        logP = prior.log_prob_for_models(
            targets_models,
            observables=observables_models
            if observables_models is not None
            else X_models,
        )
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


# Numba-dedicated likelihood for increased performance


@njit(parallel=True, fastmath=True, cache=True)
def _quadform_diag_parallel(X, x, h):
    N, P = X.shape
    out = np.empty(N, dtype=np.float64)
    invh = 1.0 / h
    for i in prange(N):
        s = 0.0
        Xi = X[i]
        for j in range(P):
            d = (Xi[j] - x[j]) * invh[j]
            s += d * d
        out[i] = s
    return out

@njit(parallel=True, fastmath=True, cache=True)
def _loglike_gaussprod_diag(X, x, h):
    q = _quadform_diag_parallel(X, x, h)
    return -0.5 * q


class NumbaGaussianProductLikelihood(GaussianProductLikelihood):
    """
    Diagonal Gaussian product likelihood accelerated with Numba.

    log L_i = -0.5 * sum_j ((X_ij - x_j)/h_j)^2
    """

    def __init__(
        self,
        bandwidth_floor: float = 0.0,
        scale: float = 1.0,
        prefer_batch: bool = False,
    ):
        super().__init__(bandwidth_floor=bandwidth_floor, scale=scale)
        self.prefer_batch = prefer_batch

    def log_likelihood(self, x_native, sigma_native, X_models):
        # C-contiguous float64 for best performance (https://stackoverflow.com/questions/67784563/how-to-make-two-arrays-contiguous-so-that-numba-can-speed-up-np-dot)
        x = np.ascontiguousarray(x_native, dtype=np.float64)
        X = np.ascontiguousarray(X_models, dtype=np.float64)
        sigma = np.ascontiguousarray(sigma_native, dtype=np.float64)
        # Match parent class: apply scale and bandwidth_floor
        h = np.maximum(self.scale * sigma, self.bandwidth_floor)
        h = np.ascontiguousarray(h, dtype=np.float64)

        return _loglike_gaussprod_diag(X, x, h)
