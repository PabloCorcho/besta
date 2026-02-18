"""
Model grid container and fitting machinery.

This module defines the ModelGrid class, which stores a grid of models with
their parameters and provides methods for fitting and evaluating these models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Mapping, Iterator, Union
from abc import ABC, abstractmethod
from itertools import product

import os
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.table import Table, Column
from astropy.io import fits
import h5py
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

from besta.grid.prob import (
    Prior,
    FlatPrior,
    ObservableDependentPrior,
    Likelihood,
    GaussianProductLikelihood,
    posterior_over_models as posterior_over_models_fn,
)
from besta.postprocess import (
    enclosed_fraction_map,
    pit_from_discrete_posterior,
    pdf_stats,
    photoz_metrics,
    weighted_quantiles,
)

from .transforms import LinearStandardiser

from besta.utils import available_memory_bytes
from besta.logging import get_logger

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

logger = get_logger(__name__)


def _guess_slices(n_objects, n_observables, n_jobs, tasks_per_worker=6):
    # fewer, larger slices when P is large
    base_tasks = n_jobs * tasks_per_worker
    scale = max(1, n_observables // 8)
    T = max(1, base_tasks // scale)
    # turn T into contiguous slices
    q, r = divmod(n_objects, T)
    s = 0
    out = []
    for i in range(T):
        size = q + (1 if i < r else 0)
        out.append((s, s + size))
        s += size
    return out


def _truncate_posterior_mass(
    cand_idx: np.ndarray,
    w: np.ndarray,
    *,
    keep_mass: float,
    min_candidates: int = 0,
    max_candidates: Optional[int] = None,
    keep_ties: bool = False,
    tie_rtol: float = 0.0,
    tie_atol: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Truncate a discrete posterior ``w`` over candidates to the smallest set
    whose cumulative mass is at least ``keep_mass`` (after sorting by weight
    in descending order).

    Parameters
    ----------
    cand_idx : numpy.ndarray
        Integer array of candidate indices corresponding to the weights in
        ``w``. This is typically an index into a larger grid or catalogue.
    w : numpy.ndarray
        Array of (possibly unnormalised) posterior weights for each
        candidate. Values are normalised internally to sum to one before
        truncation.
    keep_mass : float
        Target cumulative posterior mass to retain in the truncated set.
        Must be between 0 and 1. The function selects the smallest number
        of highest-weight candidates whose cumulative mass meets or exceeds
        this value.
    min_candidates : int, optional
        Minimum number of candidates to keep, regardless of the posterior
        mass accumulated. This is applied after sorting by descending weight
        but before enforcing ``keep_mass``.
    max_candidates : int or None, optional
        Optional upper bound on the number of candidates to keep. If not
        ``None``, the truncated set will contain at most this many
        candidates, even if the requested ``keep_mass`` has not yet been
        reached.
    keep_ties : bool, optional
        If ``True``, include all candidates whose weight is numerically tied
        (within ``tie_rtol`` and ``tie_atol``) with the cut-off candidate at
        the truncation boundary, potentially increasing the number of kept
        candidates beyond ``max_candidates``.
    tie_rtol : float, optional
        Relative tolerance used when determining whether two weights are
        considered equal for the purpose of tie handling. Passed to
        ``numpy.isclose`` via the ``rtol`` parameter.
    tie_atol : float, optional
        Absolute tolerance used when determining whether two weights are
        considered equal for the purpose of tie handling. Passed to
        ``numpy.isclose`` via the ``atol`` parameter.

    Returns
    -------
    cand_kept : numpy.ndarray
        Array of indices of the candidates retained after truncation.
    w_kept : numpy.ndarray
        Array of posterior weights for the retained candidates, renormalised
        to sum to one after truncation.
    meta : dict
        Dictionary with bookkeeping information about the truncation, such as
        the number of candidates before and after truncation, the total mass
        kept and discarded, the weight at the truncation cut, and effective
        sample sizes before and after truncation.
    """
    cand_idx = np.asarray(cand_idx, dtype=np.int64)
    w = np.asarray(w, dtype=float)

    n_before = int(w.size)
    if n_before == 0:
        meta = {
            "n_before": 0,
            "n_after": 0,
            "keep_mass": float(keep_mass),
            "mass_kept": 0.0,
            "mass_dropped": 1.0,
            "cut_weight": np.nan,
            "ess_before": 0.0,
            "ess_after": 0.0,
        }
        return cand_idx, w, meta

    # TODO: remove once properly validated that w is always 1
    s = np.sum(w)
    if not np.isfinite(s) or s <= 0:
        w = np.full_like(w, 1.0 / n_before, dtype=float)
        s = 1.0
    else:
        w = w / s

    # Effective sample size
    ess_before = 1.0 / np.sum(np.square(w)) if np.all(np.isfinite(w)) else 0.0

    # TODO: remove float enforcement to speed up
    keep_mass = float(keep_mass)
    keep_mass = max(0.0, min(1.0, keep_mass))

    # If keep_mass==1, we still may apply max_candidates
    # TODO: if keep_mass==1, return all candidates

    # Compute the cumulative weights from high to low
    ord_desc = np.argsort(w)[::-1]
    w_sorted = w[ord_desc]
    cs = np.cumsum(w_sorted)

    if keep_mass <= 0.0:
        k = 0
    else:
        k = int(np.searchsorted(cs, keep_mass, side="left") + 1)

    # Clip limits
    k = max(k, min_candidates)
    if max_candidates is not None:
        k = min(k, max_candidates)
    k = min(k, n_before)

    # Keep ties at boundary if requested (only if k>0 and k<n_before)
    cut_weight = np.nan
    if k > 0:
        cut_weight = w_sorted[k - 1]

    if keep_ties and (k > 0) and (k < n_before):
        # Include all weights equal to boundary within tolerances
        wk = w_sorted[k - 1]
        tol = tie_atol + tie_rtol * abs(wk)
        eq = np.abs(w_sorted - wk) <= tol
        # Extend beyond k-1 only
        if np.any(eq[k:]):
            last = np.max(np.where(eq)[0])
            k = int(last + 1)
            # apply max cap if present
            if max_candidates is not None:
                k = min(k, int(max_candidates))

    # Select top-k subsample of candidates
    keep_local = ord_desc[:k]
    cand_kept = cand_idx[keep_local]
    w_kept_raw = w[keep_local]
    mass_kept = np.sum(w_kept_raw)
    # Re-normalize weights
    if not np.isfinite(mass_kept) or mass_kept <= 0:
        w_kept = np.full_like(w_kept_raw, 1.0 / max(1, w_kept_raw.size), dtype=float)
        mass_kept = 0.0
    else:
        w_kept = w_kept_raw / mass_kept

    ess_after = 1.0 / np.sum(np.square(w_kept)) if w_kept.size > 0 else 0.0

    meta = {
        "n_before": n_before,
        "n_after": w_kept.size,
        "keep_mass": keep_mass,
        "mass_kept": mass_kept,
        "mass_dropped": max(0.0, 1.0 - mass_kept),
        "cut_weight": cut_weight,
        "ess_before": ess_before,
        "ess_after": ess_after,
    }
    return cand_kept, w_kept, meta


# Base model grid class


@dataclass
class ModelGrid:
    """
    Base container for a grid of models.

    This class stores:
      1) observables: array of shape (N, P)
      2) targets:    array of shape (N, Q) with continuous quantities
      3) names and metadata for both
      4) optional per-model weights

    Attributes
    ----------
    observables : ndarray, shape (N, P)
        Observable vectors per model, e.g. colours or fluxes.
    targets : ndarray, shape (N, Q)
        Continuous targets per model, e.g. SFR, logM, metallicity.
    observable_names : list of str
        Names of the P observables in order.
    target_names : list of str
        Names of the Q targets in order.
    weights : ndarray, shape (N,), optional
        Optional sampling weights for models. Defaults to uniform if None.
    meta : dict, optional
        Free-form metadata for provenance, settings, and units.
    check_boundaries : callable, optional
        Function to check if model parameters are within valid boundaries.
        If None, no boundary checks are performed. The function should take a
        model parameter vector as input and return a boolean indicating
        whether the parameters are valid. This is only used during interpolation.
    observable_standardiser : LinearStandardiser
        Standardiser for observable quantities.
    target_standardiser : LinearStandardiser
        Standardiser for target quantities.

    Examples
    --------
    >>> grid = ModelGrid(
    ...     observables=np.array([[1.0, 0.5], [0.8, 0.3], [1.2, 0.7]]),
    ...     targets=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]),
    ...     observable_names=["obs1", "obs2"],
    ...     target_names=["target1", "target2"]
    ... )
    >>> print(grid.n_models)
    3
    >>> print(grid.n_observables)
    2
    """

    observables: np.ndarray
    targets: np.ndarray
    observable_names: List[str]
    target_names: List[str]
    weights: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    check_boundaries: Optional["Callable[[np.ndarray], bool]"] = field(
        default=None, init=True, repr=False
    )
    # Cached stats for standardisation
    observable_standardiser: LinearStandardiser = field(
        default_factory=LinearStandardiser, init=False, repr=False
    )
    target_standardiser: LinearStandardiser = field(
        default_factory=LinearStandardiser, init=False, repr=False
    )

    _kdtree: Optional["cKDTree"] = field(default=None, init=False, repr=False)
    _kdtree_standardized: Optional[bool] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.observables.ndim != 2:
            raise ValueError("observables must be 2D (N, P)")
        if self.targets.ndim != 2:
            raise ValueError("targets must be 2D (N, Q)")
        if self.observables.shape[0] != self.targets.shape[0]:
            raise ValueError("observables and targets must have same N")
        if len(self.observable_names) != self.observables.shape[1]:
            raise ValueError("observable_names length must match P")
        if len(self.target_names) != self.targets.shape[1]:
            raise ValueError("target_names length must match Q")
        if self.weights is not None:
            if self.weights.shape != (self.observables.shape[0],):
                raise ValueError("weights must have shape (N,)")
        if self.check_boundaries is not None and not callable(self.check_boundaries):
            raise ValueError("check_boundaries must be callable if provided")

    # ------------ basic properties ------------
    @property
    def n_models(self) -> int:
        """Number of models N."""
        return self.observables.shape[0]

    @property
    def n_observables(self) -> int:
        """Number of observables P."""
        return self.observables.shape[1]

    @property
    def n_targets(self) -> int:
        """Number of targets Q."""
        return self.targets.shape[1]

    @property
    def grid_int(self) -> type:
        """Adaptative int type that accounts for model dimensionality."""
        if self.n_models > np.iinfo(np.int32).max:
            return np.int64
        elif self.n_models > np.iinfo(np.int16).max:
            return np.int32
        else:
            return np.int16

    @property
    def grid_float(self) -> type:
        """Adaptative float type that accounts for model precision."""
        return self.observables.dtype

    # ------------ views and subsets ------------
    def get_target(self, key):
        """Return a target column given a key."""
        col = self.target_names.index(key)
        return self.targets[:, col]

    def select(
        self, idx: np.ndarray, observables=None, targets=None, standardisers=True
    ) -> "ModelGrid":
        """
        Return a new ModelGrid containing a subset of models.

        Parameters
        ----------
        idx : ndarray of int, shape (K,)
            Indices to select.
        observables : list of str or list of int, optional, default=None
            Names or indices of observables to include.
        targets : list of str or list of int, optional, default=None
            Names or indices of targets to include.
        standardisers : bool, optional, default=True
            Whether to propagate fitted standardisers.

        Returns
        -------
        sub : ModelGrid
            Subset grid view (copies arrays).
        """
        w = None if self.weights is None else self.weights[idx].copy()
        if observables is None or len(observables) == 0:
            observables = self.observable_names
        if isinstance(observables[0], str):
            observables = [self.observable_names.index(n) for n in observables]
        if targets is None or len(targets) == 0:
            targets = self.target_names
        if isinstance(targets[0], str):
            targets = [self.target_names.index(n) for n in targets]

        sub = ModelGrid(
            observables=self.observables[idx][:, observables].copy(),
            targets=self.targets[idx][:, targets].copy(),
            observable_names=[self.observable_names[o] for o in observables],
            target_names=[self.target_names[t] for t in targets],
            weights=w,
            meta=dict(self.meta),
        )

        if standardisers:
            if self.observable_standardiser.is_fit:
                sub.observable_standardiser.mean = self.observable_standardiser.mean[
                    observables
                ].copy()
                sub.observable_standardiser.sd = self.observable_standardiser.sd[
                    observables
                ].copy()
            if self.target_standardiser.is_fit:
                sub.target_standardiser.mean = self.target_standardiser.mean[
                    targets
                ].copy()
                sub.target_standardiser.sd = self.target_standardiser.sd[targets].copy()

        return sub

    # ------------ standardisation ------------
    def fit_target_standardiser(self, mask: Optional[np.ndarray] = None) -> None:
        """Fit mean/std for target standardisation used by KDTree distances.

        Parameters
        ----------
        mask : ndarray of bool, shape (N,), optional
            If provided, use only masked entries to compute stats.
        """
        X = self.targets if mask is None else self.targets[mask]
        self.target_standardiser.fit(X, ddof=0)

    def transform_targets(self, X: np.ndarray) -> np.ndarray:
        """
        Apply fitted target standardisation.

        Parameters
        ----------
        X : ndarray, shape (..., Q)
            Targets to standardise.

        Returns
        -------
        X_std : ndarray, shape (..., Q)
            Standardised targets.

        Raises
        ------
        RuntimeError
            If fit_target_standardiser has not been called.
        """
        if not self.target_standardiser.is_fit:
            raise RuntimeError(
                "fit_target_standardiser must be called before transform_targets"
            )
        return self.target_standardiser.transform(X)

    def inverse_transform_targets(self, X_std: np.ndarray) -> np.ndarray:
        """
        Inverse of transform_targets.
        """
        if not self.target_standardiser.is_fit:
            raise RuntimeError(
                "fit_target_standardiser must be called before inverse_transform_targets"
            )
        return self.target_standardiser.inverse_transform(X_std)

    def fit_standardiser(self, mask: Optional[np.ndarray] = None) -> None:
        """
        Fit mean and standard deviation for observable standardisation.

        Parameters
        ----------
        mask : ndarray of bool, shape (N,), optional
            If provided, use only masked entries to compute stats.
        """
        X = self.observables if mask is None else self.observables[mask]
        self.observable_standardiser.fit(X, ddof=0)

    def transform_observables(self, X: np.ndarray) -> np.ndarray:
        """
        Apply fitted observable standardisation.

        Parameters
        ----------
        X : ndarray, shape (..., P)
            Observables to standardise.

        Returns
        -------
        X_std : ndarray, shape (..., P)
            Standardised observables.

        Raises
        ------
        RuntimeError
            If fit_standardiser has not been called.
        """
        if not self.observable_standardiser.is_fit:
            raise RuntimeError(
                "fit_standardiser must be called before transform_observables"
            )
        return self.observable_standardiser.transform(X)

    def inverse_transform_observables(self, X_std: np.ndarray) -> np.ndarray:
        """
        Inverse of transform_observables.

        Parameters
        ----------
        X_std : ndarray, shape (..., P)
            Standardised observables to inverse.

        Returns
        -------
        X : ndarray, shape (..., P)
            Inverse transformed observables.

        Raises
        ------
        RuntimeError
            If fit_standardiser has not been called.
        """
        if not self.observable_standardiser.is_fit:
            raise RuntimeError(
                "fit_standardiser must be called before inverse_transform_observables"
            )
        return self.observable_standardiser.inverse_transform(X_std)

    # ------------ KDTree for KNN and interpolation ------------
    def invalidate_knn_index(self) -> None:
        """Invalidate cached KDTree."""
        self._kdtree = None
        self._kdtree_standardized = None

    def get_kdtree(self, *, standardize: bool = True):
        """Get a KDTree either stored in cash or built on the fly."""
        if self._kdtree is not None and self._kdtree_standardized == standardize:
            return self._kdtree

        X = np.asarray(self.targets, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_targets:
            raise ValueError("targets must be 2D (N, Q)")

        if standardize:
            if not self.target_standardiser.is_fit:
                raise RuntimeError(
                    "target_standardiser is not fit. Call fit_target_standardiser() "
                    "before get_kdtree(standardize=True)."
                )
            Xn = self.target_standardiser.transform(X)
        else:
            Xn = X

        self._kdtree = cKDTree(Xn)
        self._kdtree_standardized = standardize
        return self._kdtree

    def in_boundaries(self, targets_query: np.ndarray) -> np.ndarray:
        """
        Check which target queries are within the model grid boundaries.

        Parameters
        ----------
        targets_query : ndarray, shape (M, Q)
            Target query points.

        Returns
        -------
        in_bounds : ndarray of bool, shape (M,)
            Whether each query is within the grid boundaries.
        """
        if self.check_boundaries is None:
            return np.ones(targets_query.shape[0], dtype=bool)
            # in_bounds = np.ones(tq.shape[0], dtype=bool)
            # for j in range(self.n_targets):
            #     tmin = np.min(self.targets[:, j])
            #     tmax = np.max(self.targets[:, j])
            #     in_bounds &= (tq[:, j] >= tmin) & (tq[:, j] <= tmax)
        else:
            in_bounds = self.check_boundaries(targets_query)
        return in_bounds

    def interpolate_observables(
        self,
        targets_query: np.ndarray,
        *,
        method: Optional[str] = None,
        fill_value: float = np.nan,
        k: int = 32,
        p: float = 2.0,
        eps: float = 1e-12,
        standardize: bool = True,
        mode: str = "local_linear",  # "idw" | "local_linear"
        ridge: float = 1e-8,  # Tikhonov regularization for stability
    ) -> np.ndarray:
        """
        Interpolate observables for arbitrary target values using cached KDTree.

        mode="nearest"     : nearest-neighbour lookup
        mode="idw"         : inverse-distance weighted KNN
        mode="local_linear": weighted local affine fit (exact for linear functions)
        """
        tq = np.atleast_2d(np.asarray(targets_query, dtype=float))
        if tq.shape[1] != self.n_targets:
            raise ValueError(
                f"targets_query must have shape (M, {self.n_targets}); got {tq.shape}"
            )

        if method is not None:
            mode = method
        mode = str(mode).lower()
        if mode not in {"nearest", "idw", "local_linear"}:
            raise ValueError(
                f"Unknown mode={mode!r} (use 'nearest', 'idw' or 'local_linear')."
            )

        bad_q = ~np.isfinite(tq).all(axis=1) | ~self.in_boundaries(tq)
        out = np.full((tq.shape[0], self.n_observables), fill_value, dtype=float)
        if bad_q.all():
            return out

        if standardize:
            if not self.target_standardiser.is_fit:
                self.fit_target_standardiser()
            tqn = self.target_standardiser.transform(tq)
        else:
            tqn = tq

        tree = self.get_kdtree(standardize=standardize)
        k_eff = 1 if mode == "nearest" else max(1, min(k, self.n_models))
        if mode == "local_linear" and k_eff < (self.n_targets + 1):
            raise ValueError(
                "local_linear interpolation requires at least n_targets + 1 neighbours "
                f"(got {k_eff}, need {self.n_targets + 1})."
            )

        dists, idx = tree.query(tqn[~bad_q], k=k_eff, workers=-1)
        if k_eff == 1:
            dists = dists[:, None]
            idx = idx[:, None]

        neigh_obs = self.observables[idx]  # (Mgood, k, P)
        neigh_tgt = (
            (self.targets[idx] if not standardize else self.targets_standardized[idx])
            if hasattr(self, "targets_standardized")
            else None
        )

        # Build standardized targets on the fly:
        if neigh_tgt is None:
            if standardize:
                self.targets_standardized = self.target_standardiser.transform(
                    self.targets
                )
                neigh_tgt = self.targets_standardized[idx]
            else:
                neigh_tgt = self.targets[idx]

        obs_q = np.empty((dists.shape[0], self.n_observables), dtype=float)

        zero = dists <= eps
        any_zero = zero.any(axis=1)

        # Exact matches: average exact neighbors
        if np.any(any_zero):
            zrows = np.where(any_zero)[0]
            for r in zrows:
                m = zero[r]
                obs_q[r] = np.nanmean(neigh_obs[r, m, :], axis=0)

        # Non-exact rows
        rows = np.where(~any_zero)[0]
        if rows.size == 0:
            out[~bad_q] = obs_q
            return out

        if mode == "nearest":
            obs_q[rows] = neigh_obs[rows, 0, :]

        elif mode == "idw":
            w = 1.0 / (np.power(dists[rows], p) + eps)
            wsum = np.sum(w, axis=1, keepdims=True)
            w = np.where(wsum > 0, w / wsum, 0.0)
            obs_q[rows] = np.einsum("nk,nkp->np", w, neigh_obs[rows])

        elif mode == "local_linear":
            # Weighted local affine fit per query:
            # y ≈ a + B (x - xq)
            # Implemented as weighted least squares on design matrix [1, (x - xq)]
            D = self.n_targets
            P = self.n_observables

            for rr in rows:
                xq = tqn[~bad_q][rr]  # (D,)
                Xn = neigh_tgt[rr]  # (k, D)
                Yn = neigh_obs[rr]  # (k, P)
                dn = dists[rr]  # (k,)

                # weights (same spirit as IDW)
                w = 1.0 / (np.power(dn, p) + eps)  # (k,)
                # build design matrix: (k, 1+D)
                A = np.empty((k_eff, 1 + D), dtype=float)
                A[:, 0] = 1.0
                A[:, 1:] = Xn - xq[None, :]

                # Apply weights via sqrt(w)
                sw = np.sqrt(w)[:, None]  # (k,1)
                Aw = A * sw  # (k,1+D)
                Yw = Yn * sw  # (k,P)

                # Solve (Aw^T Aw + ridge I) beta = Aw^T Yw
                G = Aw.T @ Aw
                if ridge > 0:
                    G = G + ridge * np.eye(G.shape[0])
                rhs = Aw.T @ Yw

                try:
                    beta = np.linalg.solve(G, rhs)  # (1+D, P)
                    obs_q[rr] = beta[0]  # prediction at xq (delta=0)
                except np.linalg.LinAlgError:
                    logger.warning("Local linear interpolation failed; falling back to IDW.")
                    # Fallback to IDW if ill-conditioned
                    ww = w / (w.sum() + eps)
                    obs_q[rr] = ww @ Yn

        out[~bad_q] = obs_q
        return out

    # ------------ I/O ------------
    @staticmethod
    def _std_to_state(std) -> Dict[str, Any]:
        """Serialisable state for LinearStandardiser."""
        if std is None:
            return {"is_fit": False, "mean": None, "sd": None}
        return {
            "is_fit": bool(getattr(std, "is_fit", False)),
            "mean": None
            if getattr(std, "mean", None) is None
            else np.asarray(std.mean),
            "sd": None if getattr(std, "sd", None) is None else np.asarray(std.sd),
        }

    @staticmethod
    def _state_to_std(state: Mapping[str, Any], std) -> None:
        """Restore LinearStandardiser from state into an existing instance."""
        if not state:
            return
        is_fit = bool(state.get("is_fit", False))
        if not is_fit:
            return
        mean = state.get("mean", None)
        sd = state.get("sd", None)
        if mean is None or sd is None:
            return
        std.mean = np.asarray(mean, dtype=float).copy()
        std.sd = np.asarray(sd, dtype=float).copy()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise grid to a dictionary that is np.savez / JSON-friendly
        (except numpy arrays, which are fine for np.savez).

        Notes
        -----
        - check_boundaries is NOT serialised (it is generally not portable).

          If you need it, store a string key in ``meta`` and resolve via a
          registry.
        """
        return {
            "observables": np.asarray(self.observables),
            "targets": np.asarray(self.targets),
            "observable_names": list(self.observable_names),
            "target_names": list(self.target_names),
            "weights": None if self.weights is None else np.asarray(self.weights),
            "meta": dict(self.meta or {}),
            "standardisers": {
                "observables": self._std_to_state(self.observable_standardiser),
                "targets": self._std_to_state(self.target_standardiser),
            },
            # Explicitly exclude runtime-only/cache/callables:
            "check_boundaries_key": self.meta.get("check_boundaries_key", None),
        }

    @classmethod
    def from_dict(
        cls,
        d: Mapping[str, Any],
        *,
        check_boundaries: Optional["Callable[[np.ndarray], np.ndarray]"] = None,
    ) -> "ModelGrid":
        """
        Build a ModelGrid from a dictionary as produced by to_dict().

        Parameters
        ----------
        d : mapping
            Mapping produced by to_dict().
        check_boundaries : callable, optional
            Optional boundary function to attach at load time.
            (Not serialised by default.)
        """
        grid = cls(
            observables=np.asarray(d["observables"]),
            targets=np.asarray(d["targets"]),
            observable_names=list(d["observable_names"]),
            target_names=list(d["target_names"]),
            weights=None
            if d.get("weights", None) is None
            else np.asarray(d["weights"]),
            meta=dict(d.get("meta", {}) or {}),
            check_boundaries=check_boundaries,
        )

        std = d.get("standardisers", {}) or {}
        cls._state_to_std(std.get("observables", {}), grid.observable_standardiser)
        cls._state_to_std(std.get("targets", {}), grid.target_standardiser)

        return grid

    @classmethod
    def from_fits_table(
        cls,
        path,
        observable_cols=None,
        target_cols=None,
        weight_col=None,
        row_mask=None,
        table_hdu=1,
        memmap=False,
        meta_key="MODELGRID_META",
    ):
        """
        Build a ModelGrid from an Astropy FITS table.

        Priority order:
          1) Auto-load if table.meta contains OBSNAME and TGTNAME written by to_fits_table.
          2) If explicit mappings (observable_cols / target_cols) are provided, use them.
          3) Fallback: load all numeric 1-D columns as observables; no targets.

        Parameters
        ----------
        path : str
            Path to the FITS file.
        observable_cols : list[str] or dict[str, str], optional
            Explicit mapping for observables. If None, try auto-discovery.
        target_cols : list[str] or dict[str, str], optional
            Explicit mapping for targets. If None, try auto-discovery.
        weight_col : str, optional
            Column name for per-model weights. If None, will auto-use "weight" if present.
        row_mask : array_like of bool, optional
            Boolean mask of length N to select rows after reading.
        table_hdu : int or str, optional
            FITS HDU index or name containing the table. Default 1.
        memmap : bool, optional
            astropy Table read memmap. Default False.
        meta_key : str, optional
            Table.meta key with JSON meta blob written by to_fits_table. Default "MODELGRID_META".

        Returns
        -------
        grid : ModelGrid
        """
        try:
            from astropy.table import Table
        except Exception as e:
            raise ImportError("astropy is required for from_fits_table") from e

        t = Table.read(path, hdu=table_hdu, memmap=memmap)

        def _stack_named(names):
            arrs = []
            for n in names:
                if n not in t.colnames:
                    raise KeyError(f"Missing FITS column '{n}'")
                a = np.asarray(t[n])
                if a.ndim > 1:
                    # allow masked scalars, but require 1-D per column
                    a = np.squeeze(a)
                if a.ndim != 1:
                    raise ValueError(
                        f"FITS column '{n}' must be 1-D; got shape {a.shape}"
                    )
                arrs.append(a)
            return np.vstack(arrs).T  # (N, P)

        def _resolve_mapping(cols):
            if cols is None:
                return None, None
            if isinstance(cols, dict):
                names_dst = list(cols.keys())
                names_src = list(cols.values())
            else:
                names_dst = list(cols)
                names_src = list(cols)
            X = _stack_named(names_src)
            return X, names_dst

        # --- 1) Auto-discovery from metadata written by to_fits_table ---
        obs_names_meta = t.meta.get("OBSNAME")
        tgt_names_meta = t.meta.get("TGTNAME")
        meta_blob_raw = t.meta.get(meta_key)
        auto_loaded = False

        if (
            obs_names_meta is not None
            and tgt_names_meta is not None
            and observable_cols is None
            and target_cols is None
        ):
            obs_names = [s for s in str(obs_names_meta).split(",") if s]
            tgt_names = [s for s in str(tgt_names_meta).split(",") if s]
            X = _stack_named(obs_names) if obs_names else np.empty((len(t), 0))
            Y = _stack_named(tgt_names) if tgt_names else np.empty((len(t), 0))
            w = None
            if weight_col is not None and weight_col in t.colnames:
                w = np.asarray(t[weight_col])
            elif "weight" in t.colnames:
                w = np.asarray(t["weight"])
            meta = {}
            if meta_blob_raw is not None:
                try:
                    meta = json.loads(meta_blob_raw)
                except Exception:
                    meta = {}
            auto_loaded = True

        # --- 2) Explicit mappings (if provided) ---
        if not auto_loaded:
            X, obs_names = _resolve_mapping(observable_cols)
            Y, tgt_names = _resolve_mapping(target_cols)

            # Fallback if still None: load all numeric 1-D columns as observables
            if X is None and Y is None:
                num_cols = []
                for name in t.colnames:
                    if name == "weight":
                        continue
                    a = np.asarray(t[name])
                    if a.ndim == 1 and np.issubdtype(a.dtype, np.number):
                        num_cols.append(name)
                if not num_cols:
                    raise ValueError("No numeric 1-D columns found to load")
                obs_names = num_cols
                X = _stack_named(obs_names)
                tgt_names = []
                Y = np.empty((X.shape[0], 0))
            elif X is None or obs_names is None:
                raise ValueError(
                    "observable_cols must be provided if auto-discovery is not available"
                )
            elif Y is None or tgt_names is None:
                # allow targets empty
                tgt_names = []
                Y = np.empty((X.shape[0], 0))

            # weights
            w = None
            if weight_col is not None:
                if weight_col not in t.colnames:
                    raise KeyError(f"weight_col '{weight_col}' not found")
                w = np.asarray(t[weight_col])
            elif "weight" in t.colnames:
                w = np.asarray(t["weight"])

            # meta
            import json

            meta = {}
            if meta_blob_raw is not None:
                try:
                    meta = json.loads(meta_blob_raw)
                except Exception:
                    meta = {}

        # optional row selection
        if row_mask is not None:
            m = np.asarray(row_mask, dtype=bool)
            if m.shape[0] != X.shape[0]:
                raise ValueError("row_mask length mismatch")
            X = X[m]
            Y = Y[m]
            if w is not None:
                w = w[m]

        return cls(
            observables=X,
            targets=Y,
            observable_names=(obs_names if auto_loaded else obs_names),
            target_names=(tgt_names if auto_loaded else tgt_names),
            weights=w,
            meta=dict(meta or {}),
        )

    @classmethod
    def from_hdf5(
        cls,
        path,
        group="/modelgrid",
        observable_dsets=None,
        target_dsets=None,
        weight_dset=None,
        row_slice=None,
    ):
        """
        Build a ModelGrid from an HDF5 file, with automatic discovery.

        Priority order:
          1) Auto-load if the group contains subgroups 'observables' and 'targets'
             as written by to_hdf5 (reads names and meta from attributes).
          2) If explicit mappings (observable_dsets / target_dsets) are provided, use them.
          3) Fallback: load all 1-D numeric datasets directly under the group as
             observables; no targets.

        Parameters
        ----------
        path : str
            Path to the HDF5 file.
        group : str, optional
            Group path to read from. Default "/modelgrid".
        observable_dsets : list[str] or dict[str, str], optional
            Explicit dataset mapping for observables (relative to group).
        target_dsets : list[str] or dict[str, str], optional
            Explicit dataset mapping for targets (relative to group).
        weight_dset : str, optional
            Dataset name for weights (relative to group). If None, will use
            "weights" if present in group.
        row_slice : slice or array_like of int or bool, optional
            Optional selection of rows after reading.

        Returns
        -------
        grid : ModelGrid
        """

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        def _ensure_1d(d, key):
            a = np.asarray(d)
            if a.ndim > 1:
                a = np.squeeze(a)
            if a.ndim != 1:
                raise ValueError(
                    f"HDF5 dataset '{key}' must be 1-D; got shape {a.shape}"
                )
            return a

        with h5py.File(path, "r") as f:
            if group not in f:
                raise KeyError(f"group '{group}' not found in file")
            g = f[group]

            # --- 1) Auto-discovery (preferred layout written by to_hdf5) ---
            auto_loaded = False
            if (
                "observables" in g
                and "targets" in g
                and observable_dsets is None
                and target_dsets is None
            ):
                gob = g["observables"]
                tgt = g["targets"]

                # names from attributes if present, else dataset keys order
                try:
                    obs_names = json.loads(g.attrs.get("observable_names", "[]"))
                    tgt_names = json.loads(g.attrs.get("target_names", "[]"))
                except Exception:
                    obs_names = []
                    tgt_names = []

                if not obs_names:
                    obs_names = list(gob.keys())
                if not tgt_names:
                    tgt_names = list(tgt.keys())

                # read columns
                cols = [
                    _ensure_1d(gob[name][...], f"{group}/observables/{name}")
                    for name in obs_names
                ]
                X = (
                    np.vstack(cols).T
                    if cols
                    else np.empty((g.attrs.get("n_models", 0), 0))
                )

                tcols = [
                    _ensure_1d(tgt[name][...], f"{group}/targets/{name}")
                    for name in tgt_names
                ]
                Y = np.vstack(tcols).T if tcols else np.empty((X.shape[0], 0))

                # weights
                w = None
                if weight_dset is not None and weight_dset in g:
                    w = _ensure_1d(g[weight_dset][...], f"{group}/{weight_dset}")
                elif "weights" in g:
                    w = _ensure_1d(g["weights"][...], f"{group}/weights")

                # meta
                try:
                    meta = json.loads(g.attrs.get("meta", "{}"))
                except Exception:
                    meta = {}

                auto_loaded = True

            # --- 2) Explicit mappings ---
            if not auto_loaded:

                def _full(k):  # relative to group
                    return f"{group.rstrip('/')}/{k.lstrip('/')}"

                def _load_map(dsets):
                    if dsets is None:
                        return None, None
                    if isinstance(dsets, dict):
                        dst = list(dsets.keys())
                        src = [_full(v) for v in dsets.values()]
                    else:
                        dst = list(dsets)
                        src = [_full(v) for v in dsets]
                    cols = []
                    for key, src_key in zip(dst, src):
                        if src_key not in f:
                            raise KeyError(f"Missing dataset '{src_key}'")
                        cols.append(_ensure_1d(f[src_key][...], src_key))
                    X = np.vstack(cols).T
                    return X, dst

                X, obs_names = _load_map(observable_dsets)
                Y, tgt_names = _load_map(target_dsets)

                # Fallback: all 1-D numeric datasets directly under group as observables
                if X is None and Y is None:
                    names = []
                    cols = []
                    for k, d in g.items():
                        if isinstance(d, h5py.Dataset):
                            a = np.asarray(d[...])
                            if (
                                a.ndim == 1
                                and np.issubdtype(a.dtype, np.number)
                                and k != "weights"
                            ):
                                names.append(k)
                                cols.append(a)
                    if not cols:
                        raise ValueError(
                            "No 1-D numeric datasets found to load under the group"
                        )
                    X = np.vstack(cols).T
                    obs_names = names
                    Y = np.empty((X.shape[0], 0))
                    tgt_names = []
                elif X is None or obs_names is None:
                    raise ValueError(
                        "observable_dsets must be provided if auto-discovery is not available"
                    )
                elif Y is None or tgt_names is None:
                    Y = np.empty((X.shape[0], 0))
                    tgt_names = []

                # weights
                w = None
                if weight_dset is not None:
                    key = _full(weight_dset)
                    if key not in f:
                        raise KeyError(f"weight_dset '{key}' not found")
                    w = _ensure_1d(f[key][...], key)
                elif "weights" in g:
                    w = _ensure_1d(g["weights"][...], f"{group}/weights")

                # meta from attribute if present
                try:
                    meta = json.loads(g.attrs.get("meta", "{}"))
                except Exception:
                    meta = {}

        # Optional row selection
        if row_slice is not None:
            X = X[row_slice]
            Y = Y[row_slice]
            if w is not None:
                w = w[row_slice]

        return cls(
            observables=X,
            targets=Y,
            observable_names=obs_names,
            target_names=tgt_names,
            weights=w,
            meta=dict(meta or {}),
        )

    def to_fits_table(
        self,
        path,
        *,
        hdu_name: str = "MODELGRID",
        overwrite: bool = False,
        include_meta: bool = True,
        meta_key: str = "MODELGRID_META",
        fill_value=np.nan,
    ):
        try:
            from astropy.table import Table, Column
            from astropy.io import fits
        except Exception as e:
            raise ImportError("astropy is required for to_fits_table") from e

        N, P = self.observables.shape
        Nt, Q = self.targets.shape
        if N != Nt:
            raise ValueError("observables and targets must have same number of rows")

        tab = Table()

        for j, name in enumerate(self.observable_names):
            col = np.asarray(self.observables[:, j])
            if hasattr(col, "mask"):
                col = np.ma.filled(col, fill_value)
            tab.add_column(Column(col, name=str(name)))

        for j, name in enumerate(self.target_names):
            col = np.asarray(self.targets[:, j])
            if hasattr(col, "mask"):
                col = np.ma.filled(col, fill_value)
            tab.add_column(Column(col, name=str(name)))

        if self.weights is not None:
            w = np.asarray(self.weights)
            if hasattr(w, "mask"):
                w = np.ma.filled(w, fill_value)
            tab.add_column(Column(w, name="weight"))

        tab.meta = dict(tab.meta or {})
        tab.meta.update(
            {
                "NMODELS": int(N),
                "NOBS": int(P),
                "NTGT": int(Q),
                "OBSNAME": ",".join(map(str, self.observable_names)),
                "TGTNAME": ",".join(map(str, self.target_names)),
            }
        )
        if include_meta:
            try:
                tab.meta[meta_key] = json.dumps(self.meta or {}, ensure_ascii=True)
            except Exception:
                tab.meta[meta_key] = json.dumps({}, ensure_ascii=True)

        # Build HDUList: Primary + BinTable
        primary = fits.PrimaryHDU()
        bintable = fits.BinTableHDU(tab, name=str(hdu_name))

        if overwrite or (not os.path.exists(path)):
            fits.HDUList([primary, bintable]).writeto(path, overwrite=overwrite)
            return

        # Replace if HDU exists by name; else append
        with fits.open(path, mode="update") as hdul:
            idx = None
            for i, h in enumerate(hdul):
                if getattr(h, "name", None) == str(hdu_name):
                    idx = i
                    break
            if idx is None:
                hdul.append(bintable)
            else:
                hdul[idx] = bintable
            hdul.flush()

    def to_hdf5(
        self,
        path,
        group="/modelgrid",
        overwrite=False,
        compression="gzip",
        compression_opts=4,
        chunks=True,
        include_meta=True,
    ):
        """
        Save the grid as an HDF5 group with per-column datasets.

        Layout
        ------
        ``{group}/observables/<name>``
            One 1-D dataset per observable.
        ``{group}/targets/<name>``
            One 1-D dataset per target.
        ``{group}/weights``
            Optional 1-D dataset if model weights are present.
        Attributes on ``{group}``
            ``observable_names``, ``target_names``, ``n_models``,
            ``n_observables``, ``n_targets``, and ``meta`` (JSON when
            ``include_meta=True``).

        Parameters
        ----------
        path : str
            Output HDF5 path. File will be created if it does not exist.
        group : str, optional
            Group path where to store the grid. Default "/modelgrid".
        overwrite : bool, optional
            If True and the group exists, it will be deleted and recreated.
            Default False.
        compression : str or None, optional
            Compression for datasets (e.g., "gzip"). Default "gzip".
        compression_opts : int, optional
            Compression level if applicable. Default 4.
        chunks : bool or tuple, optional
            Enable chunking (True) or provide explicit chunk shape. Default True.
        include_meta : bool, optional
            If True, write JSON-encoded meta as a group attribute. Default True.

        Raises
        ------
        ImportError
            If h5py is not available.
        ValueError
            If shapes are inconsistent.
        """
        try:
            import h5py
        except Exception as e:
            raise ImportError("h5py is required for to_hdf5") from e

        N, P = self.observables.shape
        Nt, Q = self.targets.shape
        if N != Nt:
            raise ValueError("observables and targets must have same number of rows")

        # Ensure file exists
        mode = "a" if os.path.exists(path) else "w"
        with h5py.File(path, mode) as f:
            # Handle group
            if group in f:
                if overwrite:
                    del f[group]
                else:
                    raise ValueError(
                        f"group '{group}' already exists; use overwrite=True"
                    )
            g = f.create_group(group)
            gob = g.create_group("observables")
            tgt = g.create_group("targets")

            # Observables datasets
            for j, name in enumerate(self.observable_names):
                d = np.asarray(self.observables[:, j])
                gob.create_dataset(
                    name,
                    data=d,
                    compression=compression,
                    compression_opts=compression_opts,
                    chunks=chunks,
                )

            # Targets datasets
            for j, name in enumerate(self.target_names):
                d = np.asarray(self.targets[:, j])
                tgt.create_dataset(
                    name,
                    data=d,
                    compression=compression,
                    compression_opts=compression_opts,
                    chunks=chunks,
                )

            # Weights
            if self.weights is not None:
                g.create_dataset(
                    "weights",
                    data=np.asarray(self.weights),
                    compression=compression,
                    compression_opts=compression_opts,
                    chunks=chunks,
                )

            # Attributes
            g.attrs["n_models"] = int(N)
            g.attrs["n_observables"] = int(P)
            g.attrs["n_targets"] = int(Q)
            # Store names as JSON to avoid fixed-length string issues
            g.attrs["observable_names"] = json.dumps(
                list(self.observable_names), ensure_ascii=True
            )
            g.attrs["target_names"] = json.dumps(
                list(self.target_names), ensure_ascii=True
            )
            if include_meta:
                try:
                    g.attrs["meta"] = json.dumps(self.meta or {}, ensure_ascii=True)
                except Exception:
                    g.attrs["meta"] = json.dumps({}, ensure_ascii=True)

            g.attrs["standardisers"] = json.dumps(
                {
                    "observables": {
                        "is_fit": self.observable_standardiser.is_fit,
                        "mean": None
                        if not self.observable_standardiser.is_fit
                        else self.observable_standardiser.mean.tolist(),
                        "sd": None
                        if not self.observable_standardiser.is_fit
                        else self.observable_standardiser.sd.tolist(),
                    },
                    "targets": {
                        "is_fit": self.target_standardiser.is_fit,
                        "mean": None
                        if not self.target_standardiser.is_fit
                        else self.target_standardiser.mean.tolist(),
                        "sd": None
                        if not self.target_standardiser.is_fit
                        else self.target_standardiser.sd.tolist(),
                    },
                },
                ensure_ascii=True,
            )

    @classmethod
    def from_pickle(cls, path):
        """
        Load a ModelGrid from a pickle file.

        Parameters
        ----------
        path : str
            Path to the pickle file.

        Returns
        -------
        grid : ModelGrid
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Pickle file does not contain a ModelGrid; got {type(obj)}"
            )
        return obj

    def to_pickle(self, path):
        """
        Save the ModelGrid to a pickle file.

        Parameters
        ----------
        path : str
            Output pickle file path.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_auto(cls, path, **kwargs):
        """
        Auto-load a ModelGrid from a file based on its extension.

        Supported formats:
          .fits, .fit, .fts      FITS table via from_fits_table
          .hdf5, .h5             HDF5 via from_hdf5
          .pkl, .pickle          Pickle via from_pickle

        Parameters
        ----------
        path : str
            Path to the file.
        **kwargs
            Additional keyword arguments passed to the specific loader.

        Returns
        -------
        grid : ModelGrid
        """
        ext = os.path.splitext(path)[1].lower()
        if ext in (".fits", ".fit", ".fts"):
            return cls.from_fits_table(path, **kwargs)
        elif ext in (".hdf5", ".h5"):
            return cls.from_hdf5(path, **kwargs)
        elif ext in (".pkl", ".pickle"):
            return cls.from_pickle(path)
        else:
            raise ValueError(f"Unsupported file extension '{ext}' for auto-loading")


# ---------------------------------------------------------------------
# ModelGrid fitter
# ---------------------------------------------------------------------
class GridFitter:
    """
    Bayesian fitter over a ModelGrid using optinal Prior and Likelihood.

    The fitter computes posterior weights over candidate models by combining
    a user-provided likelihood p(x | model) with a prior p(model). It can
    then histogram targets to obtain marginal posteriors.

    Parameters
    ----------
    grid : ModelGrid
        Training model grid that defines the observable and target spaces.
    likelihood : Likelihood, optional
        Likelihood instance. Default is GaussianProductLikelihood.
    prior : Prior, optional
        Prior instance. Default is FlatPrior.
    use_standardised : bool, optional
        If True and the ModelGrid supports standardisation, evaluation can be
        done in standardised space for internal transforms. The likelihood is
        still called with native units unless you adapt it. Default True.

    Notes
    -----
    Posterior over models is computed as:

    .. math::

       \log w_i = \log p(x \mid \mathrm{model}_i) +
                  \log p(\mathrm{model}_i) +
                  \log w_i^{\mathrm{model}}

    and then normalised to sum to one over the candidate set.
    """

    def __init__(
        self,
        grid,
        likelihood: Optional[Likelihood] = None,
        prior: Optional[Prior] = None,
        use_standardised: bool = True,
    ) -> None:
        self.grid = grid
        self.likelihood = (
            likelihood if likelihood is not None else GaussianProductLikelihood()
        )
        self.prior = prior if prior is not None else FlatPrior()
        self.use_standardised = use_standardised
        if self.use_standardised and hasattr(self.grid, "fit_standardiser"):
            self.grid.fit_standardiser()

    def posterior_over_models(
        self,
        x_native: np.ndarray,
        sigma_native: np.ndarray,
        candidate_idx: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute posterior weights over candidate models for one query.

        This method evaluates log p(x | model) + log p(model) for a set of
        candidate models and returns the normalised weights. If
        use_standardised is True, observables are transformed to the
        grid's standardised space for likelihood evaluation and the input
        uncertainties are mapped to that same space. Priors that depend on
        observables always receive native (non-standardised) observables.

        Parameters
        ----------
        x_native : ndarray, shape (P,)
            Query observables in native units and ordering matching
            grid.observable_names.
        sigma_native : ndarray, shape (P,)
            Per-dimension measurement uncertainties for the query in native
            units. If use_standardised is True, these are internally divided
            by the grid standard deviations so they live in the same space as
            the standardised observables.
        candidate_idx : ndarray of int, optional
            Indices of candidate models to consider. If None, all models in
            the grid are used.

        Returns
        -------
        w : ndarray, shape (Nc,)
            Posterior weights over the candidate set, normalised to sum to 1.

        Raises
        ------
        RuntimeError
            If use_standardised is True and the grid standardiser has not been
            fitted (fit_standardiser must be called before).
        ValueError
            If any derived bandwidth (from sigma) is non positive or not finite,
            depending on the Likelihood implementation.

        Notes
        -----
        1. Consistent spaces: when use_standardised is True, the likelihood is
        evaluated with standardised observables and uncertainties
        (x_eval, X_eval, sigma_eval). When False, evaluation is done in
        native space.
        2. Observable dependent priors: priors that require observables are
        passed native observables to avoid feeding z-scored magnitudes or
        colours into priors defined in native units (for example p(z | VIS)).
        3. Model weights: if the grid has per-model sampling weights, they are
        multiplied into the posterior before normalisation.

        See Also
        --------
        posterior_over_models_fn : Backend routine that combines likelihood and prior.
        """
        # Select model candidates
        idx = np.arange(self.grid.n_models) if candidate_idx is None else candidate_idx
        x_models_native = self.grid.observables[idx]
        Tc = self.grid.targets[idx]
        wc = (
            self.grid.weights[idx]
            if getattr(self.grid, "weights", None) is not None
            else None
        )

        # Map to evaluation space if requested
        if self.use_standardised and hasattr(self.grid, "transform_observables"):
            x_eval = self.grid.transform_observables(x_native)
            x_models_eval = self.grid.transform_observables(x_models_native)
            if not self.grid.observable_standardiser.is_fit:
                raise RuntimeError("standardiser is not fitted")
            sigma_eval = sigma_native / self.grid.observable_standardiser.sd
            # Priors that depend on observables must see native observables
            prior_obs = x_models_native
        else:
            x_eval = x_native
            x_models_eval = x_models_native
            sigma_eval = sigma_native
            prior_obs = x_models_native

        # Detect priors that need observables
        prior_needs_obs = isinstance(self.prior, ObservableDependentPrior)

        w = posterior_over_models_fn(
            x_native=x_eval,
            sigma_native=sigma_eval,
            X_models=x_models_eval,
            targets_models=Tc,
            likelihood=self.likelihood,
            prior=self.prior,
            model_weights=wc,
            prior_needs_observables=prior_needs_obs,
            observables_models=prior_obs,
        )
        return w

    def posterior_over_target(
        self,
        x_native: np.ndarray,
        sigma_native: np.ndarray,
        target_col: int | str,
        bins: np.ndarray,
        candidate_idx: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute a histogrammed posterior over one target.

        Parameters
        ----------
        x_native : ndarray, shape (P,)
            Query observables in native units.
        sigma_native : ndarray, shape (P,)
            Per-dimension uncertainties in native units.
        target_col : int or str
            Column index or name in grid.targets.
        bins : ndarray, shape (K+1,)
            Bin edges.
        candidate_idx : ndarray of int, optional
            Candidate model indices.

        Returns
        -------
        post : ndarray, shape (K,)
            Posterior over bins (sums to one).
        centers : ndarray, shape (K,)
            Bin centers.
        """
        idx = np.arange(self.grid.n_models) if candidate_idx is None else candidate_idx
        w = self.posterior_over_models(
            x_native=x_native, sigma_native=sigma_native, candidate_idx=idx
        )

        if isinstance(target_col, str):
            try:
                j = self.grid.target_names.index(target_col)
            except ValueError as e:
                raise KeyError(f"Unknown target column '{target_col}'") from e
        else:
            j = target_col

        y = self.grid.targets[idx, j]
        hist, _ = np.histogram(y, bins=bins, weights=w, density=True)
        s = np.sum(hist * np.diff(bins))
        post = (
            hist / s
            if s > 0 and np.isfinite(s)
            else np.full_like(hist, 1.0 / hist.size)
        )
        centers = 0.5 * (bins[:-1] + bins[1:])
        return post, centers

    def fit_batch(
        self,
        X_native: np.ndarray,
        SIG_native: np.ndarray,
        binner: Any | None = None,
        n_jobs: int = 1,
        backend: str = "thread",
        stats_for: Optional[Sequence[int | str]] = None,
        stats_bins: Optional[Sequence[np.ndarray]] = None,
        find_multimodal: bool = False,
        use_kde_for_stats: bool = False,
        return_pdf_for_stats: bool = False,
        verbose: bool = True,
        max_memory_gb: Optional[float] = 16.0,
        memcheck_sample: int = 256,
        safety_margin: float = 1.2,
        tasks_per_worker: Optional[int] = None,
        dry_run: bool = False,
        # Prune posterior samples
        posterior_keep_mass: Optional[float] = 0.99,
        posterior_keep_min_candidates: int = 128,
        posterior_keep_max_candidates: Optional[int] = None,
        posterior_keep_ties: bool = False,
        return_mode: str = "iter",  # "iter" | "list"
        # Save results into HDF5
        output_hdf5_path: Optional[str] = None,
        output_hdf5_group: str = "/fit_batch",
        output_hdf5_overwrite: bool = False,
        output_hdf5_compression: Optional[str] = "gzip",
        output_hdf5_compression_opts: int = 4,
        output_hdf5_flush_every: int = 256,
        output_hdf5_write_only: bool = False,
    ) -> Union[Iterator[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Evaluate model posteriors for multiple queries.

        Workflow
        --------
        The computation is split into three stages:

        Step 1 — Candidate selection
            For each query ``m``, find a set of candidate model indices ``cands[m]``
            using the optional ``binner``. If ``binner`` is None or returns an
            empty set, fall back to the full grid. Work is submitted as slices of
            contiguous queries to a thread pool. Slice sizing is heuristic:
            it scales inversely with the observable dimension P so that each
            task has a reasonable amount of work without overscheduling.

        Step 2 — Posterior over models
            For each query ``m``, compute posterior weights over ``cands[m]`` by combining
            likelihood and prior (plus per-model weights). If ``backend="process"``,
            a picklability check is performed and the code falls back to threads
            when needed. Slices are *cost-balanced* using a cheap proxy
            ``cost[m] ≃ len(cands[m]) * P`` so that each future does a similar amount
            of numerical work.

        Step 3 — Per-target statistics
            Optionally, for each requested target, histogram the posterior and
            compute summary stats (mean, std, map, central intervals, quantiles,
            modal count). This stage is NumPy-bound and relatively light, so it
            always uses a thread pool and reuses Step-2 slices (or a similarly
            sized partition).

        Parallel task sizing
        --------------------
        The number of tasks submitted to the executor is controlled by
        ``tasks_per_worker``. By default, it mirrors the current behaviour
        (~ 5 tasks per worker). Larger values create **more, smaller tasks**
        (better load-balancing, higher scheduler overhead). Smaller values create
        **fewer, larger tasks** (lower overhead, potentially less balanced).

        Memory safety
        -------------
        The previous explicit memory pre-check is currently disabled in this
        implementation. For large runs, prefer:
        - candidate selection via ``binner``,
        - posterior truncation controls,
        - ``return_mode="iter"``,
        - HDF5 output with ``output_hdf5_write_only=True`` when appropriate.

        Parameters
        ----------
        X_native : ndarray, shape (M, P)
            Query observables in native units. Order must match the grid.
        SIG_native : ndarray, shape (M, P)
            Per-dimension uncertainties in native units.
        binner : object, optional
            Must expose `.dims` and `.candidates(y_native, sigmas_native)`.
            All configuration (e.g. sigma/target-based selection) is handled
            internally by the binner itself.
        n_jobs : int, optional
            Maximum concurrency (threads or processes, per backend).
        backend : {"thread","process"}, optional
            Execution backend for Step 2 (posterior evaluation). Steps 1 and 3
            always use threads. If objects are not picklable with "process",
            we fall back to "thread".
        stats_for : sequence of {int,str}, optional
            Target columns (indices or names) to summarise. If None, Step 3 is
            skipped.
        stats_bins : sequence of ndarray, optional
            Per-target bin edges. Must have same length as `stats_for` if provided.
        find_multimodal : bool, optional
            If True, attempt to count posterior modes in each 1D histogram.
        return_pdf_for_stats : bool, optional
            If True, return the full (M, K) discrete posterior for each requested
            target (increases memory).
        verbose : bool, optional
            Print progress messages.
        max_memory_gb : float or None, optional
            Maximum allowed additional memory. None disables the check.
        memcheck_sample : int, optional
            Number of pilot queries for candidate-size estimation in the memory
            check (upper bound).
        safety_margin : float, optional
            Multiplier applied to the memory estimate before comparing to the
            allowed/available memory.
        tasks_per_worker : int or None, optional
            Controls how many tasks (slices) are queued per worker for each stage.
            Default None behaves like the previous version (~5). Increase to
            improve load balancing for heterogeneous costs; decrease to reduce
            scheduling overhead.
        dry_run : bool, optional
            If True, perform only the memory check and return an empty dict.

        Returns
        -------
        iterator or list of dict
            Controlled by ``return_mode``:
            - ``"iter"``: returns an iterator yielding one result dict per query.
            - ``"list"``: returns a list with one result dict per query.

            Each result dict contains:
            - ``"m"``: query index
            - ``"candidates"``: kept candidate indices
            - ``"post_models"``: posterior weights over kept candidates
            - ``"level"``: binner level (or ``None``)
            - ``"truncation"``: truncation metadata
            - optional ``"stats"``: per-target summary stats
            - optional ``"posts_target"``: per-target posterior over bins

        Notes
        -----
        When ``use_standardised=True``, likelihoods are evaluated in the grid's
        standardised space, but observable-dependent priors receive *native*
        observables.
        If ``SIG_native`` contains zeros/near-zeros, pre-clip or configure a
        positive floor in your Likelihood to avoid degenerate bandwidths.

        """
        X_native = np.asarray(X_native)
        SIG_native = np.asarray(SIG_native)
        if X_native.ndim != 2 or SIG_native.ndim != 2:
            raise ValueError("X_native and SIG_native must be 2D arrays (M, P)")
        if X_native.shape != SIG_native.shape:
            raise ValueError("X_native and SIG_native must have the same shape (M, P)")

        # M=queries, P=Observables
        M, P = X_native.shape

        # OTF statistics
        if stats_for is not None:
            # Resolve stats target indices once
            stats_j = []
            stats_key = []
            for tname in stats_for:
                if isinstance(tname, str):
                    j = self.grid.target_names.index(tname)
                    key = tname
                else:
                    j = int(tname)
                    key = self.grid.target_names[j]
                stats_j.append(j)
                stats_key.append(key)
            if stats_bins is None or len(stats_for) != len(stats_bins):
                logger.info("stats_bins not provided; creating bins based on grid.")
                stats_bins = [
                    np.linspace(
                        self.grid.targets[:, j].min(),
                        self.grid.targets[:, j].max(),
                        200,
                    )
                    for j in stats_j
                ]
        else:
            stats_j, stats_key = [], []

        # Backend selection
        if backend not in ("thread", "process"):
            raise ValueError("backend must be 'thread' or 'process'")

        if backend == "process":
            try:
                import pickle as _p

                _p.dumps((self.likelihood, self.prior))
            except Exception as e:
                if verbose:
                    logger.warning(
                        "[fit_batch] prior/likelihood are not picklable; "
                        "falling back to thread backend. Reason: %r",
                        e,
                    )
                backend = "thread"

        # Organise tasks per worker and slices
        tpw = 8 if (tasks_per_worker is None) else max(1, int(tasks_per_worker))
        slices = _guess_slices(M, P, max(1, int(n_jobs)), tasks_per_worker=tpw)
        # Pre-flight summary
        if verbose:
            logger.info(
                f"Starting batch fit: #queries={M}, #observables={P}, "
                f"batch backend={backend}, n_jobs={n_jobs}, job tasks={len(slices)}"
            )
            if posterior_keep_mass is not None:
                logger.info(
                    f"  - truncation: keep_mass={posterior_keep_mass}, "
                    f"\n - min candidates={posterior_keep_min_candidates}"
                    f"\n - max={posterior_keep_max_candidates}"
                    f"\n - ties={posterior_keep_ties}"
                )
            if stats_for is not None:
                logger.info("OTF statistics")
                logger.info(
                    f"  - stats_for={stats_key} (return_pdf_for_stats={return_pdf_for_stats})"
                )
        if output_hdf5_path is not None:
            logger.info("Data output")
            logger.info("  - HDF5 directory: %s:%s", output_hdf5_path, output_hdf5_group)

        # if max_memory_gb is not None:
        #     avail = available_memory_bytes()
        #     limit = min(avail, int(max_memory_gb * (1024**3)))
        #     if limit <= 0:
        #         raise MemoryError("[fit_batch] max_memory_gb yields a non-positive limit.")

        #     cand_est = self.grid.n_models
        #     if binner is not None and memcheck_sample != 0:
        #         sample_n = min(M, max(1, int(memcheck_sample)))
        #         if sample_n == M:
        #             sample_idx = range(M)
        #         else:
        #             sample_idx = np.linspace(0, M - 1, sample_n, dtype=int)
        #         sizes = []
        #         for m in sample_idx:
        #             idx, _ = binner.candidates(
        #                 y_native=X_native[m, binner.dims],
        #                 sigmas_native=SIG_native[m, binner.dims],
        #             )
        #             try:
        #                 size = int(idx.size)
        #             except AttributeError:
        #                 size = len(idx)
        #             sizes.append(size if size > 0 else self.grid.n_models)
        #         if sizes:
        #             cand_est = int(np.max(sizes))

        #     kept_est = cand_est
        #     if posterior_keep_max_candidates is not None:
        #         kept_est = min(kept_est, int(posterior_keep_max_candidates))

        #     float_size = np.dtype(float).itemsize
        #     idx_size = np.dtype(self.grid.grid_int).itemsize
        #     per_query = kept_est * (idx_size + float_size)

        #     if stats_for is not None:
        #         nstats = len(stats_j)
        #         per_query += (nstats * nstats) * float_size
        #         if return_pdf_for_stats:
        #             pdf_bins = 0
        #             for bins in stats_bins:
        #                 pdf_bins += max(0, int(np.asarray(bins).size) - 1)
        #             per_query += pdf_bins * float_size

        #     max_slice = max((e - s) for (s, e) in slices) if slices else M
        #     inflight = max(1, int(n_jobs)) if n_jobs != 1 else 1
        #     store_all = (return_mode == "list") and (not output_hdf5_write_only)
        #     if store_all:
        #         est = per_query * M + per_query * max_slice * inflight
        #     else:
        #         est = per_query * max_slice * inflight

        #     est = int(est * float(safety_margin))
        #     if est > limit:
        #         raise MemoryError(
        #             "[fit_batch] Estimated peak additional memory "
        #             f"{est / (1024**3):.2f} GB exceeds limit "
        #             f"{limit / (1024**3):.2f} GB "
        #             f"(available {avail / (1024**3):.2f} GB). "
        #             "Reduce candidates, use return_mode='iter', enable HDF5 "
        #             "write-only output, or increase max_memory_gb."
        #         )
        #     if verbose:
        #         print(
        #             f"[fit_batch] Estimated peak additional memory: "
        #             f"{est / (1024**3):.2f} GB "
        #             f"(limit {limit / (1024**3):.2f} GB, "
        #             f"safety_margin={safety_margin})."
        #         )

        if dry_run:
            return iter(()) if return_mode == "iter" else []

        # Candidate selection helper
        def _select_candidates_helper(m: int) -> Tuple[np.ndarray, Optional[int]]:
            if binner is None:
                return np.arange(self.grid.n_models, dtype=self.grid.grid_int), None
            idx, lev = binner.candidates(
                y_native=X_native[m, binner.dims],
                sigmas_native=SIG_native[m, binner.dims],
            )
            if idx.size == 0:
                idx = np.arange(self.grid.n_models, dtype=self.grid.grid_int)
                lev = lev
            return idx, lev

        # Slice worker
        def _slice_worker(s: int, e: int) -> List[Dict[str, Any]]:
            out = []
            for m in range(s, e):
                cand_idx, lev = _select_candidates_helper(m)
                w_full = self.posterior_over_models(
                    x_native=X_native[m],
                    sigma_native=SIG_native[m],
                    candidate_idx=cand_idx,
                )

                # truncation
                if posterior_keep_mass is not None:
                    cand_kept, w_kept, tmeta = _truncate_posterior_mass(
                        cand_idx,
                        w_full,
                        keep_mass=posterior_keep_mass,
                        min_candidates=posterior_keep_min_candidates,
                        max_candidates=(
                            None
                            if posterior_keep_max_candidates is None
                            else posterior_keep_max_candidates
                        ),
                        keep_ties=posterior_keep_ties,
                    )
                else:
                    cand_kept = cand_idx
                    # ensure normalized
                    # TODO: this check should be done by the master posterior_over_models
                    ssum = np.sum(w_full)
                    if not np.isfinite(ssum) or ssum <= 0:
                        # TODO: raise a warning?
                        w_kept = np.full_like(
                            w_full, 1.0 / max(1, w_full.size), dtype=float
                        )
                    else:
                        w_kept = w_full / ssum
                    tmeta = {
                        "n_before": w_full.size,
                        "n_after": w_kept.size,
                        "keep_mass": 1.0,
                        "mass_kept": 1.0,
                        "mass_dropped": 0.0,
                        "cut_weight": np.nan,
                        "ess_before": 1.0 / np.sum(np.square(w_kept))
                        if w_kept.size
                        else 0.0,
                        "ess_after": 1.0 / np.sum(np.square(w_kept))
                        if w_kept.size
                        else 0.0,
                    }

                # compute stats on truncated posterior
                stats_out = None
                posts_target_out = None
                if stats_for is not None:
                    stats_out = {}
                    if return_pdf_for_stats:
                        posts_target_out = {}

                    Y_cov = self.grid.targets[cand_kept][:, stats_j]
                    if Y_cov.ndim == 1:
                        Y_cov = Y_cov[:, None]
                    cov = np.cov(Y_cov.T, aweights=w_kept)
                    if np.ndim(cov) == 0:
                        cov = np.array([[float(cov)]])
                    elif np.ndim(cov) == 1:
                        cov = np.diag(cov)
                    stats_out["cov"] = cov
                    for key, j, bins in zip(stats_key, stats_j, stats_bins):
                        bins = np.asarray(bins, dtype=float)
                        y = self.grid.targets[cand_kept, j]
                        if use_kde_for_stats:
                            try:
                                kde = gaussian_kde(y, weights=w_kept)
                            except np.linalg.LinAlgError:
                                # Fallback to histogram if KDE fails
                                logger.warning(
                                    "KDE failed for stats target '%s'; falling back to histogram.",
                                    key,
                                )
                                kde = None
                            if kde is None:
                                post, _ = np.histogram(
                                    y, bins=bins, weights=w_kept, density=True
                                )
                            else:
                                post = kde.evaluate(0.5 * (bins[:-1] + bins[1:]))
                        else:
                            post, _ = np.histogram(
                                y, bins=bins, weights=w_kept, density=True
                            )
                        st = pdf_stats(bins, post, find_multimodal=find_multimodal)

                        stats_out[key] = {
                            "mean": st["mean"],
                            "std": st["std"],
                            "map": st["map"],
                            "median": st["median"],
                            "lo68": st["lo68"],
                            "hi68": st["hi68"],
                            "nmodes": (
                                len(st.get("modes", [st["map"]]))
                                if find_multimodal
                                else 1
                            ),
                        }

                        if return_pdf_for_stats:
                            posts_target_out[key] = post
                # Results summary
                r = {
                    "m": int(m),
                    "level": lev,
                    "candidates": cand_kept,
                    "post_models": w_kept,
                    "truncation": tmeta,
                }
                if stats_out is not None:
                    r["stats"] = stats_out
                if posts_target_out is not None:
                    r["posts_target"] = posts_target_out
                out.append(r)
            return out

        # (Optional) output to HDF5 file
        need_h5 = output_hdf5_path is not None
        writer = None
        # Prepare stats specs for writer
        if need_h5:
            stats_specs = None
            if stats_for is not None:
                stats_specs = [
                    GridFitStatsSpec(key=k, bins=b)
                    for k, b in zip(stats_key, stats_bins)
                ]

            writer = GridFitHDF5Writer(
                path=output_hdf5_path,
                group=output_hdf5_group,
                M=M,
                P=P,
                configuration={
                    "M": int(M),
                    "P": int(P),
                    "backend": str(backend),
                    "n_jobs": int(n_jobs),
                    "tasks_per_worker": int(tpw),
                    "posterior_keep_mass": None
                    if posterior_keep_mass is None
                    else float(posterior_keep_mass),
                    "posterior_keep_min_candidates": int(posterior_keep_min_candidates),
                    "posterior_keep_max_candidates": None
                    if posterior_keep_max_candidates is None
                    else int(posterior_keep_max_candidates),
                    "posterior_keep_ties": bool(posterior_keep_ties),
                    "find_multimodal": bool(find_multimodal),
                    "return_pdf_for_stats": bool(return_pdf_for_stats),
                    "stats_for": list(stats_key) if stats_for is not None else None,
                },
                stats_specs=stats_specs,
                return_pdf_for_stats=return_pdf_for_stats,
                overwrite=output_hdf5_overwrite,
                compression=output_hdf5_compression,
                compression_opts=output_hdf5_compression_opts,
                flush_every=output_hdf5_flush_every,
            )

        # Iterator producer (writes HDF5 only from the main thread/process)
        def _iter_results() -> Iterator[Dict[str, Any]]:
            try:
                if n_jobs == 1:
                    for s, e in slices:
                        if verbose:
                            logger.info("  - slice %s:%s", s, e)
                        batch = _slice_worker(s, e)
                        for r in batch:
                            if writer is not None:
                                writer.write(r)
                            if not output_hdf5_write_only:
                                yield r
                else:
                    Executor = (
                        ThreadPoolExecutor
                        if backend == "thread"
                        else ProcessPoolExecutor
                    )
                    with Executor(max_workers=max(1, int(n_jobs))) as ex:
                        futs = [ex.submit(_slice_worker, s, e) for (s, e) in slices]
                        for fut in as_completed(futs):
                            batch = fut.result()
                            for r in batch:
                                if writer is not None:
                                    writer.write(r)
                                if not output_hdf5_write_only:
                                    yield r

                if verbose:
                    logger.info("fit_batch complete.")
            finally:
                if writer is not None:
                    try:
                        writer.close()
                    except Exception:
                        pass

        if return_mode == "iter":
            return _iter_results()
        elif return_mode == "list":
            return list(_iter_results())
        else:
            raise ValueError("return_mode must be 'iter' or 'list'")

    def corner_for_targets(
        self,
        x_native: np.ndarray,
        sigma_native: np.ndarray,
        target_cols: Sequence[int | str],
        true_target_vals: Optional[np.ndarray] = None,
        candidate_idx: Optional[np.ndarray] = None,
        bins: int | Sequence[int | np.ndarray] = 50,
        figsize: Optional[Tuple[float, float]] = None,
        suptitle: Optional[str] = None,
        color: Optional[str] = None,
        kappa_sigma_edges: float = 5.0,
        alpha_hist: float = 0.6,
        alpha_mesh: float = 1.0,
        quantiles: Tuple[float, float, float] = (0.16, 0.5, 0.84),
    ) -> Tuple[plt.Figure, np.ndarray, Dict[str, Any]]:
        """
        Corner plot for selected targets using model posterior weights.

        Parameters
        ----------
        x_native : ndarray, shape (P,)
        sigma_native : ndarray, shape (P,)
        target_cols : sequence of int or str
            Target columns to include.
        true_target_vals: ndarray of float, optional
            True target values.
        candidate_idx : ndarray of int, optional
        bins : int or sequence
            Common bin count, or per-dimension specs (int or edges).
        figsize : tuple, optional
        suptitle : str, optional
        color : str, optional
        alpha_hist : float, optional
        alpha_mesh : float, optional
        quantiles : tuple, optional

        Returns
        -------
        fig, axes, summary : Figure, Axes array, dict
        """
        idx = np.arange(self.grid.n_models) if candidate_idx is None else candidate_idx
        w = self.posterior_over_models(
            x_native=x_native, sigma_native=sigma_native, candidate_idx=idx
        )

        # Resolve columns and names
        cols, names = [], []
        for tc in target_cols:
            if isinstance(tc, str):
                j = self.grid.target_names.index(tc)
            else:
                j = int(tc)
            cols.append(j)
            names.append(self.grid.target_names[j])

        Y = self.grid.targets[idx][:, cols]
        w = w / np.sum(w)
        best_fit = np.argmax(w)
        best_y = Y[best_fit]
        mean_y = np.sum(Y * w[:, None], axis=0)
        D = len(cols)
        if isinstance(bins, int):
            bins_list = [bins] * D
        else:
            bins_list = list(bins)
            if len(bins_list) == 1:
                bins_list = bins_list * D
            assert len(bins_list) == D

        if figsize is None:
            figsize = (2.2 * D, 2.2 * D)
        fig, axes = plt.subplots(D, D, figsize=figsize, squeeze=False, sharex="col")

        # Weighted summaries
        q = np.zeros((D, 3))
        mu = np.zeros(D)
        for d in range(D):
            q[d] = weighted_quantiles(Y[:, d], w, quantiles)
            mu[d] = np.sum(w * Y[:, d])

        # Diagonals
        all_edges = []
        for i in range(D):
            ax = axes[i, i]
            bi = bins_list[i]
            if isinstance(bi, int):
                lo, mid, hi = weighted_quantiles(Y[:, i], w, (0.16, 0.5, 0.84))
                lo = lo if np.isfinite(lo) else np.nanmin(Y[:, i])
                hi = hi if np.isfinite(hi) else np.nanmax(Y[:, i])
                min_v = max(mid - kappa_sigma_edges * (mid - lo), Y[:, i].min())
                max_v = min(mid + kappa_sigma_edges * (hi - mid), Y[:, i].max())
                if min_v == max_v:
                    min_v = Y[:, i].min() * 0.9
                    max_v = Y[:, i].max() * 1.1
                edges = np.linspace(min_v, max_v, bi + 1)
            else:
                edges = np.asarray(bi)

            all_edges.append(edges)
            hist, _ = np.histogram(Y[:, i], bins=edges, weights=w)
            width = np.diff(edges)
            dens = hist / (np.sum(hist) * width if np.sum(hist) > 0 else width)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.fill_between(centers, 0, dens, step="mid", alpha=alpha_hist, color=color)
            ax.plot(centers, dens, lw=1.0, color=color)
            if true_target_vals is not None:
                ax.axvline(true_target_vals[i], color="r", label="True")
            for qv in q[i]:
                ax.axvline(qv, ls="--", lw=1.0, color="gold")

            tlt = ", ".join([f"{v:.3f}" for v in q[i]])
            ax.set_title(tlt)
            ax.axvline(best_y[i], color="fuchsia", lw=1.0, label="Max-like")
            ax.axvline(mean_y[i], color="lime", lw=1.0, label="Mean")

        ax.legend(fontsize=8)
        # Lower triangle
        for i in range(1, D):
            for j in range(i):
                ax = axes[i, j]
                xedges = all_edges[j]
                yedges = all_edges[i]
                H, xe, ye = np.histogram2d(
                    Y[:, j], Y[:, i], bins=[xedges, yedges], weights=w
                )
                xb = (xedges[:-1] + xedges[1:]) / 2
                yb = (yedges[:-1] + yedges[1:]) / 2
                max_val = np.nanmax(H)
                # ax.pcolormesh(xe, ye, H.T, shading="auto", alpha=alpha_mesh,
                #               norm=LogNorm(vmin=max_val / 1e5, vmax=max_val),
                #               cmap="hot_r")
                frac = enclosed_fraction_map(H, xedges=xedges, yedges=yedges)
                ax.contourf(
                    xb, yb, frac.T, cmap="Greys", levels=[0.01, 0.05, 0.32, 0.5, 1]
                )

                ax.scatter(best_y[j], best_y[i], ec="fuchsia", fc="none")
                ax.scatter(mean_y[j], mean_y[i], ec="lime", fc="none")
                if true_target_vals is not None:
                    ax.scatter(
                        true_target_vals[j], true_target_vals[i], ec="r", fc="none"
                    )
                if i == D - 1:
                    ax.set_xlabel(names[j])
                if j == 0:
                    ax.set_ylabel(names[i])
                else:
                    ax.set_yticklabels([])

        # Hide upper triangle
        for i in range(D):
            for j in range(i + 1, D):
                axes[i, j].axis("off")
        if suptitle:
            fig.suptitle(suptitle)
        else:
            fig.suptitle(f"No. of\ncandidate models: {w.size}", fontsize="small")

        fig.tight_layout()

        # Return summary
        cov = (Y - mu).T @ ((Y - mu) * w[:, None])
        cov = 0.5 * (cov + cov.T)
        sd = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = cov / np.outer(sd, sd)
        corr[np.isnan(corr)] = 0.0

        summary = {"names": names, "q": q, "mean": mu, "cov": cov, "corr": corr}
        return fig, axes, summary


@dataclass
class GridFitStatsSpec:
    """
    Specification for one stats target to be written.

    Parameters
    ----------
    key : str
        Target name (used as group name under /stats/<key>).
    bins : ndarray, shape (K+1,)
        Bin edges used to build the discrete posterior.
    """

    key: str
    bins: np.ndarray


class GridFitHDF5Writer:
    """
    HDF5 writer for GridFitter.fit_batch streaming outputs.

    Layout written under ``<group>``::

      /configuration           attrs['json'] with a JSON blob
      /index/m                 (M,) int64  [0..M-1]
      /index/status            (M,) int8   [0=not written, 1=ok, -1=failed]
      /truncation/*            (M,) scalar datasets (level, n_before, n_after, ...)
      /candidates/candidates   (M,) vlen int64
      /candidates/post_models  (M,) vlen float64
      /stats/<key>/*           per-target scalar arrays (M,) and optional posts_target (M,K)

    Notes
    -----
    - Writes are done by index ``m`` in each result dict, so results may
      arrive unsorted.
    - Designed for use from the main thread/process: collect worker results,
      then call ``writer.write(r)`` as they arrive.
    """

    def __init__(
        self,
        path: str,
        group: str,
        *,
        M: int,
        P: int,
        configuration: Optional[Dict[str, Any]] = None,
        stats_specs: Optional[Sequence[GridFitStatsSpec]] = None,
        return_pdf_for_stats: bool = False,
        overwrite: bool = False,
        compression: Optional[str] = "gzip",
        compression_opts: int = 4,
        flush_every: int = 256,
        dtype_posts_target=np.float32,
        dtype_post_models=np.float64,
    ) -> None:
        self.path = str(path)
        self.group = str(group)
        self.M = int(M)
        self.P = int(P)
        self.return_pdf_for_stats = bool(return_pdf_for_stats)
        self.overwrite = bool(overwrite)
        self.compression = compression
        self.compression_opts = int(compression_opts)
        self.flush_every = int(flush_every) if flush_every is not None else 0
        self.dtype_posts_target = np.dtype(dtype_posts_target)
        self.dtype_post_models = np.dtype(dtype_post_models)

        self._written = 0
        self._is_closed = True

        self._stats_specs: Tuple[GridFitStatsSpec, ...] = tuple(stats_specs or ())

        # dataset handles (set in _prepare)
        self.h5: Optional[h5py.File] = None
        self.g: Optional[h5py.Group] = None
        self.ds_status = None
        self.ds_level = None
        self.ds_n_before = None
        self.ds_n_after = None
        self.ds_mass_kept = None
        self.ds_mass_dropped = None
        self.ds_cut_weight = None
        self.ds_ess_before = None
        self.ds_ess_after = None
        self.ds_candidates = None
        self.ds_post_models = None
        self._stats_handles: Dict[Tuple[str, str], h5py.Dataset] = {}
        self.ds_cov = None
        self._cov_dim = 0
        self._cov_keys: Tuple[str, ...] = ()

        self._prepare(configuration=configuration or {})

    # --------------------------
    # Context manager support
    # --------------------------
    def __enter__(self) -> "GridFitHDF5Writer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # If an exception happened, we still close cleanly.
        self.close()

    # --------------------------
    # Public API
    # --------------------------
    def write(self, r: Dict[str, Any]) -> None:
        """
        Write one result dict produced by the streaming fit_batch.

        Required keys:
          - 'm' (int)
          - 'level' (int or None)
          - 'candidates' (ndarray int)
          - 'post_models' (ndarray float, sums to 1)
          - 'truncation' (dict with n_before, n_after, mass_kept, mass_dropped, cut_weight, ess_before, ess_after)

        Optional:
          - 'stats' : dict keyed by target name
          - 'posts_target' : dict keyed by target name with (K,) posterior arrays (if enabled)
        """
        if self._is_closed:
            raise RuntimeError("Writer is closed.")

        m = int(r["m"])
        if m < 0 or m >= self.M:
            raise IndexError(f"m={m} out of bounds for M={self.M}")

        # vlen candidates + post_models
        self.ds_candidates[m] = np.asarray(r["candidates"], dtype=np.int64)
        self.ds_post_models[m] = np.asarray(
            r["post_models"], dtype=self.dtype_post_models
        )

        # truncation + level
        lev = r.get("level", None)
        self.ds_level[m] = -1 if lev is None else int(lev)

        t = r.get("truncation", {}) or {}
        self.ds_n_before[m] = int(t.get("n_before", -1))
        self.ds_n_after[m] = int(t.get("n_after", -1))
        self.ds_mass_kept[m] = np.float32(t.get("mass_kept", np.nan))
        self.ds_mass_dropped[m] = np.float32(t.get("mass_dropped", np.nan))
        self.ds_cut_weight[m] = np.float32(t.get("cut_weight", np.nan))
        self.ds_ess_before[m] = np.float32(t.get("ess_before", np.nan))
        self.ds_ess_after[m] = np.float32(t.get("ess_after", np.nan))

        # stats (scalars)
        stats = r.get("stats", None)
        if stats is not None:
            for key, st in stats.items():
                if key == "cov":
                    self._write_stat_cov(m, st)
                else:
                    self._write_stats_scalars(m, key, st)

        # stats (posterior over bins)
        if self.return_pdf_for_stats:
            posts_target = r.get("posts_target", None)
            if posts_target is not None:
                for key, post in posts_target.items():
                    self._write_posts_target(m, key, post)

        # mark status OK
        self.ds_status[m] = 1

        self._written += 1
        if self.flush_every and (self._written % self.flush_every == 0):
            self.flush()

    def mark_failed(self, m: int) -> None:
        """
        Mark one object as failed (status=-1). Useful if you catch exceptions externally.
        """
        if self._is_closed:
            raise RuntimeError("Writer is closed.")
        m = int(m)
        if m < 0 or m >= self.M:
            raise IndexError(f"m={m} out of bounds for M={self.M}")
        self.ds_status[m] = -1
        self._written += 1
        if self.flush_every and (self._written % self.flush_every == 0):
            self.flush()

    def flush(self) -> None:
        if self.h5 is not None and not self._is_closed:
            self.h5.flush()

    def close(self) -> None:
        if self._is_closed:
            return
        try:
            self.flush()
        finally:
            try:
                if self.h5 is not None:
                    self.h5.close()
            finally:
                self.h5 = None
                self.g = None
                self._is_closed = True

    def _prepare(self, configuration: Dict[str, Any]) -> None:
        mode = "a" if os.path.exists(self.path) else "w"
        self.h5 = h5py.File(self.path, mode)
        self._is_closed = False

        grp_path = self.group
        if grp_path in self.h5:
            if self.overwrite:
                del self.h5[grp_path]
            else:
                raise ValueError(
                    f"HDF5 group '{grp_path}' exists in '{self.path}'. "
                    "Set overwrite=True to replace it."
                )
        self.g = self.h5.create_group(grp_path)

        # /configuration
        gc = self.g.create_group("configuration")
        cfg = dict(configuration or {})
        cfg.setdefault("M", int(self.M))
        cfg.setdefault("P", int(self.P))
        cfg.setdefault("group", str(self.group))
        cfg.setdefault("return_pdf_for_stats", bool(self.return_pdf_for_stats))
        cfg.setdefault(
            "stats_keys",
            [s.key for s in self._stats_specs] if self._stats_specs else None,
        )
        gc.attrs["json"] = json.dumps(cfg, ensure_ascii=True)

        # /index
        gi = self.g.create_group("index")
        gi.create_dataset("m", data=np.arange(self.M, dtype=np.int64))
        self.ds_status = gi.create_dataset("status", shape=(self.M,), dtype=np.int8)
        self.ds_status[...] = 0  # 0=not written, 1=ok, -1=failed

        # /truncation
        gt = self.g.create_group("truncation")
        self.ds_level = gt.create_dataset("level", shape=(self.M,), dtype=np.int32)
        self.ds_level[...] = -1

        self.ds_n_before = gt.create_dataset(
            "n_before", shape=(self.M,), dtype=np.int32
        )
        self.ds_n_before[...] = -1
        self.ds_n_after = gt.create_dataset("n_after", shape=(self.M,), dtype=np.int32)
        self.ds_n_after[...] = -1

        self.ds_mass_kept = gt.create_dataset(
            "mass_kept", shape=(self.M,), dtype=np.float32
        )
        self.ds_mass_kept[...] = np.nan
        self.ds_mass_dropped = gt.create_dataset(
            "mass_dropped", shape=(self.M,), dtype=np.float32
        )
        self.ds_mass_dropped[...] = np.nan
        self.ds_cut_weight = gt.create_dataset(
            "cut_weight", shape=(self.M,), dtype=np.float32
        )
        self.ds_cut_weight[...] = np.nan
        self.ds_ess_before = gt.create_dataset(
            "ess_before", shape=(self.M,), dtype=np.float32
        )
        self.ds_ess_before[...] = np.nan
        self.ds_ess_after = gt.create_dataset(
            "ess_after", shape=(self.M,), dtype=np.float32
        )
        self.ds_ess_after[...] = np.nan

        # /candidates (variable-length arrays)
        gcand = self.g.create_group("candidates")
        vlen_i64 = h5py.vlen_dtype(np.dtype("int64"))
        vlen_post = h5py.vlen_dtype(self.dtype_post_models)
        self.ds_candidates = gcand.create_dataset(
            "candidates", shape=(self.M,), dtype=vlen_i64
        )
        self.ds_post_models = gcand.create_dataset(
            "post_models", shape=(self.M,), dtype=vlen_post
        )

        # /stats
        if self._stats_specs:
            gs = self.g.create_group("stats")
            self._stats_handles = {}
            for spec in self._stats_specs:
                key = str(spec.key)
                bins = np.asarray(spec.bins, dtype=float)
                if bins.ndim != 1 or bins.size < 2:
                    raise ValueError(f"Invalid bins for stats target '{key}'")

                tg = gs.create_group(key)
                K = int(bins.size - 1)
                tg.create_dataset("bins", data=bins)
                # Scalars per object
                for name, dtype, fill in [
                    ("mean", np.float32, np.nan),
                    ("std", np.float32, np.nan),
                    ("map", np.float32, np.nan),
                    ("median", np.float32, np.nan),
                    ("lo68", np.float32, np.nan),
                    ("hi68", np.float32, np.nan),
                ]:
                    d = tg.create_dataset(name, shape=(self.M,), dtype=dtype)
                    d[...] = fill
                    self._stats_handles[(key, name)] = d

                d_nm = tg.create_dataset("nmodes", shape=(self.M,), dtype=np.int16)
                d_nm[...] = 0
                self._stats_handles[(key, "nmodes")] = d_nm

                if self.return_pdf_for_stats:
                    dpt = tg.create_dataset(
                        "posts_target",
                        shape=(self.M, K),
                        dtype=self.dtype_posts_target,
                        compression=self.compression,
                        compression_opts=self.compression_opts,
                        chunks=True,
                    )
                    dpt[...] = np.nan
                    self._stats_handles[(key, "posts_target")] = dpt

            # --------------------------
            # /stats/cov : per-object covariance
            # --------------------------
            self._cov_keys = tuple(str(s.key) for s in self._stats_specs)
            self._cov_dim = len(self._cov_keys)

            gcov = gs.create_group("cov")
            gcov.create_dataset("keys", data=np.array(self._cov_keys, dtype="S"))

            # Covariance per object: (M, D, D)
            # Chunk across objects; full matrix per chunk row is typically fine.
            self.ds_cov = gcov.create_dataset(
                "cov",
                shape=(self.M, self._cov_dim, self._cov_dim),
                dtype=np.float32,
                compression=self.compression,
                compression_opts=self.compression_opts,
                chunks=(min(self.M, 64), self._cov_dim, self._cov_dim),
            )
            self.ds_cov[...] = np.nan

        # Make sure initial metadata is on disk
        self.flush()

    # --------------------------
    # Internal: stats writing
    # --------------------------
    def _write_stats_scalars(self, m: int, key: str, st: Dict[str, Any]) -> None:
        """
        Write scalar stats for one target.
        Missing targets are ignored (allows partial stats).
        """
        key = str(key)
        # Only write keys that exist in file
        if (key, "mean") not in self._stats_handles:
            return

        # Support both the compact dict (as produced by our fit_batch) and
        # extended dicts (ignoring unknown fields).
        self._stats_handles[(key, "mean")][m] = np.float32(st.get("mean", np.nan))
        self._stats_handles[(key, "std")][m] = np.float32(st.get("std", np.nan))
        self._stats_handles[(key, "map")][m] = np.float32(st.get("map", np.nan))
        self._stats_handles[(key, "median")][m] = np.float32(st.get("median", np.nan))
        self._stats_handles[(key, "lo68")][m] = np.float32(st.get("lo68", np.nan))
        self._stats_handles[(key, "hi68")][m] = np.float32(st.get("hi68", np.nan))
        self._stats_handles[(key, "nmodes")][m] = np.int16(st.get("nmodes", 0))

    def _write_posts_target(self, m: int, key: str, post: Any) -> None:
        """
        Write the discrete posterior over bins for one target (shape (K,)).
        If the dataset doesn't exist or shapes mismatch, raises ValueError.
        """
        key = str(key)
        hkey = (key, "posts_target")
        if hkey not in self._stats_handles:
            return
        dpt = self._stats_handles[hkey]
        arr = np.asarray(post, dtype=self.dtype_posts_target)
        if arr.ndim != 1 or arr.shape[0] != dpt.shape[1]:
            raise ValueError(
                f"posts_target shape mismatch for '{key}': got {arr.shape}, expected ({dpt.shape[1]},)"
            )
        dpt[m, :] = arr

    def _write_stat_cov(self, m: int, st: Any) -> None:
        """
        Write the covariance matrix for object m.

        Expected input formats
        ----------------------
        st can be:
          1) a dict with key "cov" holding an array-like (D, D), OR
          2) directly an array-like (D, D)

        Where D == len(self._stats_specs) and matches /stats/cov/keys ordering.

        If the covariance dataset is not present (e.g. no stats_specs),
        this is a no-op.
        """
        if self.ds_cov is None:
            return

        # Accept dict or raw array
        if isinstance(st, dict) and ("cov" in st):
            cov = st["cov"]
        else:
            cov = st

        arr = np.asarray(cov, dtype=np.float32)

        D = int(self._cov_dim)
        if arr.ndim != 2 or arr.shape != (D, D):
            raise ValueError(
                f"cov shape mismatch: got {arr.shape}, expected ({D}, {D}). "
                f"Target order: {self._cov_keys}"
            )

        # Optional: enforce symmetry softly (avoid surprises)
        # Comment out if you want to store exactly what you compute.
        arr = 0.5 * (arr + arr.T)

        self.ds_cov[m, :, :] = arr
