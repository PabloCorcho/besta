#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 16:16:53 2025

@author: pcorchoc
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Mapping
from abc import ABC, abstractmethod
from itertools import product

import os
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.table import Table, Column
from astropy.io import fits
import h5py

from besta.grid.prob import (
    Prior,
    FlatPrior,
    ObservableDependentPrior,
    Likelihood,
    GaussianProductLikelihood,
    posterior_over_models as posterior_over_models_fn,
)
from besta.postprocess import (
    compute_fraction_from_map,
    pit_from_discrete_posterior,
    hist_stats,
    photoz_metrics,
    weighted_quantiles,
)

from besta.io import available_memory_bytes

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


def _chunk_ranges(n: int, batch_size: int):
    """
    Yield contiguous half-open index ranges covering [0, n).

    Parameters
    ----------
    n : int
        Total number of items.
    batch_size : int
        Maximum number of items per chunk. If None or invalid
        (<= 0 or >= n), a single chunk (0, n) is yielded.

    Yields
    ------
    start, stop : tuple of int
        Half-open slice indices for the current chunk.
    """
    if batch_size is None or batch_size <= 0 or batch_size >= n:
        yield 0, n
        return
    start = 0
    while start < n:
        stop = min(n, start + batch_size)
        yield start, stop
        start = stop


def _fit_batch_cands_worker(args):
    """
    Worker for batch fit step 1: select candidate models for one query.

    Parameters
    ----------
    args : tuple
        (m, X_native, SIG_native, binner, target_factor, expand_factor, grid_n)

    Returns
    -------
    m : int
        Query index.
    idx : ndarray of int
        Candidate indices (unique, sorted).
    lev : int or None
        Level used by the binner, or None if no binner.
    """
    (m, X_native, SIG_native, binner, target_factor, expand_factor, grid_n) = args
    if binner is None:
        idx = np.arange(grid_n)
        lev = None
    else:
        y_sub = X_native[binner.dims]
        s_sub = SIG_native[binner.dims]
        idx, lev = binner.candidates(
            y_native=y_sub,
            sigmas_native=s_sub,
            target_factor=target_factor,
            expand_factor=expand_factor,
        )
        if idx.size == 0:
            idx = np.arange(grid_n)
    return m, idx, lev


def _fit_batch_post_worker(args):
    """
    Worker for batch fit step 2: compute posterior weights for one query.

    Parameters
    ----------
    args : tuple
        (m, X_m, SIG_m, idx, grid_dict, use_std, likelihood, prior, is_obs_dep_prior)
        where:
          m : int
              Query index (for ordering results).
          X_m : ndarray, shape (P,)
              Query observables in native units.
          SIG_m : ndarray, shape (P,)
              Per-dimension uncertainties in native units.
          idx : ndarray of int
              Candidate model indices.
          grid_dict : dict
              Serializable dictionary with keys:
              "observables", "targets", "weights", "_obs_mu", "_obs_sd".
          use_std : bool
              If True, evaluate in standardised space.
          likelihood : Likelihood
              Likelihood instance (must be picklable for process backend).
          prior : Prior
              Prior instance (must be picklable for process backend).
          is_obs_dep_prior : bool
              Whether the prior depends on observables.

    Returns
    -------
    m : int
        The input query index.
    w : ndarray, shape (Nc,)
        Normalised posterior weights over candidate models.
    """
    (m, X_m, SIG_m, idx, grid_dict, use_std, likelihood, prior, is_obs_dep_prior) = args

    Xc_native = grid_dict["observables"][idx]
    Tc = grid_dict["targets"][idx]
    wc = grid_dict["weights"][idx] if grid_dict["weights"] is not None else None

    obs_mu = grid_dict.get("_obs_mu", None)
    obs_sd = grid_dict.get("_obs_sd", None)

    if use_std and (obs_mu is not None) and (obs_sd is not None):
        x_eval = (X_m - obs_mu) / obs_sd
        X_eval = (Xc_native - obs_mu) / obs_sd
        sigma_eval = SIG_m / obs_sd
        prior_obs = Xc_native  # native for priors depending on observables
    else:
        x_eval = X_m
        X_eval = Xc_native
        sigma_eval = SIG_m
        prior_obs = Xc_native

    # Prior
    if is_obs_dep_prior:
        logP = prior.log_prob_for_models(Tc, observables=prior_obs)
    else:
        logP = prior.log_prob_for_models(Tc)

    # Likelihood
    logL = likelihood.log_likelihood(x_eval, sigma_eval, X_eval)

    logw = logL + logP
    if wc is not None:
        with np.errstate(divide="ignore"):
            logw = logw + np.log(np.clip(wc, 1e-300, np.inf))
    a = np.max(logw)
    w = np.exp(logw - a)
    s = w.sum()
    w = w / s if s > 0 and np.isfinite(s) else np.full_like(w, 1.0 / w.size)
    return m, w


def _fit_batch_stats_worker(args):
    """
    Worker for batch fit step 3: compute one target's histogram and stats for one query.

    Parameters
    ----------
    args : tuple
        (m, j, bins, centers, candidates_m, posts_m, targets)
        where j is the target column index.

    Returns
    -------
    m : int
        Query index.
    post : ndarray, shape (K,)
        Normalised posterior over target bins.
    st : dict
        Summary dictionary from `hist_stats(centers, post, ...)`.
    """
    (m, j, bins, centers, candidates_m, posts_m, targets) = args
    y = targets[candidates_m, j]
    hist, _ = np.histogram(y, bins=bins, weights=posts_m, density=False)
    s = hist.sum()
    post = hist / s if s > 0 else np.full_like(hist, 1.0 / hist.size)
    st = hist_stats(centers, post, find_multimodal=False)
    return m, post, st


def _guess_slices_step_1(n_objects, n_observables, n_jobs, tasks_per_worker=6):
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


def _make_cost_balanced_slices(costs, n_jobs, tasks_per_worker=6):
    """
    Partition queries into ~n_jobs*tasks_per_worker slices with roughly equal total cost.
    costs: 1D array-like of cost per query (e.g., len(cands[m]) * P)
    """
    M = len(costs)
    T = max(1, n_jobs * tasks_per_worker)
    total = float(np.sum(costs)) if M else 0.0
    target = total / T if T > 0 else total

    slices = []
    s = 0
    acc = 0.0
    for m in range(M):
        acc += float(costs[m])
        # cut when we exceed ~target (but ensure at least one element)
        if acc >= target and m + 1 - s > 0:
            slices.append((s, m + 1))
            s = m + 1
            acc = 0.0
    if s < M:
        slices.append((s, M))
    # If we ended up with fewer than T slices, that’s fine. Executor will still load balance.
    return slices


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
    """

    observables: np.ndarray
    targets: np.ndarray
    observable_names: List[str]
    target_names: List[str]
    weights: Optional[np.ndarray] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    # Cached stats for standardisation
    _obs_mu: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _obs_sd: Optional[np.ndarray] = field(default=None, init=False, repr=False)

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

    # ------------ views and subsets ------------
    def select(self, idx: np.ndarray, observables=None, targets=None) -> "ModelGrid":
        """
        Return a new ModelGrid containing a subset of models.

        Parameters
        ----------
        idx : ndarray of int, shape (K,)
            Indices to select.

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

        return ModelGrid(
            observables=self.observables[idx][:, observables].copy(),
            targets=self.targets[idx][:, targets].copy(),
            observable_names=[self.observable_names[o] for o in observables],
            target_names=[self.target_names[t] for t in targets],
            weights=w,
            meta=dict(self.meta),
        )

    # ------------ standardisation ------------
    def fit_standardiser(self, mask: Optional[np.ndarray] = None) -> None:
        """
        Fit mean and standard deviation for observable standardisation.

        Parameters
        ----------
        mask : ndarray of bool, shape (N,), optional
            If provided, use only masked entries to compute stats.
        """
        X = self.observables if mask is None else self.observables[mask]
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=0)
        sd = np.where(sd == 0.0, 1.0, sd)
        self._obs_mu = mu
        self._obs_sd = sd

    def transform_observables(self, X: np.ndarray) -> np.ndarray:
        """
        Apply fitted standardisation to a matrix of observables.

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
        if self._obs_mu is None or self._obs_sd is None:
            raise RuntimeError(
                "fit_standardiser must be called before transform_observables"
            )
        return (X - self._obs_mu) / self._obs_sd

    # ------------ I/O ------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise grid to a simple dictionary.

        Returns
        -------
        out : dict
            Dictionary with arrays, names, and metadata.
        """
        return {
            "observables": self.observables.copy(),
            "targets": self.targets.copy(),
            "observable_names": list(self.observable_names),
            "target_names": list(self.target_names),
            "weights": None if self.weights is None else self.weights.copy(),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModelGrid":
        """
        Build a ModelGrid from a dictionary as produced by to_dict.

        Parameters
        ----------
        d : mapping
            Mapping with keys observables, targets, observable_names,
            target_names, weights (optional), and meta (optional).

        Returns
        -------
        grid : ModelGrid
        """
        return cls(
            observables=np.asarray(d["observables"]),
            targets=np.asarray(d["targets"]),
            observable_names=list(d["observable_names"]),
            target_names=list(d["target_names"]),
            weights=None
            if d.get("weights", None) is None
            else np.asarray(d["weights"]),
            meta=dict(d.get("meta", {})),
        )

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
        Build a ModelGrid from a FITS table, with automatic discovery.

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
        table_hdu=1,
        overwrite=False,
        include_meta=True,
        meta_key="MODELGRID_META",
        fill_value=np.nan,
    ):
        """
        Save the grid as a FITS table.

        Observables and targets are written as scalar columns. Column names
        are taken from observable_names and target_names. If weights are
        present they are written as a column named "weight". The table
        metadata will include a JSON blob with grid metadata if include_meta
        is True.

        Parameters
        ----------
        path : str
            Output FITS path.
        table_hdu : int or str, optional
            HDU index or name to write the table to. If the file does not
            exist, it will be created with a primary HDU plus one table HDU.
            Default is 1.
        overwrite : bool, optional
            If True, overwrite an existing file. Default False.
        include_meta : bool, optional
            If True, write JSON-encoded meta and some info into the table
            header. Default True.
        meta_key : str, optional
            Header key for the JSON metadata. Default "MODELGRID_META".
        fill_value : float, optional
            Value used to fill masked values if present. Default NaN.

        Raises
        ------
        ImportError
            If astropy is not available.
        ValueError
            If shapes are inconsistent.
        """
        N, P = self.observables.shape
        Nt, Q = self.targets.shape
        if N != Nt:
            raise ValueError("observables and targets must have same number of rows")

        # Build table
        tab = Table()
        # Observables
        for j, name in enumerate(self.observable_names):
            col = np.asarray(self.observables[:, j])
            if hasattr(col, "mask"):
                col = np.ma.filled(col, fill_value)
            tab.add_column(Column(col, name=name))

        # Targets
        for j, name in enumerate(self.target_names):
            col = np.asarray(self.targets[:, j])
            if hasattr(col, "mask"):
                col = np.ma.filled(col, fill_value)
            tab.add_column(Column(col, name=name))

        # Weights
        if self.weights is not None:
            w = np.asarray(self.weights)
            tab.add_column(Column(w, name="weight"))

        # Minimal meta in table header
        tab.meta = tab.meta or {}
        tab.meta["NMODELS"] = int(N)
        tab.meta["NOBS"] = int(P)
        tab.meta["NTGT"] = int(Q)
        tab.meta["OBSNAME"] = ",".join(self.observable_names)
        tab.meta["TGTNAME"] = ",".join(self.target_names)
        if include_meta:
            try:
                tab.meta[meta_key] = json.dumps(self.meta or {}, ensure_ascii=True)
            except Exception:
                # Fallback to empty meta if not serialisable
                tab.meta[meta_key] = json.dumps({}, ensure_ascii=True)

        # Write
        # If writing to a fresh file or overwrite, astropy can write directly.
        # If writing to a specific HDU index in an existing file, rebuild HDUList.
        if overwrite or (not os.path.exists(path)):
            tab.write(path, overwrite=overwrite)
            if table_hdu not in (1, "1"):
                # Re-open and rename the table HDU to the desired index/name if needed
                with fits.open(path, mode="update") as hdul:
                    if table_hdu != 1:
                        # Replace HDU order by inserting Primary if not present
                        # For simplicity keep table at HDU 1; users can change later if needed
                        pass
            return

        # File exists and overwrite=False -> append or replace table HDU
        with fits.open(path, mode="append") as hdul:
            # Append new table HDU
            hdu = fits.table.TableHDU(data=tab.as_array())
            hdu.name = "MODELGRID" if isinstance(table_hdu, int) else str(table_hdu)
            hdul.append(hdu)
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

        Layout:
          {group}/observables/<name>    1-D dataset per observable
          {group}/targets/<name>        1-D dataset per target
          {group}/weights               optional 1-D dataset
          Attributes on {group}:
            observable_names, target_names, n_models, n_observables,
            n_targets, meta (JSON if include_meta)

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


# ---------------------------------------------------------------------
# ModelGrid fitter
# ---------------------------------------------------------------------
class GridFitter:
    """
    Bayesian fitter over a ModelGrid using pluggable Prior and Likelihood.

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
    Posterior over models is computed with
        log w_i = log p(x | model_i) + log p(model_i) + log weight_i
    and normalised to sum to one over the candidate set.
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
            if self.grid._obs_sd is None:
                raise RuntimeError("standardiser is not fitted")
            sigma_eval = sigma_native / self.grid._obs_sd
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
            j = int(target_col)

        y = self.grid.targets[idx, j]
        hist, _ = np.histogram(y, bins=bins, weights=w, density=False)
        s = hist.sum()
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
        target_factor: float = 2.0,
        expand_factor: float = 2.0,
        n_jobs: int = 1,
        backend: str = "thread",
        stats_for: Optional[Sequence[int | str]] = None,
        stats_bins: Optional[Sequence[np.ndarray]] = None,
        find_multimodal: bool = False,
        return_posts_for_stats: bool = False,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        max_memory_gb: Optional[float] = 16.0,
        memcheck_sample: int = 256,
        safety_margin: float = 1.2,
        tasks_per_worker: Optional[int] = None,
        dry_run: bool = False,
    ) -> dict:
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
        A pre-flight memory check (``check_fit_batch_memory``) estimates the
        additional RAM required by the data structures (candidates, posteriors,
        and optional stats) and raises a ``MemoryError`` if the estimate (with
        safety margin) exceeds the configured limit or the currently available
        memory.

        Parameters
        ----------
        X_native : ndarray, shape (M, P)
            Query observables in native units. Order must match the grid.
        SIG_native : ndarray, shape (M, P)
            Per-dimension uncertainties in native units.
        binner : object, optional
            Must expose `.dims` and `.candidates(y_native, sigmas_native, ...)`.
            If None, all models are considered as candidates for every query.
        target_factor : float, optional
            For sigma-based rectangular binners, factor to match bin widths.
        expand_factor : float, optional
            For sigma-based rectangular binners, neighbourhood expansion factor.
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
        return_posts_for_stats : bool, optional
            If True, return the full (M, K) discrete posterior for each requested
            target (increases memory).
        batch_size : int or None, deprecated
            Kept for compatibility; ignored by the adaptive slicers.
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
        out : dict
            {
            "post_models": List[np.ndarray],  # per-query posterior weights
            "candidates":  List[np.ndarray],  # per-query candidate indices
            "levels":      List[Optional[int]],  # binner level per query (if any)
            # optionally:
            "stats": {target_name: {...}},   # per-target summaries
            "posts_target": {target_name: (M, K) arrays}  # if requested
            }

        Notes
        -----
        When ``use_standardised=True``, likelihoods are evaluated in the grid's
        standardised space, but observable-dependent priors receive *native*
        observables.
        If ``SIG_native`` contains zeros/near-zeros, pre-clip or configure a
        positive floor in your Likelihood to avoid degenerate bandwidths.
        """
        M, P = X_native.shape
        if verbose:
            print("Starting batch fit...")

        # Default behaviour: ~5 tasks per worker, matching prior implementation.
        tpw = 5 if (tasks_per_worker is None) else max(1, int(tasks_per_worker))

        # -------- Memory pre-flight --------
        breakdown = self.check_fit_batch_memory(
            M,
            X_native=X_native,
            SIG_native=SIG_native,
            binner=binner,
            target_factor=target_factor,
            expand_factor=expand_factor,
            memcheck_sample=memcheck_sample,
            stats_for=stats_for,
            stats_bins=stats_bins,
            return_posts_for_stats=return_posts_for_stats,
            max_memory_gb=max_memory_gb,
            safety_margin=safety_margin,
        )
        if dry_run:
            if verbose:
                print("Dry-run mode")
                print(breakdown.get("message", ""))
            return {}

        # -------- Outputs --------
        posts: List[np.ndarray] = [None] * M
        cands: List[np.ndarray] = [None] * M
        levels: List[Optional[int]] = [None] * M

        # =========================
        # Step 1: candidates (threads)
        # =========================
        step1_slices = _guess_slices_step_1(M, P, n_jobs, tasks_per_worker=tpw)
        if verbose:
            print(
                f"[Step 1/3] Selecting candidates for {M} queries "
                f"(n_jobs={n_jobs}, tasks={len(step1_slices)})"
            )

        def _cands_chunk(s: int, e: int):
            out = []
            if binner is None:
                idx_full = np.arange(self.grid.n_models, dtype=np.int64)
                for m in range(s, e):
                    out.append((m, idx_full, None))
                return out
            for m in range(s, e):
                _, idx, lev = _fit_batch_cands_worker(
                    (
                        m,
                        X_native[m],
                        SIG_native[m],
                        binner,
                        target_factor,
                        expand_factor,
                        self.grid.n_models,
                    )
                )
                out.append((m, idx, lev))
            return out

        if n_jobs == 1:
            for s, e in step1_slices:
                if verbose:
                    print(f"  - candidates slice {s}:{e}")
                for m, idx, lev in _cands_chunk(s, e):
                    cands[m] = idx
                    levels[m] = lev
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=max(1, int(n_jobs))) as ex:
                futs = [ex.submit(_cands_chunk, s, e) for (s, e) in step1_slices]
                for fut in as_completed(futs):
                    for m, idx, lev in fut.result():
                        cands[m] = idx
                        levels[m] = lev

        # =========================
        # Step 2: posterior over models (threads or processes)
        # =========================
        costs = np.array(
            [(0 if cands[m] is None else max(1, len(cands[m]))) * P for m in range(M)]
        )
        slices_step2 = _make_cost_balanced_slices(costs, n_jobs, tasks_per_worker=tpw)
        if verbose:
            print(
                f"[Step 2/3] Evaluating posteriors "
                f"(backend={backend}, n_jobs={n_jobs}, tasks={len(slices_step2)})"
            )

        # If processes requested, ensure picklable prior/likelihood
        if backend == "process":
            try:
                import pickle

                pickle.dumps((self.likelihood, self.prior))
            except Exception as e:
                if verbose:
                    print(
                        "[fit_batch] prior/likelihood are not picklable; "
                        "falling back to thread backend. Reason:",
                        repr(e),
                    )
                backend = "thread"

        grid_dict = {
            "observables": self.grid.observables,
            "targets": self.grid.targets,
            "weights": self.grid.weights,
            "_obs_mu": getattr(self.grid, "_obs_mu", None),
            "_obs_sd": getattr(self.grid, "_obs_sd", None),
        }
        is_obs_dep_prior = isinstance(self.prior, ObservableDependentPrior)

        def _post_chunk_thread(s: int, e: int):
            out = []
            for m in range(s, e):
                w = self.posterior_over_models(
                    x_native=X_native[m],
                    sigma_native=SIG_native[m],
                    candidate_idx=cands[m],
                )
                out.append((m, w))
            return out

        def _post_chunk_proc(s: int, e: int):
            out = []
            for m in range(s, e):
                _, w = _fit_batch_post_worker(
                    (
                        m,
                        X_native[m],
                        SIG_native[m],
                        cands[m],
                        grid_dict,
                        self.use_standardised,
                        self.likelihood,
                        self.prior,
                        is_obs_dep_prior,
                    )
                )
                out.append((m, w))
            return out

        if n_jobs == 1:
            for s, e in slices_step2:
                if verbose:
                    print(f"  - posteriors slice {s}:{e}")
                for m, w in _post_chunk_thread(s, e):
                    posts[m] = w
        else:
            Executor = (
                ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
            )
            with Executor(max_workers=max(1, int(n_jobs))) as ex:
                if backend == "thread":
                    futs = [
                        ex.submit(_post_chunk_thread, s, e) for (s, e) in slices_step2
                    ]
                else:
                    futs = [
                        ex.submit(_post_chunk_proc, s, e) for (s, e) in slices_step2
                    ]
                for fut in as_completed(futs):
                    for m, w in fut.result():
                        posts[m] = w

        # =========================
        # Step 3: per-target stats (threads)
        # =========================
        slices = slices_step2  # reuse balanced slices
        stats = {}
        posts_target = {}

        if stats_for is not None:
            if stats_bins is None or len(stats_for) != len(stats_bins):
                raise ValueError(
                    "stats_bins must be provided and match stats_for length"
                )

            if verbose:
                tnames = [
                    t if isinstance(t, str) else self.grid.target_names[int(t)]
                    for t in stats_for
                ]
                print(
                    f"[Step 3/3] Computing stats for targets: {tnames} "
                    f"(n_jobs={n_jobs}, tasks={len(slices)})"
                )

            for tname, bins in zip(stats_for, stats_bins):
                if isinstance(tname, str):
                    j = self.grid.target_names.index(tname)
                    key = tname
                else:
                    j = int(tname)
                    key = self.grid.target_names[j]

                K = len(bins) - 1
                centers = 0.5 * (bins[:-1] + bins[1:])
                mean = np.empty(M)
                std = np.empty(M)
                vmap = np.empty(M)
                lo68 = np.empty(M)
                hi68 = np.empty(M)
                q16 = np.empty(M)
                q50 = np.empty(M)
                q84 = np.empty(M)
                nmodes = np.empty(M, int)
                posts_k = np.zeros((M, K), dtype=float)

                def _stats_chunk_thread(s: int, e: int):
                    out = []
                    for m in range(s, e):
                        y = self.grid.targets[cands[m], j]
                        hist, _ = np.histogram(
                            y, bins=bins, weights=posts[m], density=False
                        )
                        post = (
                            hist / hist.sum()
                            if hist.sum() > 0
                            else np.full_like(hist, 1.0 / hist.size)
                        )
                        st = hist_stats(centers, post, find_multimodal=find_multimodal)
                        out.append((m, post, st))
                    return out

                if n_jobs == 1:
                    for s, e in slices:
                        if verbose:
                            print(f"  - stats[{key}] slice {s}:{e}")
                        for m, post, st in _stats_chunk_thread(s, e):
                            posts_k[m] = post
                            mean[m], std[m], vmap[m] = st["mean"], st["std"], st["map"]
                            lo68[m], hi68[m] = st["lo68"], st["hi68"]
                            q16[m], q50[m], q84[m] = st["q"]
                            nmodes[m] = (
                                len(st.get("modes", [st["map"]]))
                                if find_multimodal
                                else 1
                            )
                else:
                    with ThreadPoolExecutor(max_workers=max(1, int(n_jobs))) as ex:
                        futs = [
                            ex.submit(_stats_chunk_thread, s, e) for (s, e) in slices
                        ]
                        for fut in as_completed(futs):
                            for m, post, st in fut.result():
                                posts_k[m] = post
                                mean[m], std[m], vmap[m] = (
                                    st["mean"],
                                    st["std"],
                                    st["map"],
                                )
                                lo68[m], hi68[m] = st["lo68"], st["hi68"]
                                q16[m], q50[m], q84[m] = st["q"]
                                nmodes[m] = (
                                    len(st.get("modes", [st["map"]]))
                                    if find_multimodal
                                    else 1
                                )

                stats[key] = {
                    "centers": centers,
                    "mean": mean,
                    "std": std,
                    "map": vmap,
                    "q16": q16,
                    "q50": q50,
                    "q84": q84,
                    "lo68": lo68,
                    "hi68": hi68,
                    "nmodes": nmodes,
                }
                if return_posts_for_stats:
                    posts_target[key] = posts_k

        out = {"post_models": posts, "candidates": cands, "levels": levels}
        if stats:
            out["stats"] = stats
        if return_posts_for_stats and posts_target:
            out["posts_target"] = posts_target
        if verbose:
            print("fit_batch complete.")
        return out

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
                edges = np.linspace(
                    mid - kappa_sigma_edges * (mid - lo),
                    mid + kappa_sigma_edges * (hi - mid),
                    bi + 1,
                )
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
                ax.axvline(true_target_vals[i], color="r")
            for qv in q[i]:
                ax.axvline(qv, ls="--", lw=1.0, color=color)

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
                frac = compute_fraction_from_map(H, xedges=xedges, yedges=yedges)
                ax.contourf(
                    xb, yb, frac.T, cmap="Spectral", levels=[0.01, 0.05, 0.32, 0.5, 1]
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
        axes[0, 0].set_title(f"No. of\ncandidate models: {w.size}", fontsize="small")
        if suptitle:
            fig.suptitle(suptitle)
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

    @staticmethod
    def compute_pit(
        z_true: np.ndarray, posts: np.ndarray, edges: np.ndarray
    ) -> np.ndarray:
        """
        PIT values from discrete posteriors.

        Parameters
        ----------
        z_true : ndarray, shape (N,)
        posts : ndarray, shape (N, K)
        edges : ndarray, shape (K+1,)

        Returns
        -------
        pit : ndarray, shape (N,)
        """
        return pit_from_discrete_posterior(z_true, posts, edges)

    @staticmethod
    def photoz_metrics(z_true: np.ndarray, z_est: np.ndarray) -> dict:
        """
        Standard photo-z metrics on z_true vs point estimates.
        """
        return photoz_metrics(z_true, z_est)

    @staticmethod
    def _human_bytes(nbytes: int) -> str:
        """Return a human-friendly string for a byte count."""
        units = ["B", "KB", "MB", "GB", "TB", "PB"]
        x = float(nbytes)
        i = 0
        while x >= 1024.0 and i < len(units) - 1:
            x /= 1024.0
            i += 1
        return f"{x:.2f} {units[i]}"

    def estimate_fit_batch_memory(
        self,
        M: int,
        *,
        # Optional pilot sampling to estimate candidate set sizes
        X_native: Optional[np.ndarray] = None,
        SIG_native: Optional[np.ndarray] = None,
        binner: Optional[Any] = None,
        target_factor: float = 2.0,
        expand_factor: float = 2.0,
        memcheck_sample: int = 256,
        rng: Optional[np.random.Generator] = None,
        # Stats footprint
        stats_for: Optional[Sequence[int | str]] = None,
        stats_bins: Optional[Sequence[np.ndarray]] = None,
        return_posts_for_stats: bool = False,
        # Dtype sizes
        float_bytes: int = np.dtype(np.float64).itemsize,
        int_bytes: int = np.dtype(np.int64).itemsize,
    ) -> Dict[str, int]:
        """
        Estimate peak additional RAM needed by `fit_batch` data structures.

        The estimate includes:
          - candidates per query (int arrays)
          - posterior weights per query (float arrays)
          - per-target statistics arrays (mean, std, map, lo68, hi68, q16, q50, q84, nmodes)
          - optional posts_target (M x K) per requested target

        It does NOT include the already-loaded ModelGrid arrays themselves,
        Python interpreter overhead, BLAS scratch memory, or OS allocator
        fragmentation. A 10-20% headroom is recommended.

        Parameters
        ----------
        M : int
            Number of queries to evaluate.
        X_native, SIG_native : ndarray, optional
            If provided with a `binner`, a pilot sample is used to estimate the
            average number of candidate models per query.
        binner : object, optional
            Must expose `dims` and a `candidates(y_native, sigmas_native, target_factor, expand_factor)`
            method. If omitted or pilot cannot run, falls back to worst-case (all models).
        target_factor, expand_factor : float, optional
            Passed to the `binner.candidates` call in the pilot.
        memcheck_sample : int, optional
            Max number of queries to sample for candidate-size estimation.
        rng : numpy.random.Generator, optional
            RNG for sampling. Default uses np.random.default_rng().
        stats_for, stats_bins : sequence, optional
            As in `fit_batch`. If provided, stats arrays are counted. If
            `return_posts_for_stats` is True, M x K floats per target are counted.
        return_posts_for_stats : bool, optional
            Whether to include storage for full posterior-over-bins per target.
        float_bytes, int_bytes : int, optional
            Byte size of float and int elements (default float64/int64).

        Returns
        -------
        breakdown : dict
            Keys: 'candidates_bytes', 'post_models_bytes', 'levels_bytes',
                  'stats_bytes', 'posts_target_bytes', 'total_bytes'.
        """
        N = self.grid.n_models

        # --- pilot estimate of candidate sizes per query ---
        cand_mean = None
        if (binner is not None) and (X_native is not None) and (SIG_native is not None):
            m = min(M, len(X_native))
            ns = min(memcheck_sample, m)
            if ns > 0:
                rng = rng or np.random.default_rng()
                sample_idx = rng.choice(m, size=ns, replace=False)
                sizes = []
                for i in sample_idx:
                    y_sub = X_native[i, binner.dims]
                    s_sub = SIG_native[i, binner.dims]
                    try:
                        idx, _lev = binner.candidates(
                            y_native=y_sub,
                            sigmas_native=s_sub,
                            target_factor=target_factor,
                            expand_factor=expand_factor,
                        )
                        nci = int(np.asarray(idx).size)
                        sizes.append(
                            nci if nci > 0 else N
                        )  # fall back to full grid if empty
                    except Exception:
                        sizes = []
                        break
                if sizes:
                    cand_mean = float(np.mean(sizes))

        # Worst case if no pilot or pilot failed
        if cand_mean is None or not np.isfinite(cand_mean) or cand_mean <= 0:
            cand_mean = float(N)

        # --- memory for candidates, posts, and levels ---
        # For each query we store one int array 'cands[m]' and one float array 'posts[m]'
        candidates_bytes = int(M * cand_mean * int_bytes)
        post_models_bytes = int(M * cand_mean * float_bytes)
        # Optional levels (one int or None per query); store as int64 estimate
        levels_bytes = int(M * int_bytes)

        # --- stats memory footprint ---
        stats_bytes = 0
        posts_target_bytes = 0
        if stats_for is not None:
            T = len(stats_for)
            # Per target we keep 8 float arrays of length M (mean,std,map,lo68,hi68,q16,q50,q84)
            # and 1 int array of length M (nmodes).
            per_target_stats = (8 * M * float_bytes) + (M * int_bytes)
            stats_bytes += T * per_target_stats

            if return_posts_for_stats:
                if stats_bins is None or len(stats_bins) != T:
                    # be conservative: assume K=64 per target
                    Ks = [64] * T
                else:
                    Ks = [max(0, (np.asarray(b).size - 1)) for b in stats_bins]
                posts_target_bytes = int(sum(M * K * float_bytes for K in Ks))

        total_bytes = (
            candidates_bytes
            + post_models_bytes
            + levels_bytes
            + stats_bytes
            + posts_target_bytes
        )

        return {
            "candidates_bytes": candidates_bytes,
            "post_models_bytes": post_models_bytes,
            "levels_bytes": levels_bytes,
            "stats_bytes": stats_bytes,
            "posts_target_bytes": posts_target_bytes,
            "total_bytes": total_bytes,
        }

    def check_fit_batch_memory(
        self,
        M: int,
        *,
        X_native: Optional[np.ndarray] = None,
        SIG_native: Optional[np.ndarray] = None,
        binner: Optional[Any] = None,
        target_factor: float = 2.0,
        expand_factor: float = 2.0,
        memcheck_sample: int = 256,
        stats_for: Optional[Sequence[int | str]] = None,
        stats_bins: Optional[Sequence[np.ndarray]] = None,
        return_posts_for_stats: bool = False,
        max_memory_gb: float = 16.0,
        safety_margin: float = 1.2,
    ) -> Dict[str, int]:
        """
        Estimate and validate memory needs for `fit_batch`.

        Raises a MemoryError if the estimated total exceeds `max_memory_gb`
        after applying a safety margin.

        Returns the same breakdown as `estimate_fit_batch_memory` on success.

        Parameters
        ----------
        M : int
            Number of queries to evaluate.
        X_native, SIG_native, binner, target_factor, expand_factor, memcheck_sample
            Passed to `estimate_fit_batch_memory` to refine candidate sizes.
        stats_for, stats_bins, return_posts_for_stats
            Passed through to count stats arrays and optional posts_target storage.
        max_memory_gb : float, optional
            Limit in GB. Use None to disable checking.
        safety_margin : float, optional
            Multiplier applied to the estimate to account for allocator/BLAS/OS overhead.
        """
        if max_memory_gb is None:
            # Checking disabled
            return {
                "candidates_bytes": 0,
                "post_models_bytes": 0,
                "levels_bytes": 0,
                "stats_bytes": 0,
                "posts_target_bytes": 0,
                "total_bytes": 0,
            }

        breakdown = self.estimate_fit_batch_memory(
            M,
            X_native=X_native,
            SIG_native=SIG_native,
            binner=binner,
            target_factor=target_factor,
            expand_factor=expand_factor,
            memcheck_sample=memcheck_sample,
            stats_for=stats_for,
            stats_bins=stats_bins,
            return_posts_for_stats=return_posts_for_stats,
        )

        est = int(breakdown["total_bytes"] * float(safety_margin))
        avail_ram = available_memory_bytes()
        limit = min(avail_ram, int(max_memory_gb * (1024**3)))

        avail_ram = self._human_bytes(avail_ram)
        hb_est = self._human_bytes(est)
        hb_lim = self._human_bytes(limit)
        hb_c = self._human_bytes(breakdown["candidates_bytes"])
        hb_p = self._human_bytes(breakdown["post_models_bytes"])
        hb_s = self._human_bytes(breakdown["stats_bytes"])
        hb_pt = self._human_bytes(breakdown["posts_target_bytes"])

        if est > limit:
            raise MemoryError(
                "fit_batch memory pre-check failed: "
                f"estimated peak (with safety margin) {hb_est} exceeds limit {hb_lim}.\n"
                f"Breakdown (pre-margin): candidates={hb_c}, post_models={hb_p}, "
                f"stats={hb_s}, posts_target={hb_pt}.\n"
                "Suggestions:\n"
                "  • Use a tighter binner or reduce target/expand factors to shrink candidate sets.\n"
                "  • Disable `return_posts_for_stats` or reduce the number of bins per target.\n"
                "  • Compute fewer targets in `stats_for`, or run in smaller M with external batching.\n"
                "  • Persist or stream results instead of keeping all per-query posteriors in memory."
            )
        else:
            breakdown["message"] = (
                "fit_batch memory check"
                f"- estimated peak (with safety margin) {hb_est} (current limit {hb_lim}, avail RAM {avail_ram}).\n"
                f"- breakdown (pre-margin): candidates={hb_c}, post_models={hb_p}, "
                f"- stats={hb_s}, posts_target={hb_pt}.\n"
            )
        return breakdown
