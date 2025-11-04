#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 16:16:53 2025

@author: pcorchoc
"""

# besta/grid/core.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterable, Mapping, Any
from abc import ABC, abstractmethod
from itertools import product

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.table import Table, Column
from astropy.io import fits
import h5py

from prob import (
    Prior, FlatPrior, ObservableDependentPrior,
    Likelihood, GaussianProductLikelihood,
    posterior_over_models as posterior_over_models_fn,
)

def _weighted_quantiles(x: np.ndarray, w: np.ndarray,
                        qs: Sequence[float]) -> np.ndarray:
    x = np.asarray(x); w = np.asarray(w)
    m = np.isfinite(x) & np.isfinite(w) & (w >= 0)
    if not m.any():
        return np.array([np.nan] * len(qs))
    x = x[m]; w = w[m]
    order = np.argsort(x)
    x = x[order]; w = w[order]
    cdf = np.cumsum(w); cdf = cdf / cdf[-1]
    return np.interp(qs, cdf, x)



class ModelGridGenerator(ABC):
    """
    Abstract interface for producing a ModelGrid from inputs.

    Implementations may synthesise spectra, SEDs, photometry, or any
    observable set, and attach associated continuous targets such as
    SFR, sSFR, mass, metallicity, age, or redshift.

    Subclasses should implement the generate method.
    """

    @abstractmethod
    def generate(self, **kwargs) -> "ModelGrid":
        """
        Build and return a ModelGrid.

        Parameters
        ----------
        **kwargs
            Implementation-specific keyword arguments such as parameter
            ranges, library choices, and resolution settings.

        Returns
        -------
        grid : ModelGrid
            A populated model grid instance.
        """
        raise NotImplementedError


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
    def select(self, idx: np.ndarray) -> "ModelGrid":
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
        return ModelGrid(
            observables=self.observables[idx].copy(),
            targets=self.targets[idx].copy(),
            observable_names=list(self.observable_names),
            target_names=list(self.target_names),
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
            raise RuntimeError("fit_standardiser must be called before transform_observables")
        return (X - self._obs_mu) / self._obs_sd

    # ------------ export helpers ------------
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
            weights=None if d.get("weights", None) is None else np.asarray(d["weights"]),
            meta=dict(d.get("meta", {})),
        )

    @classmethod
    def from_fits_table(cls,
                        path,
                        observable_cols=None,
                        target_cols=None,
                        weight_col=None,
                        row_mask=None,
                        table_hdu=1,
                        memmap=False,
                        meta_key="MODELGRID_META"):
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
        import json
        import numpy as np
    
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
                    raise ValueError(f"FITS column '{n}' must be 1-D; got shape {a.shape}")
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
    
        if obs_names_meta is not None and tgt_names_meta is not None and observable_cols is None and target_cols is None:
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
                raise ValueError("observable_cols must be provided if auto-discovery is not available")
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
    def from_hdf5(cls,
                  path,
                  group="/modelgrid",
                  observable_dsets=None,
                  target_dsets=None,
                  weight_dset=None,
                  row_slice=None):
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
                raise ValueError(f"HDF5 dataset '{key}' must be 1-D; got shape {a.shape}")
            return a
    
        with h5py.File(path, "r") as f:
            if group not in f:
                raise KeyError(f"group '{group}' not found in file")
            g = f[group]
    
            # --- 1) Auto-discovery (preferred layout written by to_hdf5) ---
            auto_loaded = False
            if "observables" in g and "targets" in g and observable_dsets is None and target_dsets is None:
                gob = g["observables"]; tgt = g["targets"]
    
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
                cols = [_ensure_1d(gob[name][...], f"{group}/observables/{name}") for name in obs_names]
                X = np.vstack(cols).T if cols else np.empty((g.attrs.get("n_models", 0), 0))
    
                tcols = [_ensure_1d(tgt[name][...], f"{group}/targets/{name}") for name in tgt_names]
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
                        src = [ _full(v) for v in dsets.values() ]
                    else:
                        dst = list(dsets)
                        src = [ _full(v) for v in dsets ]
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
                            if a.ndim == 1 and np.issubdtype(a.dtype, np.number) and k != "weights":
                                names.append(k)
                                cols.append(a)
                    if not cols:
                        raise ValueError("No 1-D numeric datasets found to load under the group")
                    X = np.vstack(cols).T
                    obs_names = names
                    Y = np.empty((X.shape[0], 0))
                    tgt_names = []
                elif X is None or obs_names is None:
                    raise ValueError("observable_dsets must be provided if auto-discovery is not available")
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


    def to_fits_table(self,
                      path,
                      table_hdu=1,
                      overwrite=False,
                      include_meta=True,
                      meta_key="MODELGRID_META",
                      fill_value=np.nan):
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
            raise ValueError(
                "observables and targets must have same number of rows")
    
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
                tab.meta[meta_key] = json.dumps(self.meta or {},
                                                ensure_ascii=True)
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

    def to_hdf5(self,
                path,
                group="/modelgrid",
                overwrite=False,
                compression="gzip",
                compression_opts=4,
                chunks=True,
                include_meta=True):
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
                    raise ValueError(f"group '{group}' already exists; use overwrite=True")
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
            g.attrs["observable_names"] = json.dumps(list(self.observable_names), ensure_ascii=True)
            g.attrs["target_names"] = json.dumps(list(self.target_names), ensure_ascii=True)
            if include_meta:
                try:
                    g.attrs["meta"] = json.dumps(self.meta or {}, ensure_ascii=True)
                except Exception:
                    g.attrs["meta"] = json.dumps({}, ensure_ascii=True)

# ---------------------------------------------------------------------
# 3) HierarchicalGridBinner (generalised)
# ---------------------------------------------------------------------
class HierarchicalGridBinner:
    """
    Multi-resolution rectilinear binning on a chosen observable subspace.

    Works on any subset of columns from grid.observables.
    It stores per-level maps from integer cell keys to model indices.

    Parameters
    ----------
    dims : list of int
        Indices of observable columns to bin on (D dimensions).
    levels : int, optional
        Number of grid levels (default 5).
    base_bins : int, optional
        Number of bins per dimension at level 0 (default 6).

    Attributes
    ----------
    mu : ndarray, shape (D,)
        Training mean per binned dimension.
    sd : ndarray, shape (D,)
        Training standard deviation per binned dimension.
    bin_edges : list of list of ndarray
        Per level, per dimension bin edges in standardised units.
    layers : list of dict
        Per level maps: key tuple of ints to array of model indices.
    """

    def __init__(self, dims: List[int], levels: int = 5, base_bins: int = 6) -> None:
        self.dims = list(dims)
        self.levels = levels
        self.base_bins = base_bins
        self.mu: Optional[np.ndarray] = None
        self.sd: Optional[np.ndarray] = None
        self.bin_edges: List[List[np.ndarray]] = []
        self.layers: List[Dict[Tuple[int, ...], np.ndarray]] = []

    @staticmethod
    def _digitize_nd(x_std: np.ndarray, edges_per_dim: List[np.ndarray]) -> np.ndarray:
        """Return integer indices for points on an N-D grid."""
        idx = []
        for d, edges in enumerate(edges_per_dim):
            j = np.digitize(x_std[..., d], edges) - 1
            j = np.clip(j, 0, len(edges) - 2)
            idx.append(j)
        return np.stack(idx, axis=-1)

    def fit(self, grid: ModelGrid) -> None:
        """
        Build multi-resolution bins from a ModelGrid.

        Parameters
        ----------
        grid : ModelGrid
            The training model grid.
        """
        X = grid.observables[:, self.dims]
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=0)
        sd = np.where(sd == 0.0, 1.0, sd)
        self.mu = mu
        self.sd = sd
        Xs = (X - mu) / sd

        dmin = np.nanmin(Xs, axis=0)
        dmax = np.nanmax(Xs, axis=0)
        pad = 0.01 * (dmax - dmin + 1e-6)
        dmin -= pad
        dmax += pad

        self.bin_edges = []
        self.layers = []
        for lev in range(self.levels):
            n_bins = self.base_bins * (2 ** lev)
            edges = [np.linspace(dmin[j], dmax[j], n_bins + 1) for j in range(len(self.dims))]
            self.bin_edges.append(edges)

            cell_idx = self._digitize_nd(Xs, edges)
            keys = [tuple(cell_idx[i]) for i in range(Xs.shape[0])]
            layer: Dict[Tuple[int, ...], List[int]] = {}
            for i, key in enumerate(keys):
                (layer.setdefault(key, [])).append(i)
            # convert lists to arrays
            layer = {k: np.asarray(v, dtype=np.int64) for k, v in layer.items()}
            self.layers.append(layer)

    def choose_level(self, sigmas_native: np.ndarray, target_factor: float = 2.0) -> int:
        """
        Choose a level whose average bin width matches target_factor * sigma.

        Parameters
        ----------
        sigmas_native : ndarray, shape (D,)
            Per-dimension uncertainties in native units for the query.
        target_factor : float, optional
            Target multiple of sigma to approximate with the bin width.

        Returns
        -------
        level : int
            Level index between 0 and levels - 1.
        """
        if self.mu is None or self.sd is None:
            raise RuntimeError("fit must be called before choose_level")
        sig_std = sigmas_native / self.sd
        best_lev, best_score = 0, np.inf
        for lev in range(self.levels):
            edges = self.bin_edges[lev]
            widths = np.array([np.diff(e).mean() for e in edges])
            target = target_factor * sig_std
            score = np.mean(np.abs(np.log((widths + 1e-12) / (target + 1e-12))))
            if score < best_score:
                best_score = score
                best_lev = lev
        return best_lev

    def neighbours(self, y_native: np.ndarray, lev: int, sigmas_native: np.ndarray,
                   expand_factor: float = 2.0) -> List[Tuple[int, ...]]:
        """
        Return cell keys around a query point covering about expand_factor * sigma.

        Parameters
        ----------
        y_native : ndarray, shape (D,)
            Query coordinates in native units.
        lev : int
            Grid level to use.
        sigmas_native : ndarray, shape (D,)
            Per-dimension uncertainties in native units.
        expand_factor : float, optional
            Multiple of sigma to cover by cell expansion.

        Returns
        -------
        keys : list of tuple of int
            Cell keys to search at the given level.
        """
        if self.mu is None or self.sd is None:
            raise RuntimeError("fit must be called before neighbours")
        y_std = (y_native - self.mu) / self.sd
        sig_std = sigmas_native / self.sd
        edges = self.bin_edges[lev]
        cell = self._digitize_nd(y_std[None, :], edges)[0]
        widths = np.array([np.diff(e).mean() for e in edges])
        half_span = np.ceil((expand_factor * sig_std) / widths).astype(int)
        ranges = [
            range(max(0, cell[d] - half_span[d]),
                  min(len(edges[d]) - 1, cell[d] + half_span[d]) + 1)
            for d in range(len(self.dims))
        ]
        return list(product(*ranges))

    def candidates(self, y_native: np.ndarray, sigmas_native: np.ndarray,
                   target_factor: float = 2.0, expand_factor: float = 2.0) -> Tuple[np.ndarray, int]:
        """
        Collect candidate model indices given a query point and errors.

        Parameters
        ----------
        y_native : ndarray, shape (D,)
            Query in native units for the binned dimensions.
        sigmas_native : ndarray, shape (D,)
            Per-dimension uncertainties in native units.
        target_factor : float, optional
            Level selection target multiple of sigma.
        expand_factor : float, optional
            Neighbourhood expansion multiple of sigma.

        Returns
        -------
        idx : ndarray, shape (K,), dtype=int
            Sorted unique indices of candidate models.
        level : int
            The level used for the selection.
        """
        lev = self.choose_level(sigmas_native=sigmas_native, target_factor=target_factor)
        keys = self.neighbours(y_native=y_native, lev=lev, sigmas_native=sigmas_native,
                               expand_factor=expand_factor)
        layer = self.layers[lev]
        idxs = [layer[k] for k in keys if k in layer]
        if not idxs:
            return np.array([], dtype=np.int64), lev
        return np.unique(np.concatenate(idxs)), lev


# ---------------------------------------------------------------------
# 4) GridFitter
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

    def __init__(self,
                 grid,
                 likelihood: Optional[Likelihood] = None,
                 prior: Optional[Prior] = None,
                 use_standardised: bool = True) -> None:
        self.grid = grid
        self.likelihood = likelihood if likelihood is not None else GaussianProductLikelihood()
        self.prior = prior if prior is not None else FlatPrior()
        self.use_standardised = use_standardised
        if self.use_standardised and hasattr(self.grid, "fit_standardiser"):
            self.grid.fit_standardiser()

    def _to_eval_space(self,
                       x_native: np.ndarray,
                       X_native: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map to evaluation space if standardisation is enabled.

        Returns
        -------
        x_eval : ndarray, shape (P,)
        X_eval : ndarray, shape (Nc, P)
        """
        if self.use_standardised and hasattr(self.grid, "transform_observables"):
            return (self.grid.transform_observables(x_native),
                    self.grid.transform_observables(X_native))
        return x_native, X_native

    def posterior_over_models(self,
                          x_native: np.ndarray,
                          sigma_native: np.ndarray,
                          candidate_idx: Optional[np.ndarray] = None) -> np.ndarray:
        idx = np.arange(self.grid.n_models) if candidate_idx is None else candidate_idx
        Xc_native = self.grid.observables[idx]
        Tc = self.grid.targets[idx]
        wc = self.grid.weights[idx] if getattr(self.grid, "weights", None) is not None else None
    
        # Map to evaluation space if requested
        if self.use_standardised and hasattr(self.grid, "transform_observables"):
            x_eval = self.grid.transform_observables(x_native)
            X_eval = self.grid.transform_observables(Xc_native)
            if self.grid._obs_sd is None:
                raise RuntimeError("standardiser is not fitted")
            sigma_eval = sigma_native / self.grid._obs_sd
            # Priors that depend on observables must see NATIVE observables
            prior_obs = Xc_native
        else:
            x_eval = x_native
            X_eval = Xc_native
            sigma_eval = sigma_native
            prior_obs = Xc_native
    
        # Detect priors that need observables
        prior_needs_obs = isinstance(self.prior, ObservableDependentPrior)
    
        w = posterior_over_models_fn(
            x_native=x_eval,
            sigma_native=sigma_eval,          # now in the SAME space as x_eval/X_eval
            X_models=X_eval,
            targets_models=Tc,
            likelihood=self.likelihood,
            prior=self.prior,
            model_weights=wc,
            prior_needs_observables=prior_needs_obs,
            observables_models=prior_obs,     # NATIVE for priors like p(z|m)
        )
        return w


    def posterior_over_target(self,
                              x_native: np.ndarray,
                              sigma_native: np.ndarray,
                              target_col: int | str,
                              bins: np.ndarray,
                              candidate_idx: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
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
        w = self.posterior_over_models(x_native=x_native, sigma_native=sigma_native, candidate_idx=idx)

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
        post = hist / s if s > 0 and np.isfinite(s) else np.full_like(hist, 1.0 / hist.size)
        centers = 0.5 * (bins[:-1] + bins[1:])
        return post, centers

    def fit_batch(self,
                  X_native: np.ndarray,
                  SIG_native: np.ndarray,
                  binner: Any | None = None,
                  target_factor: float = 2.0,
                  expand_factor: float = 2.0) -> Dict[str, Any]:
        """
        Evaluate model posteriors for multiple queries.

        Parameters
        ----------
        X_native : ndarray, shape (M, P)
            Query observables in native units.
        SIG_native : ndarray, shape (M, P)
            Per-object uncertainties in native units.
        binner : object, optional
            Candidate selector with method:
              candidates(y_native, sigmas_native, target_factor, expand_factor)
            returning (indices, level_used).
        target_factor : float, optional
            Passed to binner for level choice.
        expand_factor : float, optional
            Passed to binner for neighbourhood expansion.

        Returns
        -------
        out : dict
            Keys:
              post_models : list of arrays, posterior weights per query
              candidates  : list of index arrays per query
              levels      : list of ints or None
        """
        M, P = X_native.shape
        posts, cands, levels = [], [], []

        for m in range(M):
            idx = None
            lev = None
            if binner is not None:
                y_sub = X_native[m, binner.dims]
                s_sub = SIG_native[m, binner.dims]
                idx, lev = binner.candidates(
                    y_native=y_sub,
                    sigmas_native=s_sub,
                    target_factor=target_factor,
                    expand_factor=expand_factor,
                )
                if idx.size == 0:
                    idx = np.arange(self.grid.n_models)

            w = self.posterior_over_models(
                x_native=X_native[m],
                sigma_native=SIG_native[m],
                candidate_idx=idx,
            )
            posts.append(w)
            cands.append(idx if idx is not None else np.arange(self.grid.n_models))
            levels.append(lev)

        return {"post_models": posts, "candidates": cands, "levels": levels}

    def corner_for_targets(self,
                           x_native: np.ndarray,
                           sigma_native: np.ndarray,
                           target_cols: Sequence[int | str],
                           true_target_vals : Optional[np.ndarray] = None,
                           candidate_idx: Optional[np.ndarray] = None,
                           bins: int | Sequence[int | np.ndarray] = 50,
                           figsize: Optional[Tuple[float, float]] = None,
                           suptitle: Optional[str] = None,
                           color: Optional[str] = None,
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
        w = self.posterior_over_models(x_native=x_native,
                                       sigma_native=sigma_native,
                                       candidate_idx=idx)

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
        fig, axes = plt.subplots(D, D, figsize=figsize, squeeze=False)

        # Weighted summaries
        q = np.zeros((D, 3))
        mu = np.zeros(D)
        for d in range(D):
            q[d] = _weighted_quantiles(Y[:, d], w, quantiles)
            mu[d] = np.sum(w * Y[:, d])

        # Diagonals
        for i in range(D):
            ax = axes[i, i]
            bi = bins_list[i]
            if isinstance(bi, int):
                lo, hi = _weighted_quantiles(Y[:, i], w, (0.01, 0.99))
                lo = lo if np.isfinite(lo) else np.nanmin(Y[:, i])
                hi = hi if np.isfinite(hi) else np.nanmax(Y[:, i])
                edges = np.linspace(lo, hi, bi + 1)
            else:
                edges = np.asarray(bi)
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
            ax.set_yticks([])
            ax.set_xlabel(names[i])

        # Lower triangle
        for i in range(1, D):
            for j in range(i):
                ax = axes[i, j]
                bi = bins_list[j]; bj = bins_list[i]
                if isinstance(bi, int):
                    lo, hi = _weighted_quantiles(Y[:, j], w, (0.01, 0.99))
                    xedges = np.linspace(lo, hi, bi + 1)
                else:
                    xedges = np.asarray(bi)
                if isinstance(bj, int):
                    lo, hi = _weighted_quantiles(Y[:, i], w, (0.01, 0.99))
                    yedges = np.linspace(lo, hi, bj + 1)
                else:
                    yedges = np.asarray(bj)
                H, xe, ye = np.histogram2d(
                    Y[:, j], Y[:, i], bins=[xedges, yedges], weights=w)
                ax.pcolormesh(xe, ye, H.T, shading="auto", alpha=alpha_mesh,
                              norm=LogNorm())
                if i == D - 1:
                    ax.set_xlabel(names[j])
                else:
                    ax.set_xticklabels([])
                if j == 0:
                    ax.set_ylabel(names[i])
                else:
                    ax.set_yticklabels([])

        # Hide upper triangle
        for i in range(D):
            for j in range(i + 1, D):
                axes[i, j].axis("off")
        axes[0, 0].set_title(f"No. of candidate models: {w.size}")
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
