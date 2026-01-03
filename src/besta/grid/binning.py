from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterable, List, Tuple, Optional, Dict, Sequence, Any
from itertools import product
import json
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from besta.grid.grid import ModelGrid

# ----------------------------
# Utilities
# ----------------------------

def _fit_transform(dims_array: np.ndarray, mode: str = "standardize",
                   pca_variance: float = 1.0) -> Dict[str, Any]:
    """
    Fit a linear transform on dims_array and return a dict with parameters.

    Parameters
    ----------
    dims_array : ndarray, shape (N, D)
        Training data for selected dimensions.
    mode : {"none", "standardize", "pca_whiten"}
        Transform type.
    pca_variance : float in (0,1]
        Fraction of variance to keep if mode == "pca_whiten".

    Returns
    -------
    t : dict
        Transformation. Contains keys: mode, mu (mean), sd (standard dev.),
        W, b, kept_dims.
    """
    mode = (mode or "standardize").lower()
    if mode == "none":
        return {"mode": "none", "mu": None, "sd": None, "W": None,
                "b": None, "kept_dims": slice(None)}

    # Training dataset
    X = np.asarray(dims_array, float)
    mu = np.nanmean(X, axis=0)
    Xc = X - mu

    if mode == "standardize":
        sd = np.nanstd(Xc, axis=0)
        sd = np.where(sd == 0.0, 1.0, sd)
        # y = (x - mu) / sd
        return {"mode": "standardize", "mu": mu, "sd": sd, "W": None, "b": None,
                "kept_dims": slice(None)}

    if mode == "pca_whiten":
        # PCA on covariance of centered data
        U, S, Vt = np.linalg.svd(Xc / np.sqrt(max(1, Xc.shape[0] - 1)),
                                 full_matrices=False)
        eigvals = S**2
        # Determine number of components
        cum = np.cumsum(eigvals) / np.sum(eigvals) if np.sum(eigvals) > 0 else np.ones_like(eigvals)
        r = np.searchsorted(cum, float(pca_variance)) + 1
        # Clip to valid range
        r = max(1, min(r, Vt.shape[0]))
        Vr = Vt[:r, :]                   # r x D
        lambdar = eigvals[:r]            # r
        inv_sqrt = 1.0 / np.sqrt(np.where(lambdar > 0, lambdar, 1.0))
        # Transformation matrix
        W = (Vr * inv_sqrt[:, None])     # r x D
        return {"mode": "pca_whiten", "mu": mu, "sd": None, "W": W, "b": None,
                "kept_dims": np.arange(r)}

    raise ValueError(f"Unknown transform mode: {mode}")


def _apply_transform(X: np.ndarray, T: Dict[str, Any]) -> np.ndarray:
    """Apply a transformation to a set in native space."""
    m = T["mode"]
    if m == "none":
        return X
    if m == "standardize":
        return (X - T["mu"]) / T["sd"]
    if m == "pca_whiten":
        return (T["W"] @ (X - T["mu"]).T).T
    raise ValueError(f"Unknown transform mode: {m}")


def _chunk_ranges(n: int, batch_size: Optional[int]):
    """Provide the starting and end indices of a batch."""
    if batch_size is None or batch_size <= 0 or batch_size >= n:
        yield 0, n
        return
    s = 0
    while s < n:
        e = min(n, s + batch_size)
        yield s, e
        s = e

def _sigma_to_space(sig_native: np.ndarray, T: dict) -> np.ndarray:
    """Map 1-sigma vector from native into the transform space described by T."""
    mode = (T.get("mode") or "none").lower()
    if mode == "none" or T.get("sd") is None and T.get("W") is None:
        return sig_native
    if mode == "standardize":
        sd = T["sd"]
        return sig_native / sd
    if mode == "pca_whiten":
        # approximate diag propagation through linear map
        W = T["W"]  # shape r x D
        return np.sqrt(np.clip((W**2) @ (sig_native**2), 1e-12, np.inf))
    # Fallback
    return sig_native

# ----------------------------
# Base class
# ----------------------------

class BaseBinner:
    """
    Abstract candidate selector interface.

    Required methods
    ----------------
    fit(grid)
    candidates(y_native, sigmas_native=None, **kwargs) -> (indices, aux_level_or_radius)
    dims property
    save(path), load(path) for persistence
    batch_candidates(Y_native, SIG_native=None, **kwargs) convenience method
    """

    def fit(self, grid: ModelGrid) -> "BaseBinner":
        raise NotImplementedError

    def candidates(self,
                   y_native: np.ndarray,
                   sigmas_native: Optional[np.ndarray] = None,
                   **kwargs) -> Tuple[np.ndarray, Optional[int]]:
        raise NotImplementedError

    def info(self) -> Dict[str, Any]:
        return {}

    def save(self, path: str) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> "BaseBinner":
        raise NotImplementedError

    def batch_candidates(self,
                         Y_native: np.ndarray,
                         SIG_native: Optional[np.ndarray] = None,
                         batch_size: Optional[int] = None,
                         **kwargs) -> Tuple[List[np.ndarray], List[Optional[int]]]:
        """Select candidates in batch."""
        M = Y_native.shape[0]
        out_idx: List[np.ndarray] = [None] * M
        out_aux: List[Optional[int]] = [None] * M
        # Loop over all elements
        for s, e in _chunk_ranges(M, batch_size):
            for m in range(s, e):
                sig = None if SIG_native is None else SIG_native[m]
                # normalize once here so subclasses can assume the unified keys
                idx, aux = self.candidates(Y_native[m], sig, **kwargs)
                out_idx[m] = idx
                out_aux[m] = aux
        return out_idx, out_aux

    def plot_candidates(self,
                        grid: ModelGrid,
                        y_native: np.ndarray,
                        sigmas_native: Optional[np.ndarray] = None,
                        *,
                        dims_plot: Optional[Sequence[int]] = None,
                        # candidate selection kwargs
                        candidate_kwargs: Optional[dict] = None,
                        # plotting controls
                        plot_space: str = "native",   # "native" | "transformed" | "both"
                        max_background: int = 50_000,
                        background_alpha: float = 0.15,
                        background_ms: float = 1.0,
                        cand_alpha: float = 0.9,
                        cand_ms: float = 4.0,
                        query_ms: float = 10.0,
                        show_sigma_boxes: bool = True,
                        sigma_scale: float = 1.0,
                        figsize: Optional[Tuple[float, float]] = None,
                        suptitle: Optional[str] = None,
                        random_state: Optional[int] = 42
                        ):
        """
        Pairwise visualisation of binner candidates versus the model grid.

        Parameters
        ----------
        grid : ModelGrid
        y_native : np.ndarray
        sigmas_native : np.ndarray
        dims_plot : list of int, optional
        candidate_kwargs : dict, optional
        TODO
        """
        # Resolve dims to plot
        P = grid.n_observables
        if dims_plot is None:
            dims_plot = getattr(self, "dims", list(range(P)))
        dims_plot = list(dims_plot)
        D = len(dims_plot)

        # Get candidates (always in native space)
        if candidate_kwargs is None:
            candidate_kwargs = dict()
        cand_idx, aux = self.candidates(
                y_native=y_native,
                sigmas_native=sigmas_native,
                **candidate_kwargs)

        # Common data
        X = grid.observables
        names = grid.observable_names
        rng = np.random.default_rng(random_state)
        n_back = min(X.shape[0], max_background)
        n_candidates = cand_idx.size
        frac_candidates = n_candidates / X.shape[0]

        if n_back > 0:
            bg_idx = rng.choice(X.shape[0], size=n_back, replace=False)
        else:
            bg_idx = np.array([], dtype=int)
        xq_nat = y_native[dims_plot]
        sq_nat = None if sigmas_native is None else sigmas_native[dims_plot]

        # Try to retrieve transform (RectBinner / KDTreeBinner expose _T)
        T = getattr(self, "_T", {"mode": "none"})
        has_transform = isinstance(T, dict) and (T.get("mode") or "none").lower() != "none"

        def _make_pairplot(X_bg, X_c, xq, sq, axis_labels, title_suffix):
            if figsize is None:
                fsz = (2.2 * D, 2.2 * D)
            else:
                fsz = figsize
            fig, axes = plt.subplots(D, D, figsize=fsz,
                                     squeeze=False, sharex="col")

            # Diagonals
            for i in range(D):
                ax = axes[i, i]
                if X_bg is not None:
                    ax.hist(X_bg[:, i], bins=40, density=False, alpha=0.25,
                            lw=0, label="grid",
                            log=True)
                if X_c.size:
                    ax.hist(X_c[:, i], bins=40, density=False, alpha=0.5, lw=0, label="candidates",
                            log=True)
                ax.axvline(xq[i], color="k", lw=1.2)
                ax.axvline(xq[i] - sq[i], color="k", lw=1.2, ls=":")
                ax.axvline(xq[i] + sq[i], color="k", lw=1.2, ls=":")
                ax.set_ylabel("density")
                ax.set_xlabel(axis_labels[i])
                if i == 0:
                    ax.legend(frameon=False, fontsize="small")

            # Off-diagonals
            for i in range(D):
                for j in range(i + 1, D):
                    ax = axes[j, i]  # lower triangle
                    if X_bg is not None:
                        ax.plot(X_bg[:, i], X_bg[:, j], ",",
                                alpha=background_alpha,
                                ms=background_ms, color="0.5")
                    if X_c.size:
                        ax.plot(X_c[:, i], X_c[:, j], ",", alpha=cand_alpha,
                                ms=cand_ms, color="C0")
                    ax.plot(xq[i], xq[j], marker="+", ms=query_ms, color="fuchsia")
                    if show_sigma_boxes and (sq is not None):
                        wi = sigma_scale * float(sq[i])
                        wj = sigma_scale * float(sq[j])
                        if np.isfinite(wi) and np.isfinite(wj) and wi > 0 and wj > 0:
                            ax.add_patch(plt.Rectangle((xq[i] - wi, xq[j] - wj),
                                                       2 * wi, 2 * wj,
                                                       fill=False, ec="k", lw=0.8, alpha=0.9))
                    if j == D - 1:
                        ax.set_xlabel(axis_labels[i])
                    if i == 0:
                        ax.set_ylabel(axis_labels[j])
                    # hide upper triangle
                    axes[i, j].axis("off")

            title = f"Candidates ({n_candidates} / {frac_candidates:.3f}%) vs grid" + (f" [{title_suffix}]" if title_suffix else "")
            if suptitle:
                title = f"{title} – {suptitle}"
            fig.suptitle(title)
            fig.tight_layout()
            return fig, axes

            # end _make_pairplot

        # Build native-space plot
        out = {}
        make_native = plot_space in ("native", "both")
        make_trans = plot_space in ("transformed", "both") and has_transform

        if make_native:
            if bg_idx.size > 0:
                X_bg_nat = X[bg_idx][:, dims_plot]
            else:
                X_bg_nat = None
            X_c_nat = X[cand_idx][:, dims_plot] if cand_idx.size else np.empty((0, D))
            figN, axN = _make_pairplot(
                X_bg_nat, X_c_nat, xq_nat, sq_nat,
                [names[d] for d in dims_plot],
                "native"
            )
            if plot_space == "native":
                return figN, axN, cand_idx, aux
            out["native"] = (figN, axN)

        # Transformed-space plot (if available)
        if make_trans:
            # transform background and candidates using the same transform the binner fitted
            X_dims = X[:, dims_plot]
            X_bg = X_dims[bg_idx]
            X_c = X_dims[cand_idx] if cand_idx.size else np.empty((0, D))
            X_bg_z = _apply_transform(X_bg, T)
            X_c_z = _apply_transform(X_c, T) if X_c.size else X_c
            xq_z = _apply_transform(xq_nat[None, :], T)[0]
            sq_z = None if sq_nat is None else _sigma_to_space(sq_nat, T)
            # label axes according to transform used
            mode = (T.get("mode") or "none").lower()
            if mode == "standardize":
                labels = [f"z({names[d]})" for d in dims_plot]
            elif mode == "pca_whiten":
                labels = [f"PC{i+1}" for i in range(X_bg_z.shape[1])]
            else:
                labels = [names[d] for d in dims_plot]
            figZ, axZ = _make_pairplot(
                X_bg_z, X_c_z, xq_z, sq_z,
                labels,
                f"transformed: {mode}"
            )
            if plot_space == "transformed":
                return figZ, axZ, cand_idx, aux
            out["transformed"] = (figZ, axZ)

        # If "both" requested but no transform available, we only had native
        if plot_space == "both" and "transformed" not in out:
            # gentle hint in the title that no transform was available
            fig, axes = out["native"]
            fig.suptitle(fig._suptitle.get_text() + " – no transform available")
            return out, cand_idx, aux

        return out, cand_idx, aux

# ----------------------------
# Rectangular multi-resolution binner
# ----------------------------
@dataclass
class RectBinner(BaseBinner):
    """
    Multi-resolution rectilinear bins with optional whitening and quantile edges.

    Parameters
    ----------
    dims : list of int
        Observable columns to index (D dims).
    levels : int
        Number of refinement levels. Bin count per dim grows ~ base_bins * 2^level.
    base_bins : int
        Bins per dim at level 0.
    transform : {"none","standardize","pca_whiten"}
        Linear transform fitted on training data for selected dims.
    pca_variance : float in (0,1]
        Variance fraction to keep if transform == "pca_whiten".
    edges_mode : {"quantile","linear"}
        Strategy for bin edges per dim.

    Notes
    -----
    - Uses quantile edges by default to equalize mass across bins.
    - Supports two selection modes in candidates():
        a) error expansion: expand by multiples of transformed sigmas
        b) target_k expansion: expand neighbourhood until at least target_k models are included
    """

    dims: List[int]
    levels: int = 5
    base_bins: int = 6
    transform: str = "standardize"
    pca_variance: float = 1.0
    edges_mode: str = "quantile"
    mode: str = "sigma"
    target_factor: float = 2.0
    expand_factor: float = 2.0
    target_k: Optional[int] = None
    k: Optional[int] = None
    max_expand_steps: int = 4

    # fitted attributes
    _T: Dict[str, Any] = field(default_factory=dict)
    _edges: List[List[np.ndarray]] = field(default_factory=list)  # per level per dim
    _layers: List[Dict[Tuple[int, ...], np.ndarray]] = field(default_factory=list)  # key -> model idx
    _grid_n: int = 0
    _D_eff: int = 0

    def __post_init__(self):
        # normalise and route through setter
        self.dims = list(self.dims)

    @property
    def dims(self) -> List[int]:
        return self._dims

    @dims.setter
    def dims(self, value: Sequence[int]) -> None:
        self._dims = list(value)

    def fit(self, grid: ModelGrid) -> "RectBinner":
        """Fit the binner"""
        # Select a subset of the grid dimensions
        X = np.asarray(grid.observables[:, self.dims], float)
        self._grid_n = X.shape[0]
        # fit transform
        self._T = _fit_transform(X, self.transform, pca_variance=self.pca_variance)
        Xz = _apply_transform(X, self._T)
        self._D_eff = Xz.shape[1]

        self._edges = []
        self._layers = []
        # Store the characteristic bin size per level and dimension
        self._level_bin_size = np.full((self.levels, self._D_eff),
                                       fill_value=np.nan)
        # bounds per dim to build edges
        if self.edges_mode == "quantile":
            for lev in range(self.levels):
                # Number of bins at this level
                n_bins = self.base_bins * (2 ** lev)
                q = np.linspace(0.0, 1.0, n_bins + 1)
                edges = np.quantile(Xz, q, axis=0, method="linear").T
                span = edges[:, -1] - edges[:, 0]
                pad = np.where(span > 0, 1e-6 * span, 1.0)
                edges[:, 0] -= pad
                edges[:, -1] += pad                 
                self._level_bin_size[lev] = np.diff(edges, axis=1).mean(axis=1)
                self._edges.append(edges)

                # assign cells
                cell_idx = self._digitize_nd(Xz, edges)  # (N, D_eff)
                keys = [tuple(cell_idx[i]) for i in range(Xz.shape[0])]
                layer = {}
                for i, key in enumerate(keys):
                    (layer.setdefault(key, [])).append(i)
                self._layers.append({k: np.asarray(v, dtype=np.int64) for k, v in layer.items()})

        elif self.edges_mode == "linear":
            lo = np.nanmin(Xz, axis=0)
            hi = np.nanmax(Xz, axis=0)
            pad = 0.01 * (hi - lo + 1e-6)
            lo -= pad
            hi += pad
            for lev in range(self.levels):
                n_bins = self.base_bins * (2 ** lev)
                edges = np.array(
                    [np.linspace(lo[d], hi[d], n_bins + 1) for d in range(self._D_eff)])
                self._level_bin_size[lev] = np.diff(edges, axis=1).mean(axis=1)
                self._edges.append(edges)
                cell_idx = self._digitize_nd(Xz, edges)
                keys = [tuple(cell_idx[i]) for i in range(Xz.shape[0])]
                layer = {}
                for i, key in enumerate(keys):
                    (layer.setdefault(key, [])).append(i)
                self._layers.append({k: np.asarray(v, dtype=np.int64) for k, v in layer.items()})
        else:
            raise ValueError("edges_mode must be 'quantile' or 'linear'")
        return self

    @staticmethod
    def _digitize_nd(X: np.ndarray, edges_per_dim: List[np.ndarray]) -> np.ndarray:
        return np.stack(
            [np.clip(np.digitize(X[:, d], ed) - 1, 0, len(ed) - 2) for d, ed in enumerate(edges_per_dim)],
            axis=1)

    def _choose_level_by_sigma(self, sig_trans: np.ndarray,
                               target_factor: float = 2.0) -> int:
        """Choose the binning level given an vector of uncertainties."""
        target = target_factor * sig_trans
        score = np.sum(
            np.log((self._level_bin_size + 1e-12) / (target + 1e-12)), axis=1)
        best_lev = np.argmin(score)
        return best_lev

    def _choose_level_by_targetK(self, y_trans: np.ndarray, target_k: int) -> int:
        for lev in reversed(range(self.levels)):
            edges = self._edges[lev]
            cell = self._digitize_nd(y_trans[None, :], edges)[0]
            layer = self._layers[lev]
            n_here = len(layer.get(tuple(cell), ()))
            if n_here >= target_k:
                return lev
        # Otherwise return the lowest level
        return lev

    def _expand_neighbour_keys(self,
                               y_trans: np.ndarray,
                               lev: int,
                               sig_trans: Optional[np.ndarray] = None,
                               expand_factor: float = 2.0,
                               min_keys_radius: int = 0) -> List[Tuple[int, ...]]:
        edges = self._edges[lev]
        cell = self._digitize_nd(y_trans[None, :], edges)[0]
        widths = self._level_bin_size[lev]
        if sig_trans is not None:
            half_span = np.ceil((expand_factor * sig_trans) / widths).astype(int)
        else:
            half_span = np.full(self._D_eff, min_keys_radius, dtype=int)
        ranges = [range(max(0, cell[d] - half_span[d]),
                        min(len(edges[d]) - 1, cell[d] + half_span[d]) + 1)
                  for d in range(self._D_eff)]
        return list(product(*ranges))

    def candidates(self,
                   y_native: np.ndarray,
                   sigmas_native: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, int]:
        """
        Candidate selector.
        """
        # Build transformed query and sigma
        Xq = y_native[self.dims]
        y_trans = _apply_transform(Xq[None, :], self._T)[0]
        sig_trans = None
        if sigmas_native is not None:
            s_native = sigmas_native[self.dims]
            tmode = self._T.get("mode", "none")
            if tmode == "standardize":
                sig_trans = s_native / self._T["sd"]
            elif tmode == "pca_whiten":
                W = self._T["W"]
                sig_trans = np.sqrt(np.clip((W**2 @ (s_native**2)), 1e-12, np.inf))
            else:
                sig_trans = s_native

        if self.mode == "target_k":
            if self.target_k is None or int(self.target_k) <= 0:
                self.target_k = 1
            lev = self._choose_level_by_targetK(y_trans, target_k=int(self.target_k))
            layer = self._layers[lev]
            base_key = tuple(self._digitize_nd(y_trans[None, :], self._edges[lev])[0])
            idxs = [layer[k] for k in (base_key,) if k in layer]
            total = sum(len(a) for a in idxs)
            step = 1
            while total < self.target_k and step <= int(self.max_expand_steps):
                keys = self._expand_neighbour_keys(y_trans, lev, sig_trans=None,
                                                   expand_factor=1.0, min_keys_radius=step)
                idxs = [layer[k] for k in keys if k in layer]
                total = sum(len(a) for a in idxs)
                step += 1
            if not idxs:
                return np.array([], dtype=np.int64), lev
            return np.unique(np.concatenate(idxs)), lev

        if sig_trans is None:
            raise ValueError("sigmas_native must be provided for mode='radius'")
        lev = self._choose_level_by_sigma(
            sig_trans, target_factor=float(self.target_factor))
        keys = self._expand_neighbour_keys(
            y_trans, lev, sig_trans=sig_trans, expand_factor=self.expand_factor)
        layer = self._layers[lev]
        idxs = [layer[k] for k in keys if k in layer]
        if not idxs:
            return np.array([], dtype=np.int64), lev
        return np.unique(np.concatenate(idxs)), lev

    def info(self) -> Dict[str, Any]:
        return {
            "name": "RectBinner",
            "levels": self.levels,
            "base_bins": self.base_bins,
            "transform": self.transform,
            "pca_variance": self.pca_variance,
            "edges_mode": self.edges_mode,
            "n_models": self._grid_n,
            "D_eff": self._D_eff,
        }

    def save(self, path: str) -> None:
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        blob = {
            "cls": "RectBinner",
            "dims": self.dims,
            "levels": self.levels,
            "base_bins": self.base_bins,
            "transform": self.transform,
            "pca_variance": self.pca_variance,
            "edges_mode": self.edges_mode,
            "_T": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self._T.items()},
            "_edges": [[e.tolist() for e in lev] for lev in self._edges],
            "_layers": {str(li): {str(k): v.tolist() for k, v in layer.items()} for li, layer in enumerate(self._layers)},
            "_level_bin_size": self._level_bin_size.tolist(),
            "_grid_n": self._grid_n,
            "_D_eff": self._D_eff,
        }
        with open(path, "w") as f:
            json.dump(blob, f)

    @classmethod
    def load(cls, path: str) -> "RectBinner":
        with open(path, "r") as f:
            d = json.load(f)
        obj = cls(dims=list(d["dims"]),
                  levels=int(d["levels"]),
                  base_bins=int(d["base_bins"]),
                  transform=d["transform"],
                  pca_variance=float(d["pca_variance"]),
                  edges_mode=d["edges_mode"])
        T = d["_T"]
        for k in ("mu", "sd", "W", "b"):
            if k in T and T[k] is not None:
                T[k] = np.asarray(T[k], float)
        obj._T = T
        obj._edges = [[np.asarray(e, float) for e in lev] for lev in d["_edges"]]
        layers = []
        for li in range(len(obj._edges)):
            layer_d = d["_layers"][str(li)]
            layer: Dict[Tuple[int, ...], np.ndarray] = {}
            for k_str, v in layer_d.items():
                # Robust tuple parsing from the str(key) form
                k_clean = k_str.strip().strip("()")
                parts = [p.strip() for p in k_clean.split(",") if p.strip() != ""]
                key = tuple(int(p) for p in parts)
                layer[key] = np.asarray(v, dtype=np.int64)
            layers.append(layer)
        obj._layers = layers
        obj._grid_n = int(d["_grid_n"])
        obj._D_eff = int(d["_D_eff"])
        obj._level_bin_size = np.asarray(d["_level_bin_size"], float)
        return obj

    def plot_bins(self, grid):
        X = np.asarray(grid.observables[:, self.dims], float)
        Xz = _apply_transform(X, self._T)
        dim = Xz.shape[1]

        fig, axs = plt.subplots(nrows=self.levels, ncols=dim, sharex="col",
                                sharey=True, constrained_layout=True)
        for lev in range(self.levels):
            for d in range(dim):
                ax = axs[lev, d]
                ax.hist(Xz[:, d], bins="auto", color="k", alpha=0.7, log=True)
                for e in self._edges[lev][d]:
                    ax.axvline(e, color="r", ls="-", lw=0.8)
        
        for ith, s in enumerate(self.dims):
            ax = axs[-1, ith]
            ax.set_xlabel(f"z({grid.observable_names[s]})")

        return fig, axs


# ----------------------------
# KDTree binner
# ----------------------------
@dataclass
class KDTreeBinner(BaseBinner):
    """
    Nearest-neighbour binner using cKDTree over transformed dims.

    Parameters
    ----------
    dims : list of int
        Observable columns to index (D dims).
    transform : {"none","standardize","pca_whiten"}
        Linear transform to fit on training dims.
    pca_variance : float
        Variance to keep if using pca_whiten.
    leafsize : int
        KDTree leaf size, trade off build vs query speed.
    select_mode : {"radius","knn"}
        Candidate selection mode.
    target_k : int
        Target number of candidates for select_mode="knn".
    radius_factor : float
        Radius scaling factor for select_mode="radius".
    """

    dims: List[int]
    transform: str = "standardize"
    pca_variance: float = 1.0
    leafsize: int = 100
    select_mode: str = "knn"  # "radius" | "knn"
    target_k: int = 100
    radius_factor: float = 2.0
    radius_shape: str = "ellipsoid"  # "ball" | "ellipsoid" | "box"
    sigma_floor: float = 1e-12

    _T: Dict[str, Any] = field(default_factory=dict)
    _tree: Optional[cKDTree] = field(default=None)
    _Xz: Optional[np.ndarray] = field(default=None)

    def __post_init__(self):
        self.dims = list(self.dims)

    @property
    def dims(self) -> List[int]:
        return self._dims

    @dims.setter
    def dims(self, value: Sequence[int]) -> None:
        self._dims = list(value)

    def fit(self, grid: ModelGrid) -> "KDTreeBinner":
        X = np.asarray(grid.observables[:, self.dims], float)
        self._T = _fit_transform(X, self.transform, pca_variance=self.pca_variance)
        Xz = _apply_transform(X, self._T)
        self._Xz = Xz
        self._tree = cKDTree(Xz, leafsize=self.leafsize)
        return self

    def _sigma_to_tree_space(self, s_native: np.ndarray) -> np.ndarray:
        """
        Map per-dimension 1-sigma uncertainties from native dims -> tree space dims.

        Returns
        -------
        s_z : (D,) ndarray
            1-sigma vector in the KDTree coordinate system (same dims as Xz).
        """
        s = np.asarray(s_native[self.dims], float)
        tmode = self._T.get("mode", "none")

        if tmode == "standardize":
            s_z = s / self._T["sd"]
        elif tmode == "pca_whiten":
            # Xz = (X - mu) @ W.T + b (typical), here we propagate diagonal covariance
            # If Cov_native = diag(s^2), then var along dim j in z-space is sum_i W[j,i]^2 s_i^2
            W = np.asarray(self._T["W"], float)  # shape (D, D) (assumed)
            s_z = np.sqrt(np.clip((W**2) @ (s**2), self.sigma_floor, np.inf))
        else:
            s_z = s

        # ensure strictly positive
        s_z = np.maximum(s_z, self.sigma_floor)
        return s_z

    def candidates(self,
                   y_native: np.ndarray,
                   sigmas_native: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, Optional[int]]:
        """Candidate selector."""
        if self._tree is None or self._Xz is None:
            raise RuntimeError("fit must be called before candidates")

        x = np.asarray(y_native[self.dims], float)[None, :]
        xz = _apply_transform(x, self._T)[0]

        # Map modes
        if self.select_mode == "knn":
            kk = self.target_k
            kk = max(1, min(kk, self._Xz.shape[0]))
            d, ind = self._tree.query(xz, k=kk)
            ind = np.atleast_1d(ind).astype(np.int64)
            return np.unique(ind), kk

        # radius mode
        if sigmas_native is None:
            raise ValueError("sigmas_native required for select_mode='radius'")

        s_z = self._sigma_to_tree_space(sigmas_native)

        shape = (self.radius_shape or "ball").lower()
        if shape == "ball":
            r = float(self.radius_factor * np.linalg.norm(s_z, ord=2))
            inds = self._tree.query_ball_point(xz, r=r)
        else:
            # conservative candidate fetch with a scalar radius,
            # then exact filter.
            if shape == "ellipsoid":
                # if ||(dx / s_z)||_2 <= radius_factor, then ||dx||_2 <= radius_factor * ||s_z||_2
                r_cons = float(self.radius_factor * np.linalg.norm(s_z, ord=2))
                cand = np.asarray(self._tree.query_ball_point(xz, r=r_cons),
                                  dtype=np.int64)

                if cand.size:
                    dz = self._Xz[cand] - xz[None, :]
                    u = dz / s_z[None, :]                  # normalize per dimension
                    ok = (np.sum(u*u, axis=1) <= (self.radius_factor**2))
                    inds = cand[ok].tolist()
                else:
                    inds = []

            elif shape == "box":
                # if max_i |dx_i|/s_i <= radius_factor, then ||dx||_2 <= radius_factor * ||s_z||_2
                # (still a valid conservative ball for candidates)
                r_cons = float(self.radius_factor * np.linalg.norm(s_z, ord=2))
                cand = np.asarray(self._tree.query_ball_point(xz, r=r_cons), dtype=np.int64)

                if cand.size:
                    dz = np.abs(self._Xz[cand] - xz[None, :])
                    ok = np.all(dz <= (self.radius_factor * s_z)[None, :], axis=1)
                    inds = cand[ok].tolist()
                else:
                    inds = []

            else:
                raise ValueError(f"Unknown radius_shape={self.radius_shape!r} "
                                 f"(expected 'ball','ellipsoid','box')")

        inds = np.asarray(inds, dtype=np.int64)
        if inds.size == 0:
            # fallback to 1-NN
            _, ind2 = self._tree.query(xz, k=1)
            return np.asarray([int(ind2)], dtype=np.int64), None
        return np.unique(inds), None

    def info(self) -> Dict[str, Any]:
        n = 0 if self._Xz is None else self._Xz.shape[0]
        d = 0 if self._Xz is None else self._Xz.shape[1]
        return {"name": "KDTreeBinner", "n": n, "d": d,
                "leafsize": self.leafsize, "transform": self.transform}

    def save(self, path: str) -> None:
        if self._Xz is None:
            raise RuntimeError("fit must be called before save")
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        blob = {
            "cls": "KDTreeBinner",
            "dims": self.dims,
            "transform": self.transform,
            "pca_variance": self.pca_variance,
            "_T": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self._T.items()},
            "_Xz": self._Xz.tolist(),
            "leafsize": self.leafsize,
        }
        with open(path, "w") as f:
            json.dump(blob, f)

    @classmethod
    def load(cls, path: str) -> "KDTreeBinner":
        with open(path, "r") as f:
            d = json.load(f)
        obj = cls(dims=list(d["dims"]),
                  transform=d["transform"],
                  pca_variance=float(d.get("pca_variance", 1.0)),
                  leafsize=int(d.get("leafsize", 40)))
        T = d["_T"]
        for k in ("mu", "sd", "W", "b"):
            if k in T and T[k] is not None:
                T[k] = np.asarray(T[k], float)
        obj._T = T
        Xz = np.asarray(d["_Xz"], float)
        obj._Xz = Xz
        obj._tree = cKDTree(Xz, leafsize=obj.leafsize)
        return obj
