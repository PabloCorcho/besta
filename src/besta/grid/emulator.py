# besta/grid/emulator.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np
from joblib import dump, load

from besta.io import _mkdir
from besta.grid.transforms import LinearStandardiser


# -----------------------------------------------------------------------------
# utils
# -----------------------------------------------------------------------------
def _ensure_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        return a[:, None]
    return a

def _as_f32(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    return a.astype(np.float32, copy=False) if a.dtype != np.float32 else a


# -----------------------------------------------------------------------------
# TransformPack
# -----------------------------------------------------------------------------
@dataclass
class TransformPack:
    """
    Serializable inference-time transforms.

    - Standardises targets -> features X
    - Optionally standardises y (in whatever user-chosen y_space)
    """
    target_names: List[str]
    observable_names: List[str]

    target_standardiser: LinearStandardiser
    y_standardiser: Optional[LinearStandardiser] = None  # standardises *y_space* outputs

    def X_from_targets(self, targets: np.ndarray) -> np.ndarray:
        return self.target_standardiser.transform(targets)

    def y_to_model_space(self, y_space: np.ndarray) -> np.ndarray:
        return y_space if self.y_standardiser is None else self.y_standardiser.transform(y_space)

    def y_from_model_space(self, y_model: np.ndarray) -> np.ndarray:
        return y_model if self.y_standardiser is None else self.y_standardiser.inverse_transform(y_model)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_names": list(self.target_names),
            "observable_names": list(self.observable_names),
            "target_standardiser": self.target_standardiser.to_dict(),
            "y_standardiser": None if self.y_standardiser is None else self.y_standardiser.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TransformPack":
        return cls(
            target_names=list(d["target_names"]),
            observable_names=list(d["observable_names"]),
            target_standardiser=LinearStandardiser.from_dict(d["target_standardiser"]),
            y_standardiser=None if d.get("y_standardiser") is None else LinearStandardiser.from_dict(d["y_standardiser"]),
        )

    @classmethod
    def from_grid(
        cls,
        grid: Any,  # besta.grid.grid.ModelGrid
        *,
        y_space: str,                 # "mag" or "flux" (metadata only)
        standardize_y: bool = True,
        y_for_stats: Optional[np.ndarray] = None,
        stats_mask: Optional[np.ndarray] = None,
    ) -> "TransformPack":
        """
        Build transforms from a ModelGrid.

        Notes
        -----
        - Fits target_standardiser on grid.targets (or mask).
        - Fits y_standardiser on y_for_stats if provided, else:
            - if y_space == "flux": uses grid.observables (or mask)
            - if y_space == "mag" : you MUST pass y_for_stats (magnitude array) explicitly
              to avoid baking mag logic into ModelGrid/emulator.
        """
        # Ensure target standardiser exists and is fit
        if not hasattr(grid, "target_standardiser") or grid.target_standardiser is None:
            raise AttributeError("ModelGrid must have attribute target_standardiser (LinearStandardiser).")
        if not getattr(grid.target_standardiser, "is_fit", False):
            grid.fit_target_standardiser(mask=stats_mask)

        target_std = grid.target_standardiser

        y_std = None
        if standardize_y:
            if y_for_stats is not None:
                Y = y_for_stats if stats_mask is None else y_for_stats[stats_mask]
                y_std = LinearStandardiser().fit(np.asarray(Y, dtype=np.float32), ddof=0)
            else:
                if y_space == "flux":
                    Y = grid.observables if stats_mask is None else grid.observables[stats_mask]
                    y_std = LinearStandardiser().fit(np.asarray(Y, dtype=np.float32), ddof=0)
                elif y_space == "mag":
                    raise ValueError(
                        "y_space='mag' requires y_for_stats to be provided (magnitudes array). "
                        "This keeps MagTransform logic out of the emulator."
                    )
                else:
                    raise ValueError("y_space must be 'mag' or 'flux'")

        return cls(
            target_names=list(grid.target_names),
            observable_names=list(grid.observable_names),
            target_standardiser=target_std,
            y_standardiser=y_std,
        )


# -----------------------------------------------------------------------------
# EmulatorConfig
# -----------------------------------------------------------------------------
@dataclass
class EmulatorConfig:
    y_space: str = "mag"              # metadata only: "mag", "flux", "ew" etc.
    standardize_y: bool = True
    predict_batch_size: int = 131072
    version: str = "EmulatorV5"

    # error model
    error_model: str = "zero"         # "zero" | "ensemble_std"
    error_floor: float = 0.0          # constant floor in y_space units


# -----------------------------------------------------------------------------
# Emulator
# -----------------------------------------------------------------------------
class Emulator:
    """Generic emulator.

    Parameters
    ----------
    model : Any
        Trained regression model with a .predict(X) method.
    transforms : TransformPack
        Transforms for targets -> X and model_space -> y_space.
    config : EmulatorConfig, optional
        Emulator configuration.
    
    Methods
    -------
    predict(targets, return_model_space=False) -> np.ndarray
        Predict observables in y_space for given targets.
    predict_error(targets) -> np.ndarray
        Predict errors in y_space for given targets.
    predict_with_error(targets) -> Tuple[np.ndarray, np.ndarray]
        Predict observables and errors in y_space for given targets.
    save(outdir, name="emulator", compress=("lz4", 3)) -> Dict[str, str]
        Save the emulator to disk.
    load(outdir, name="emulator") -> Emulator
        Load an emulator from disk.
    """

    def __init__(self, model: Any, transforms: TransformPack, config: Optional[EmulatorConfig] = None):
        self.model = model
        self.transforms = transforms
        self.config = config or EmulatorConfig()

    def _predict_model_space(self, X: np.ndarray) -> np.ndarray:
        y = self.model.predict(X)
        y = _ensure_2d(y)
        return _as_f32(y)

    def predict(self, targets: np.ndarray, *, return_model_space: bool = False) -> np.ndarray:
        targets = _ensure_2d(targets)
        X_all = _as_f32(self.transforms.X_from_targets(targets))

        bs = int(self.config.predict_batch_size) if self.config.predict_batch_size else 0
        if bs <= 0 or X_all.shape[0] <= bs:
            y_model = self._predict_model_space(X_all)
        else:
            out = []
            for i in range(0, X_all.shape[0], bs):
                out.append(self._predict_model_space(X_all[i:i + bs]))
            y_model = np.vstack(out)

        if return_model_space:
            return y_model

        # model space -> y_space
        if self.config.standardize_y:
            return self.transforms.y_from_model_space(y_model)
        return y_model

    def predict_error(self, targets: np.ndarray) -> np.ndarray:
        y = self.predict(targets)
        err = np.zeros_like(y, dtype=np.float32)
        if self.config.error_floor and self.config.error_floor > 0:
            err = err + np.float32(self.config.error_floor)
        return err

    def predict_with_error(self, targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        y = self.predict(targets)
        return y, self.predict_error(targets)

    # persistence
    def save(self, outdir: str, name: str = "emulator", compress: Union[Tuple[str, int], None] = ("lz4", 3)) -> Dict[str, str]:
        _mkdir(outdir)
        model_path = os.path.join(outdir, f"{name}.joblib")
        meta_path = os.path.join(outdir, f"{name}.json")

        dump(self.model, model_path, compress=compress)

        meta = {
            "type": "Emulator",
            "config": asdict(self.config),
            "transforms": self.transforms.to_dict(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)

        return {"model": model_path, "meta": meta_path}

    @classmethod
    def load(cls, outdir: str, name: str = "emulator") -> "Emulator":
        model_path = os.path.join(outdir, f"{name}.joblib")
        meta_path = os.path.join(outdir, f"{name}.json")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        model = load(model_path)
        transforms = TransformPack.from_dict(meta["transforms"])
        config = EmulatorConfig(**meta["config"])
        return cls(model=model, transforms=transforms, config=config)


# -----------------------------------------------------------------------------
# EnsembleEmulator
# -----------------------------------------------------------------------------
class EnsembleEmulator:
    """
    Ensemble wrapper.

    - predict() -> mean prediction in y_space
    - predict_error() -> std across members in y_space (+ optional floor)
    """

    def __init__(self, members: List[Emulator], error_floor: float = 0.0):
        if not members:
            raise ValueError("EnsembleEmulator requires at least one member.")
        self.members = members
        self.error_floor = float(error_floor)

        # sanity checks: identical transforms and y_space
        ref_t = members[0].transforms.to_dict()
        ref_space = members[0].config.y_space
        for m in members[1:]:
            if m.transforms.to_dict() != ref_t:
                raise ValueError("All ensemble members must share identical transforms.")
            if m.config.y_space != ref_space:
                raise ValueError("All ensemble members must share y_space.")

        self.transforms = members[0].transforms
        self.y_space = ref_space

    def predict(self, targets: np.ndarray) -> np.ndarray:
        preds = [m.predict(targets) for m in self.members]
        P = np.stack(preds, axis=0)
        return np.mean(P, axis=0).astype(np.float32)

    def predict_error(self, targets: np.ndarray, ddof: int = 0) -> np.ndarray:
        preds = [m.predict(targets) for m in self.members]
        P = np.stack(preds, axis=0)
        err = np.std(P, axis=0, ddof=ddof).astype(np.float32)
        if self.error_floor > 0:
            err = np.sqrt(err**2 + self.error_floor**2).astype(np.float32)
        return err

    def predict_with_error(self, targets: np.ndarray, ddof: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        preds = [m.predict(targets) for m in self.members]
        P = np.stack(preds, axis=0)
        mean = np.mean(P, axis=0).astype(np.float32)
        err = np.std(P, axis=0, ddof=ddof).astype(np.float32)
        if self.error_floor > 0:
            err = np.sqrt(err**2 + self.error_floor**2).astype(np.float32)
        return mean, err

    def save(self, outdir: str, name: str = "ensemble") -> Dict[str, Any]:
        _mkdir(outdir)
        member_dirs = []
        for k, mem in enumerate(self.members):
            d = os.path.join(outdir, f"{name}_member{k:02d}")
            mem.save(d, name="emulator")
            member_dirs.append(d)

        index = {
            "type": "EnsembleEmulator",
            "name": name,
            "member_dirs": member_dirs,
            "error_floor": self.error_floor,
        }
        index_path = os.path.join(outdir, f"{name}.json")
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2, sort_keys=True)

        return {"index": index_path, "member_dirs": member_dirs}

    @classmethod
    def load(cls, outdir: str, name: str = "ensemble") -> "EnsembleEmulator":
        index_path = os.path.join(outdir, f"{name}.json")
        with open(index_path, "r") as f:
            index = json.load(f)

        members = [Emulator.load(d, name="emulator") for d in index["member_dirs"]]
        return cls(members=members, error_floor=float(index.get("error_floor", 0.0)))
