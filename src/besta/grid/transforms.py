"""Transforms used by model grids and emulators."""
#
#
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class LinearStandardiser:
    """
    Per-dimension linear standardisation: x_std = (x - mean) / sd.
    """

    mean: Optional[np.ndarray] = None
    sd: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, *, ddof: int = 0) -> "LinearStandardiser":
	"""Fit the standariser parameters."""
       X = np.asarray(X)
       mu = np.nanmean(X, axis=0)
       std = np.nanstd(X, axis=0, ddof=ddof)
       std = np.where(std == 0.0, 1.0, std)
       self.mean = mu
       self.sd = std
       return self

    @property
    def is_fit(self) -> bool:
        return self.mean is not None and self.sd is not None

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError(
                "LinearStandardiser must be fit before calling transform()."
            )
        return (np.asarray(X) - self.mean) / self.sd

    def inverse_transform(self, X_std: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError(
                "LinearStandardiser must be fit before calling inverse_transform()."
            )
        return np.asarray(X_std) * self.sd + self.mean

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": None if self.mean is None else self.mean.tolist(),
            "sd": None if self.sd is None else self.sd.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LinearStandardiser":
        mean = None if d.get("mean") is None else np.asarray(d["mean"], dtype=float)
        sd = None if d.get("sd") is None else np.asarray(d["sd"], dtype=float)
        return cls(mean=mean, sd=sd)

# TODO: vmin/vmax linear transform


