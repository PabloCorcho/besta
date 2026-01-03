# pst/transforms.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class LinearStandardiser:
    """
    Simple per-dimension standardisation: x_std = (x - mean) / sd.

    Notes
    -----
    - Uses ddof=0 by default (population std), matching your ModelGrid.
    - If sd == 0, it is set to 1 to avoid division by zero.
    """
    mean: Optional[np.ndarray] = None
    sd: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, *, ddof: int = 0) -> "LinearStandardiser":
        X = np.asarray(X)
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0, ddof=ddof)
        sd = np.where(sd == 0.0, 1.0, sd)
        self.mean = mu
        self.sd = sd
        return self

    @property
    def is_fit(self) -> bool:
        return self.mean is not None and self.sd is not None

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError("LinearStandardiser must be fit before calling transform().")
        return (np.asarray(X) - self.mean) / self.sd

    def inverse_transform(self, X_std: np.ndarray) -> np.ndarray:
        if not self.is_fit:
            raise RuntimeError("LinearStandardiser must be fit before calling inverse_transform().")
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


@dataclass
class MagTransform:
    """
    Magnitude transform.

    mag = -2.5 log10(flux) + zero_point
    flux = 10^((zero_point - mag)/2.5)
    """
    zero_point: float = 0.0
    eps: float = 1e-30

    def flux_to_mag(self, flux: np.ndarray) -> np.ndarray:
        f = np.maximum(np.asarray(flux), self.eps)
        return (-2.5 * np.log10(f)) + self.zero_point

    def mag_to_flux(self, mag: np.ndarray) -> np.ndarray:
        m = np.asarray(mag)
        return np.power(10.0, (self.zero_point - m) / 2.5)

    def to_dict(self) -> Dict[str, Any]:
        return {"zero_point": float(self.zero_point), "eps": float(self.eps)}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MagTransform":
        return cls(zero_point=float(d.get("zero_point", 0.0)),
                   eps=float(d.get("eps", 1e-30)))
