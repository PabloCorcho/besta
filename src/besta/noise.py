

"""Noise models used by BESTA pipeline modules."""

from __future__ import annotations

import numpy as np


class NoiseModel:
    """Base noise model returning the input inverse variance unchanged."""

    def __init__(self, config):
        self.name = config.get("NoiseModelName", "noise")
        self.ivar = np.asarray(config["ivar"], dtype=float)

    def parse_parameters(self, block):
        """Parse noise parameters from a CosmoSIS DataBlock (if any)."""
        return None

    def inverse_variance(self, block):
        """Return the effective inverse variance for the current sample."""
        return self.ivar


class MultiplicativeNoiseModel(NoiseModel):
    """Scale inverse variance by a sampled factor ``noise/beta``."""

    def parse_parameters(self, block):
        beta = block[self.name, "beta"]
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError("Noise parameter 'noise/beta' must be finite and > 0.")
        return beta

    def inverse_variance(self, block):
        beta = self.parse_parameters(block)
        return self.ivar * beta


def make_noise_model(model_name: str, config):
    """Create a noise model instance by class name."""
    model_name = str(model_name or "NoiseModel").strip()
    model = globals().get(model_name)
    if model is None or not isinstance(model, type) or not issubclass(model, NoiseModel):
        raise ValueError(
            f"Unknown noise model '{model_name}'. "
            "Available models: NoiseModel, MultiplicativeNoiseModel"
        )
    return model(config)