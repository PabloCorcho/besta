"""Photometric SFH inference pipeline module backed by an emulator."""

import pickle
import numpy as np
from astropy import units as u

from cosmosis.datablock import names as section_names
from cosmosis.datablock import SectionOptions

from besta.pipeline_modules.base_module import PhotometryFitModule, EmulatorMixin
from besta.logging import get_logger

logger = get_logger(__name__)

class SFHPhotometryEmulatorModule(PhotometryFitModule, EmulatorMixin):
    """Infer SFH parameters from photometry using an emulator-backed model."""

    name = "SFHPhotometryEmulator"

    def __init__(self, options):
        """Set up the module from a CosmoSIS configuration block."""

        super().__init__(options, likelihood_kind="photometry")
        options = self.parse_options(options)
        # Pipeline values file
        self.prepare_observed_photometry(options)
        self.prepare_emulator(options)

    def make_observable(self, block, parse=False):
        """Create the photometric model and emulator uncertainty.

        Parameters
        ----------
        block : DataBlock
            Current CosmoSIS parameter block.
        parse : bool, optional
            Kept for API compatibility; parameters are read directly from
            ``block``.

        Returns
        -------
        tuple of ndarray
            Model fluxes and model-flux uncertainties after normalization.
        """
        targets = [block["parameters", k] for k in self.config["emulator"].target_names]
        # microJy / Msun at 10 parsec
        flux_model, flux_model_err = self.config["emulator"].predict_with_error(
            targets)
        flux_model = flux_model.squeeze()
        flux_model_err = flux_model_err.squeeze()
        normalization = np.mean(self.config["photometry_flux"] / flux_model)
        block["parameters", "normalization"] = normalization
        return flux_model * normalization, flux_model_err * normalization

    def execute(self, block):
        """Evaluate the emulator-backed photometric likelihood."""
        flux_model, flux_model_err = self.make_observable(block)
        if not np.all(np.isfinite(flux_model)):
            logger.warning(
                "Invalid sample found; setting log-likelihood to large negative value."
            )
            block[section_names.likelihoods, self.like_name] = -1e5
            block["parameters", "normalization"] = 0.0
            return 0
        # Final posterior for sampling
        like = self.log_like(
            self.config["photometry_flux"],
            flux_model,
            self.config["photometry_flux_var"] + flux_model_err**2,
        )
        block[section_names.likelihoods, self.like_name] = like
        return 0

    def cleanup(self):
        """Release resources after an emulator-backed photometry fit run."""
        pass

def setup(options):
    """Create the CosmoSIS-facing module instance."""

    options = SectionOptions(options)
    mod = SFHPhotometryEmulatorModule(options)
    return mod


def execute(block, mod):
    """Run one likelihood evaluation for the configured module."""

    mod.execute(block)
    return 0


def cleanup(mod):
    """Release module resources after sampling."""

    mod.cleanup()
