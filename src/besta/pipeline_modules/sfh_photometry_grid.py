"""Photometric SFH inference pipeline module backed by a model grid."""

import pickle
import numpy as np
from astropy import units as u

from cosmosis.datablock import names as section_names
from cosmosis.datablock import SectionOptions

from besta.pipeline_modules.base_module import PhotometryFitModule, GridFitMixin
from besta.logging import get_logger

logger = get_logger(__name__)

class SFHPhotometryGridModule(PhotometryFitModule, GridFitMixin):
    """Infer SFH parameters from photometry using direct grid interpolation."""

    name = "SFHPhotometryGrid"

    def __init__(self, options):
        """Set up the module from a CosmoSIS configuration block."""

        super().__init__(options, likelihood_kind="photometry")
        options = self.parse_options(options)
        # Pipeline values file
        self.prepare_observed_photometry(options)
        self.prepare_grid_model(options)

    def make_observable(self, block, parse=False):
        """Create the photometric model by interpolating the model grid.

        Parameters
        ----------
        block : DataBlock
            Current CosmoSIS parameter block.
        parse : bool, optional
            Kept for API compatibility; parameters are read directly from
            ``block``.

        Returns
        -------
        ndarray
            Normalized model fluxes.
        """
        targets = [block["parameters", k] for k in self.config["model_grid"].target_names]
        # microJy / Msun at 10 parsec
        flux_model = self.config["model_grid"].interpolate_observables(
            targets, k=self.config.get("knn", 1)).squeeze()
        normalization = np.mean(self.config["photometry_flux"] / flux_model)
        block["parameters", "normalization"] = normalization
        return flux_model * normalization

    def execute(self, block):
        """Evaluate the grid-interpolated photometric likelihood."""
        flux_model = self.make_observable(block)
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
            self.config["photometry_flux_var"],
        )
        block[section_names.likelihoods, self.like_name] = like
        return 0

    def cleanup(self):
        """Release resources after a grid-based photometry fit run."""
        pass

def setup(options):
    """Create the CosmoSIS-facing module instance."""

    options = SectionOptions(options)
    mod = SFHPhotometryGridModule(options)
    return mod


def execute(block, mod):
    """Run one likelihood evaluation for the configured module."""

    mod.execute(block)
    return 0


def cleanup(mod):
    """Release module resources after sampling."""

    mod.cleanup()
