"""Photometric galaxy-fitting pipeline module."""

from astropy import units as u
from besta.pipeline_modules.base_module import PhotometryFitModule
import numpy as np

from cosmosis.datablock import names as section_names
from cosmosis.datablock import SectionOptions

import warnings
from astropy.units import UnitsWarning
from besta.logging import get_logger

logger = get_logger(__name__)

# Ignore only astropy units warnings
warnings.simplefilter('ignore', category=UnitsWarning)

class GalaxyPhotometryModule(PhotometryFitModule):
    """Fit a galaxy model to broadband photometric measurements."""

    name = "GalaxyPhotometry"

    def __init__(self, options, **kwargs):
        """Set up the module from a CosmoSIS configuration block."""

        super().__init__(options, likelihood_kind="photometry", **kwargs)
        options = self.parse_options(options)
        self.prepare_observed_photometry(options)
        self.prepare_amplitude_weights()
        self.prepare_galaxy(options)

        # Set parameters fixed in this module
        self.config["galaxy"].redshift.fixed = True

    def prepare_amplitude_weights(self):
        """Precompute the static terms of the maximum-likelihood amplitude.

        The amplitude ``s = sum(iv * d * m) / sum(iv * m**2)`` uses detections
        only (upper/lower limits do not constrain a Gaussian amplitude). ``iv``
        and ``iv * d`` do not change during sampling, so they are computed once
        here and each likelihood call only needs two dot products.
        """
        flux = np.asarray(self.config["photometry_flux"], dtype=float)
        var = np.asarray(self.config["photometry_flux_var"], dtype=float)
        usable = np.isfinite(flux) & np.isfinite(var) & (var > 0)
        for limits in (self.config.get("photometry_upper_limit"),
                       self.config.get("photometry_lower_limit")):
            if limits is not None:
                usable &= ~np.asarray(limits, dtype=bool)
        inv_var = np.zeros_like(flux)
        inv_var[usable] = 1.0 / var[usable]
        self.config["amplitude_inv_var"] = inv_var
        self.config["amplitude_weighted_flux"] = np.where(usable, inv_var * flux, 0.0)

    def make_observable(self, block, parse=False, include_spec=False):
        """Create the photometric model from the input parameters."""
        if parse:
            # This updates the SFH parameters
            self.config["sfh_model"].parse_datablock(block)

        # Update parameters for each remaining component
        parameters = self.get_galaxy_parameters(block)

        galaxy = self.config["galaxy"]
        galaxy.update_parameters(parameters, strict=False, validate=False)
        # Synthesis
        flux_model = 1e10 * galaxy.emission_photometry(to_obs_frame=True).to_value(
            self.config["photometry_flux_unit"])
        sfh_model = self.config["sfh_model"]
        if sfh_model.use_mass_normalization:
            # Maximum-likelihood amplitude from detections only, using the
            # terms precomputed in prepare_amplitude_weights()
            if "amplitude_inv_var" not in self.config:
                self.prepare_amplitude_weights()
            denominator = np.dot(self.config["amplitude_inv_var"],
                                 flux_model * flux_model)
            normalization = (
                np.dot(self.config["amplitude_weighted_flux"], flux_model)
                / denominator if denominator > 0 else np.nan)
            if np.isfinite(normalization) and normalization > 0:
                block["extra", "stellar_mass"] = np.log10(normalization) + 10
            else:
                logger.warning("Undefined or non-positive photometric amplitude")
                normalization = 0
                block["extra", "stellar_mass"] = np.nan
        else:
            normalization = 1.0
            block["extra", "stellar_mass"] = sfh_model.model.stellar_mass_formed(
                sfh_model.today
            ).to_value("Msun")

        # Save SFH mass-fraction times
        if self.config.get("save_t_frac_at", False):
            for frac in self.config.get("t_frac_at", []):
                self.get_t_frac_at(block, self.config["sfh_model"], frac)
        if self.config.get("save_ssfr_over_tau", False):
            for tau in self.config.get("ssfr_tau", []):
                self.get_ssfr_over_tau(block, self.config["sfh_model"], tau)
        # Mostly for visualization purposes
        if include_spec:
            full_spec = 1e10 * galaxy.emission_spectrum(to_obs_frame=True).to_value(
                    self.config["photometry_flux_unit"],
                    u.spectral_density(galaxy.target_wavelength))
            return flux_model * normalization, full_spec * normalization
        return flux_model * normalization

    def execute(self, block):
        """Function executed by sampler
        This is the function that is executed many times by the sampler. The
        likelihood resulting from this function is the evidence on the basis
        of which the parameter space is sampled.
        """
        valid, log_prior = self.config["sfh_model"].parse_datablock(block)
        if log_prior is None:
            log_prior = 0.0
        if not valid:
            # Reject with a large negative log-likelihood.
            block[section_names.likelihoods, self.like_name] = -1e20
            block["extra", "stellar_mass"] = np.nan
            return 0
        # Obtain parameters from setup
        flux_model = self.make_observable(block)

        # Calculate likelihood-value of the fit
        like = self.log_like(self.config["photometry_flux"],
                             flux_model,
                             self.config["photometry_flux_var"],
                             is_lower=self.config["photometry_lower_limit"],
                             is_upper=self.config["photometry_upper_limit"])
        # Final posterior for sampling (includes the SFH log-prior)
        block[section_names.likelihoods, self.like_name] = like + log_prior
        return 0

    def cleanup(self):
        """Release resources after a galaxy photometry fit run."""
        pass


def setup(options):
    """Create the CosmoSIS-facing module instance."""

    options = SectionOptions(options)
    mod = GalaxyPhotometryModule(options)
    return mod


def execute(block, mod):
    """Run one likelihood evaluation for the configured module."""

    mod.execute(block)
    return 0


def cleanup(mod):
    """Release module resources after sampling."""

    mod.cleanup()

module = GalaxyPhotometryModule
