"""Spectroscopic galaxy-fitting pipeline module."""

from besta.pipeline_modules.base_module import SpectraFitModule
import numpy as np

from cosmosis.datablock import names as section_names
from cosmosis.datablock import SectionOptions
from besta import spectrum
from besta.logging import get_logger

logger = get_logger(__name__)

class GalaxySpectraModule(SpectraFitModule):
    """Fit a galaxy emission model to observed spectra."""

    name = "GalaxySpectra"

    def __init__(self, options, **kwargs):
        """Set up the module from a CosmoSIS configuration block."""

        super().__init__(options, likelihood_kind="spectra", **kwargs)
        options = self.parse_options(options)
        self.prepare_observed_spectra(options)
        self.prepare_galaxy(options)
        self.prepare_legendre_polynomials(options)
        self.prepare_losvd_kernel(options)
        self._losvd_kernel = self.config["losvd_kernel"]

        # Set parameters fixed in this module
        self.config["galaxy"].redshift.fixed = True

    def make_observable(self, block, parse=False):
        """Create the spectra model from the input parameters"""
        if parse:
            # This updates the SFH parameters
            self.config["sfh_model"].parse_datablock(block)

        # Update parameters for each remaining component
        parameters = self.get_galaxy_parameters(block)

        galaxy = self.config["galaxy"]
        galaxy.update_parameters(parameters, strict=False)
        # Synthesis
        flux_model = self.luminosity_values(galaxy.emission_spectrum(
            to_obs_frame=False)) / self.config["dl_sq"]

        # Kinematics: convolve and trim to the observed grid
        self._losvd_kernel.parse_parameters(block)
        flux_model = self.convolve_losvd_and_trim(flux_model)
        weights = self.config["weights"].copy()
        # Multiplicative polynomial, applied before the normalization
        flux_model = flux_model * self.legendre_polynomial(block)

        sfh_model = self.config["sfh_model"]
        if sfh_model.use_mass_normalization:
            # Maximum-likelihood amplitude (same weights and ivar as the likelihood)
            flux_model, normalization = self.normalize_to_data(
                flux_model, weights, block)
            block["extra", "stellar_mass"] = np.log10(normalization)
        else:
            block["extra", "stellar_mass"] = np.log10(
                sfh_model.model.stellar_mass_formed(
                    sfh_model.today).to_value("Msun"))
        # Save SFH mass-fraction times and sSFRs (one mass-history evaluation)
        self.save_sfh_extras(block, self.config["sfh_model"])

        return flux_model, weights

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
        flux_model, weights = self.make_observable(block)
        # Calculate likelihood-value of the fit
        good_pixels = weights > 0
        ivar_eff = self.get_effective_ivar(block)
        like = self.log_like(self.config["flux"][good_pixels],
                             flux_model[good_pixels],
                             ivar_eff[good_pixels] * weights[good_pixels],
                             include_norm=True)
        # Final posterior for sampling (includes the SFH log-prior)
        block[section_names.likelihoods, self.like_name] = like + log_prior

        if self.config.get("save_chi2", False):
            block["extra", self.like_name + "_chi2"] = -2 * like

        return 0

    def cleanup(self):
        """Release resources after a galaxy spectra fit run."""
        pass


def setup(options):
    """Create the CosmoSIS-facing module instance."""

    options = SectionOptions(options)
    mod = GalaxySpectraModule(options)
    return mod


def execute(block, mod):
    """Run one likelihood evaluation for the configured module."""

    mod.execute(block)
    return 0


def cleanup(mod):
    """Release module resources after sampling."""

    mod.cleanup()

module = GalaxySpectraModule
