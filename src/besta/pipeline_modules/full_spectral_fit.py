"""Full spectral fitting pipeline module."""

from besta.pipeline_modules.base_module import SpectraFitModule
import numpy as np

from cosmosis.datablock import names as section_names
from cosmosis.datablock import SectionOptions
from besta import kinematics
from besta import spectrum
from besta.logging import get_logger

logger = get_logger(__name__)

class FullSpectralFitModule(SpectraFitModule):
    """Fit stellar populations and kinematics directly from galaxy spectra."""

    name = "FullSpectralFit"

    def __init__(self, options, **kwargs):
        """
        Set up the full spectral fit module.

        Parameters
        ----------
        options : dict or DataBlock
            Options from the startup configuration.
        **kwargs : dict
            Extra keyword arguments forwarded to ``SpectraFitModule``.
        """
        super().__init__(options, likelihood_kind="spectra", **kwargs)
        options = self.parse_options(options)

        # Check for the necessary options and prepare the models
        if not options.has_value("velscale"):
            raise ValueError("Option 'velscale' is required for setting up FullSpectralFitModule.")

        self.prepare_observed_spectra(options)
        self.prepare_ssp_model(options)
        self.prepare_sfh_model(options)
        self.prepare_extinction_law(options)
        self.prepare_legendre_polynomials(options)
        self.prepare_losvd_kernel(options)
        self._losvd_kernel = self.config["losvd_kernel"]

    @spectrum.legendre_decorator
    def make_observable(self, block, parse=False):
        """Create the spectra model from the input parameters"""
        # Stellar population synthesis
        sfh_model = self.config["sfh_model"]
        if parse:
            sfh_model.parse_datablock(block)
        luminosity_model = sfh_model.model.compute_SED(
            self.config["ssp_model"], t_obs=sfh_model.today, allow_negative=False
        )
        flux_model = 1e10 * luminosity_model.to_value(self._default_luminosity_units
        ) / self.config["dl_sq"]

        # Kinematics
        self._losvd_kernel.parse_parameters(block)
        # Perform the convolution
        flux_model = self._losvd_kernel.convolve(flux_model)
        # Track those pixels at the edges
        mask = flux_model > 0
        mask[:self._losvd_kernel.size // 2] = False
        mask[-self._losvd_kernel.size // 2:] = False
        # Sample to observed resolution
        extra_pixels = self.config["extra_pixels"]
        pixels = slice(extra_pixels, -extra_pixels)
        flux_model = flux_model[pixels]
        mask = mask[pixels]

        # Apply dust extinction
        dust_model = self.config["extinction_law"]
        flux_model = dust_model.apply_extinction(
            self.config["wavelength"], flux_model, a_v=block["dust_attenuation", "a_v"]
        ).value

        weights = self.config["weights"] * mask
        normalization = np.nanmedian(
            self.config["flux"][weights > 0] / flux_model[weights > 0]
        )
        block["extra", "stellar_mass"] = np.log10(normalization) + 10
        return flux_model * normalization, weights

    def execute(self, block):
        """Function executed by sampler
        This is the function that is executed many times by the sampler. The
        likelihood resulting from this function is the evidence on the basis
        of which the parameter space is sampled.
        """
        valid, penalty = self.config["sfh_model"].parse_datablock(block)
        if not valid:
            # To track invalid samples users can set debug=T
            # logger.warning("Invalid sample")
            block[section_names.likelihoods, self.like_name] = -1e20 * penalty
            block["extra", "stellar_mass"] = np.nan
            return 0
        # Obtain parameters from setup
        flux_model, weights = self.make_observable(block)
        # Calculate likelihood-value of the fit
        good_pixels = weights > 0
        like = self.log_like(self.config["flux"][good_pixels],
                             flux_model[good_pixels],
                             self.config["ivar"][good_pixels] * weights[good_pixels])
        # To make it compatible with photometric likelihoods
        like /= np.sum(good_pixels)
        # Final posterior for sampling
        block[section_names.likelihoods, self.like_name] = like
        return 0

    def cleanup(self):
        """Release resources after a full spectral fit run."""
        pass


def setup(options):
    """Create the CosmoSIS-facing module instance."""

    options = SectionOptions(options)
    mod = FullSpectralFitModule(options)
    return mod


def execute(block, mod):
    """Run one likelihood evaluation for the configured module."""

    mod.execute(block)
    return 0


def cleanup(mod):
    """Release module resources after sampling."""

    mod.cleanup()

module = FullSpectralFitModule
