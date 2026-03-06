from besta.pipeline_modules.base_module import SpectraFitModule
import numpy as np

from cosmosis.datablock import names as section_names
from cosmosis.datablock import SectionOptions
from besta import kinematics
from besta import spectrum
from besta.logging import get_logger

logger = get_logger(__name__)

class FullSpectralFitModule(SpectraFitModule):
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
        super().__init__(options, **kwargs)
        options = self.parse_options(options)
        self.prepare_observed_spectra(options)
        self.prepare_ssp_model(options)
        self.prepare_sfh_model(options)
        self.prepare_extinction_law(options)
        self.prepare_legendre_polynomials(options)

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
        flux_model = 1e10 * luminosity_model.to_value("1e-16 erg / (s Angstrom)"
        ) / self.config["dl_sq"]

        # Kinematics
        velscale = self.config["velscale"]
        # Kinematics
        sigma_pixel = block["kinematics", "los_sigma"] / velscale
        veloffset_pixel = block["kinematics", "los_vel"] / velscale
        # Build the kernel. TOO SLOW? Initialise only once?
        kernel_model = kinematics.GaussHermite(
            4,
            mean=veloffset_pixel,
            stddev=sigma_pixel,
            h3=block["kinematics", "los_h3"],
            h4=block["kinematics", "los_h4"],
        )
        kernel_n_pixel = 10 * np.clip(int(np.round(np.abs(veloffset_pixel) + sigma_pixel)), 1,
                                      None) + 1
        kernel = kinematics.get_losvd_kernel(
            kernel_model,
            x_size=kernel_n_pixel
        )
        # Perform the convolution
        flux_model = kinematics.convolve_spectra_with_kernel(flux_model, kernel)
        # Track those pixels at the edges
        mask = flux_model > 0
        mask[: int(10 * sigma_pixel)] = False
        mask[-int(10 * sigma_pixel) :] = False
        # Sample to observed resolution
        extra_pixels = self.config["extra_pixels"]
        pixels = slice(extra_pixels, -extra_pixels)
        flux_model = flux_model[pixels]
        mask = mask[pixels]

        # Apply dust extinction
        dust_model = self.config["extinction_law"]
        flux_model = dust_model.apply_extinction(
            self.config["wavelength"], flux_model, a_v=block["dust.extinction", "a_v"]
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
            logger.warning("Invalid sample")
            block[section_names.likelihoods, self.like_name] = -1e20 * penalty
            block["extra", "stellar_mass"] = np.nan
            return 0
        # Obtain parameters from setup
        cov = self.config["var"]
        flux_model, weights = self.make_observable(block)
        # Calculate likelihood-value of the fit
        good_pixels = weights > 0
        like = self.log_like(self.config["flux"][good_pixels],
                             flux_model[good_pixels],
                             cov[good_pixels],
                             weights=weights[good_pixels])
        # Final posterior for sampling
        block[section_names.likelihoods, self.like_name] = like
        return 0

    def cleanup(self):
        pass


def setup(options):
    options = SectionOptions(options)
    mod = FullSpectralFitModule(options)
    return mod


def execute(block, mod):
    mod.execute(block)
    return 0


def cleanup(mod):
    mod.cleanup()

module = FullSpectralFitModule
