"""Spectroscopic galaxy-fitting pipeline module."""

from besta.pipeline_modules.base_module import SpectraFitModule
import numpy as np

from cosmosis.datablock import names as section_names
from cosmosis.datablock import SectionOptions
from besta import kinematics
from besta import spectrum
from besta.logging import get_logger

logger = get_logger(__name__)

class GalaxySpectraModule(SpectraFitModule):
    """Fit a galaxy emission model to observed spectra."""

    name = "GalaxySpectra"

    def __init__(self, options, **kwargs):
        """Set up the module from a CosmoSIS configuration block."""

        super().__init__(options, **kwargs)
        options = self.parse_options(options)
        self.prepare_observed_spectra(options)
        self.prepare_galaxy(options)
        self.prepare_legendre_polynomials(options)

        # Set parameters fixed in this module
        self.config["galaxy"].redshift.fixed = True

    @spectrum.legendre_decorator
    def make_observable(self, block, parse=False):
        """Create the spectra model from the input parameters"""
        if parse:
            # This updates the SFH parameters
            self.config["sfh_model"].parse_datablock(block)

        # Update parameters for each remaining component
        keys = block.keys()
        values = [block[s, k] for (s, k) in keys if self.config["sfh_model"].sect_name not in s]
        keys = [".".join((s, k)) for (s, k) in keys if self.config["sfh_model"].sect_name not in s]
        parameters = dict(zip(keys, values))

        galaxy = self.config["galaxy"]
        galaxy.update_parameters(parameters, strict=False)
        # Synthesis
        flux_model = 1e10 * galaxy.emission_spectrum(
            to_obs_frame=False).to_value("1e-16 erg / (s Angstrom)") / self.config["dl_sq"]

        # Kinematics #TODO: this should be done by PST stars.kinematics
        velscale = self.config["velscale"]
        sigma_pixel = block["kinematics", "los_sigma"] / velscale
        veloffset_pixel = block["kinematics", "los_vel"] / velscale

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
