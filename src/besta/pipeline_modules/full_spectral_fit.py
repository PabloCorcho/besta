from besta.pipeline_modules.base_module import BaseModule
import numpy as np

from cosmosis.datablock import names as section_names
from cosmosis.datablock import SectionOptions
from besta import kinematics
from besta import spectrum

class FullSpectralFitModule(BaseModule):
    name = "FullSpectralFit"

    def __init__(self, options):
        """Set-up the COSMOSIS sampler.
        Args:
            options: options from startup file (i.e. .ini file)
        Returns:
            config: parameters or objects that are passed to
                the sampler.

        """
        options = self.parse_options(options)
        # Pipeline values file
        self.config = {}
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
        flux_model = sfh_model.model.compute_SED(
            self.config["ssp_model"], t_obs=sfh_model.today, allow_negative=False
        ).value

        # Kinematics
        velscale = self.config["velscale"]
        # Kinematics
        sigma_pixel = block["parameters", "los_sigma"] / velscale
        veloffset_pixel = block["parameters", "los_vel"] / velscale
        # Build the kernel. TOO SLOW? Initialise only once?
        kernel_model = kinematics.GaussHermite(
            4,
            mean=veloffset_pixel,
            stddev=sigma_pixel,
            h3=block["parameters", "los_h3"],
            h4=block["parameters", "los_h4"],
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
            self.config["wavelength"], flux_model, a_v=block["parameters", "av"]
        ).value

        weights = self.config["weights"] * mask
        normalization = np.nanmedian(
            self.config["flux"][weights > 0] / flux_model[weights > 0]
        )
        block["parameters", "normalization"] = normalization
        return flux_model * normalization, weights

    def execute(self, block):
        """Function executed by sampler
        This is the function that is executed many times by the sampler. The
        likelihood resulting from this function is the evidence on the basis
        of which the parameter space is sampled.
        """
        valid, penalty = self.config["sfh_model"].parse_datablock(block)
        if not valid:
            print("Invalid")
            block[section_names.likelihoods, f"{self.name}_like"] = -1e20 * penalty
            block["parameters", "normalization"] = 0.0
            return 0
        # Obtain parameters from setup
        cov = self.config["cov"]
        flux_model, weights = self.make_observable(block)
        # Calculate likelihood-value of the fit
        like = self.log_like(
            self.config["flux"][weights > 0], flux_model[weights > 0], cov[weights > 0]
        )
        # Final posterior for sampling
        block[section_names.likelihoods, f"{self.name}_like"] = like
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
