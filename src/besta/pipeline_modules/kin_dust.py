from besta.pipeline_modules.base_module import BaseModule
import numpy as np
from scipy.optimize import nnls

from cosmosis.datablock import names as section_names
from cosmosis.datablock import SectionOptions
from besta import kinematics
from besta import spectrum

class KinDustModule(BaseModule):
    name = "KinDust"

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
        if options.has_value("save_ssp"):
            self.solution = []
            self.ssp_output = options["save_ssp"]
        else:
            self.ssp_output, self.solution = None, None
            print(f"Saving SSP solution at {self.ssp_output}")

        self.prepare_observed_spectra(options)
        self.prepare_ssp_model(options)
        self.prepare_extinction_law(options)
        self.prepare_legendre_polynomials(options)

    @spectrum.legendre_decorator
    def make_observable(self, block, parse=False):
        """Create the spectra model from the input parameters"""
        dust_model = self.config["extinction_law"]
        sed, mask = kinematics.convolve_ssp(
            self.config,
            los_vel=block["parameters", "los_vel"],
            los_sigma=block["parameters", "los_sigma"],
            los_h3=block["parameters", "los_h3"],
            los_h4=block["parameters", "los_h4"],
        )
        sed *= dust_model.get_extinction(
            self.config["wavelength"], a_v=block["parameters", "av"]
        )[np.newaxis]
        weights = self.config["weights"] * mask
        sq_weights = np.sqrt(weights)
        solution, _ = nnls(
            sq_weights[:, None] * sed.T,
            sq_weights * self.config["flux"],
            maxiter=sed.shape[0] * 10,
        )
        if self.ssp_output:
            self.solution.append(solution)
        flux_model = np.sum(sed * solution[:, np.newaxis], axis=0)
        return flux_model, weights

    def execute(self, block):
        """Function executed by sampler
        This is the function that is executed many times by the sampler. The
        likelihood resulting from this function is the evidence on the basis
        of which the parameter space is sampled.
        """
        # Obtain parameters from setup
        cov = self.config["cov"]
        flux_model, weights = self.make_observable(block)
        # Calculate likelihood-value of the fit
        good_pixels = weights > 0
        like = self.log_like(self.config["flux"][good_pixels],
                             flux_model[good_pixels], cov[good_pixels],
                             weights=weights[good_pixels])
        # Final posterior for sampling
        block[section_names.likelihoods, "KinDust_like"] = like
        return 0

    def cleanup(self):
        if self.ssp_output is not None:
            np.savetxt(self.ssp_output, self.solution)


def setup(options):
    options = SectionOptions(options)
    mod = KinDustModule(options)
    return mod


def execute(block, mod):
    mod.execute(block)
    return 0


def cleanup(mod):
    mod.cleanup()
