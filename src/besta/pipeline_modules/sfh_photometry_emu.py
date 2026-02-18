import pickle
import numpy as np
from astropy import units as u

from cosmosis.datablock import names as section_names
from cosmosis.datablock import SectionOptions

from besta.pipeline_modules.base_module import PhotometryFitModule, EmulatorMixin

class SFHPhotometryEmulatorModule(PhotometryFitModule, EmulatorMixin):
    name = "SFHPhotometryEmulator"

    def __init__(self, options):
        """Set-up the COSMOSIS sampler.
        Args:
            options: options from startup file (i.e. .ini file)
        Returns:
            config: parameters or objects that are passed to
                the sampler.

        """
        super().__init__(options)
        options = self.parse_options(options)
        # Pipeline values file
        self.prepare_observed_photometry(options)
        self.prepare_emulator(options)

    def make_observable(self, block, parse=False):
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
        flux_model, flux_model_err = self.make_observable(block)
        if not np.all(np.isfinite(flux_model)):
            print("Invalid sample found; setting log-likelihood to large negative value.")
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
        pass

def setup(options):
    options = SectionOptions(options)
    mod = SFHPhotometryEmulatorModule(options)
    return mod


def execute(block, mod):
    mod.execute(block)
    return 0


def cleanup(mod):
    mod.cleanup()
