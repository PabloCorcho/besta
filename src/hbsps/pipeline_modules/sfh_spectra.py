from hbsps.pipeline_modules.base_module import BaseModule
import numpy as np

from cosmosis.datablock import names as section_names
from cosmosis.datablock import SectionOptions
from hbsps import kinematics

class SFHSpectraModule(BaseModule):
    name = "SFHSpectra"
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

        if options.has_value("los_vel"):
            if options.has_value("los_sigma"):
                if options.has_value("los_h3"):
                    h3 = options["los_h3"]
                else:
                    h3 = 0
                if options.has_value("los_h4"):
                    h4 = options["los_h4"]
                else:
                    h4 = 0
                print(f"\nConvolving SSP models with Gauss-Hermite LOSVD")
                ssp, mask = kinematics.convolve_ssp_model(
                    self.config,
                    options["los_sigma"], options["los_vel"], h3, h4)
                self.config['ssp_model'] = ssp
                self.config['weights'] *= mask
                print(f"Valid pixels: {np.count_nonzero(mask)} out of {mask.size}")
        else:
            print("No kinematic information was provided")

        if options.has_value("ExtinctionLaw"):
            av = options["av"]
            self.prepare_extinction_law(options)
            print(f"\nReddening SSP models using Av={av}")
            self.config['ssp_model'] = self.config["extinction_law"].redden_ssp_model(
                self.config['ssp_model'], a_v=av)

    def make_observable(self, block):
        sfh_model = self.config['sfh_model']
        flux_model = sfh_model.model.compute_SED(self.config['ssp_model'],
										     t_obs=sfh_model.today,
											 allow_negative=False).value
        normalization = np.sum(
            self.config['flux'] / flux_model * self.config["weights"]
            ) / np.sum(self.config["weights"])
        block['parameters', 'normalization'] = normalization
        return flux_model * normalization

    def execute(self, block):
        valid, penalty = self.config['sfh_model'].parse_datablock(block)
        if not valid:
            print("Invalid")
            block[section_names.likelihoods, "SFHSpectra_like"] = -1e20 * penalty
            block['parameters', 'normalization'] = 0.0
            return 0
        flux_model = self.make_observable(block)
        like = self.log_like(self.config['flux'] * self.config["weights"],
                          flux_model * self.config["weights"],
                          self.config['cov'])
        # Final posterior for sampling
        block[section_names.likelihoods, "SFHSpectra_like"] = like
        return 0

def setup(options):
        options = SectionOptions(options)
        mod = SFHSpectraModule(options)
        return mod

def execute(block, mod):
    mod.execute(block)
    return 0

def cleanup(mod):
    mod.cleanup()
