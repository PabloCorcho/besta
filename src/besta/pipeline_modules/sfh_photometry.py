import pickle
import numpy as np
from astropy import units as u

from cosmosis.datablock import names as section_names
from cosmosis.datablock import SectionOptions

from besta.pipeline_modules.base_module import BaseModule
from besta import kinematics
from besta.config import extinction as extinction_conf

class SFHPhotometryModule(BaseModule):
    name = "SFHPhotometry"

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
        self.config = {"redshift": options["redshift"]}
        print(f"Input source redshift: {self.config['redshift']}")
        self.prepare_observed_photometry(options)
        self.prepare_ssp_model(options)
        self.prepare_sfh_model(options)
        self.prepare_extinction_law(options)

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
                print(f"Convolving SSP models with Gauss-Hermite LOSVD")
                ssp, mask = kinematics.convolve_ssp_model(
                    self.config, options["los_sigma"], options["los_vel"], h3, h4
                )
                self.config["ssp_model"] = ssp
                self.config["weights"] *= mask
                print("Valid pixels: ", np.count_nonzero(mask), mask.size)
        else:
            print("No kinematic information was provided")

        if options.has_value("av"):
            av = options["av"]
            print(f"Reddening SSP models using Av={av}")
            self.config["ssp_model"] = self.config["extinction_law"].redden_ssp_model(
                self.config["ssp_model"], av
            )

        if options.has_value("PhotometryGrid"):
            # Load pre-computed grid
            with open(options["PhotometryGrid"], 'rb') as file:
                grid = pickle.load(file)
                self.config["photometry_grid"] = grid["photometry_grid"]
                self.config["av_grid"] = grid["av_grid"]
        else:
            print("Producing photometry extinction grid")
            dust_model = self.config["extinction_law"]
            av_grid = np.linspace(extinction_conf["a_v"]["min"],
                                extinction_conf["a_v"]["max"],
                                extinction_conf["a_v"]["steps"])
            self.config["av_grid"] = av_grid
            ssps = [
                dust_model.redden_ssp_model(self.config["ssp_model"], a_v=av)
                for av in av_grid
            ]
            all_photometry = np.zeros(
                (
                    av_grid.size,
                    len(self.config["filters"]),
                    *self.config["ssp_model"].L_lambda.shape[:-1],
                )
            ) * u.Quantity("3631e-9 Jy / Msun")

            for j, ssp in enumerate(ssps):
                photo = ssp.compute_photometry(
                    filter_list=self.config["filters"], z_obs=self.config["redshift"]
                ).to("3631e-9 Jy / Msun")
                all_photometry[j] = photo

            self.config["photometry_grid"] = all_photometry
            # Save the photometry grid and the values of Av
            if options.has_value("SavePhotometryGrid"):
                print("Saving photometry grid to ",
                      options["SavePhotometryGrid"])
                with open(options["SavePhotometryGrid"], 'wb') as file:
                    pickle.dump({"photometry_grid": all_photometry,
                                 "av_grid": av_grid},
                                 file, pickle.HIGHEST_PROTOCOL)

    def make_observable(self, block, parse=False):
        sfh_model = self.config["sfh_model"]
        if parse:
            sfh_model.parse_datablock(block)
        av = block["parameters", "a_v"]
        av_idx = np.searchsorted(self.config["av_grid"], av)
        w_idx = (av - self.config["av_grid"][av_idx - 1]) / (
            self.config["av_grid"][av_idx] - self.config["av_grid"][av_idx - 1]
        )
        photometry = self.config["photometry_grid"][av_idx] * w_idx + self.config[
            "photometry_grid"
        ][av_idx - 1] * (1 - w_idx)

        flux_model = sfh_model.model.compute_photometry(
            self.config["ssp_model"], t_obs=sfh_model.today, photometry=photometry
        )
        flux_model = flux_model.to_value("3631e-9 Jy")
        normalization = np.mean(self.config["photometry_flux"] / flux_model)
        block["parameters", "normalization"] = normalization
        return flux_model * normalization

    def execute(self, block):
        valid, penalty = self.config["sfh_model"].parse_datablock(block)
        if not valid:
            print("Invalid")
            block[section_names.likelihoods, "SFHPhotometry_like"] = -1e5 * penalty
            block["parameters", "normalization"] = 0.0
            return 0
        flux_model = self.make_observable(block)
        # Final posterior for sampling
        like = self.log_like(
            self.config["photometry_flux"],
            flux_model,
            self.config["photometry_flux_var"],
        )
        block[section_names.likelihoods, "SFHPhotometry_like"] = like
        return 0


def setup(options):
    options = SectionOptions(options)
    mod = SFHPhotometryModule(options)
    return mod


def execute(block, mod):
    mod.execute(block)
    return 0


def cleanup(mod):
    mod.cleanup()
