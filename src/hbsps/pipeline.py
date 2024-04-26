"""
This module contains the tools to build a pipeline
"""

import importlib.util
import sys
import os
import argparse
import subprocess
import multiprocessing
import configparser
import numpy as np

from matplotlib import pyplot as plt

import hbsps.output as output
from hbsps import prepare_spectra
from hbsps import kinematics, sfh
from hbsps.dust_extinction import deredden_spectra, redden_ssp


class MainPipeline(object):
    def __init__(self, pipeline_configuration_list, n_cores_list=None):
        """Main HBSPS Pipeline.

        Parameters
        ----------
        - pipeline_configuration_list: list
            List of dictionaries containing the configuration parameters for each
            subpipeplie.
        - n_cores_list: list
            List containing the number of cores is on each run.
        """
        self.pipelines_config = pipeline_configuration_list
        if n_cores_list is None:
            self.n_cores_list = [1] * len(pipeline_configuration_list)
        else:
            self.n_cores_list = n_cores_list

    def run_command(self, command):
        print(f"Running command >> {command} <<")
        return subprocess.call(command, shell=True)

    def get_cosmosis_result(self, ini_file):
        return output.Reader(ini_file)

    def execute_pipeline(self, config, n_cores):
        """Execute a pipeline"""
        ini_filename = config["pipeline"]["modules"].replace(" ", "_") + "_auto.ini"
        output.make_ini_file(ini_filename, config)
        output.make_values_file(config)

        if n_cores > 1:
            command = f"mpiexec -n {n_cores} cosmosis --mpi {ini_filename}"
        else:
            command = f"cosmosis {ini_filename}"
        return_code = self.run_command(command)
        # return_code = 0
        if return_code == 0:
            return ini_filename
        else:
            print("Unsuccessful run, return code: ", return_code)
            return None

    def execute_all(self):
        print("Executing all pipelines")
        prev_solution = None
        for pipeline_config, n_cores in zip(self.pipelines_config, self.n_cores_list):
            if prev_solution is not None:
                print("Updating configuration file with previus run results")
                # Update the initial values
                # print(pipeline_config["Values"])
                # pipeline_config["Values"].update(
                #     (k, v)
                #     for k, v in prev_solution.items()
                #     if k in pipeline_config["Values"]
                # )
                # Update the input values
                pipeline_config[pipeline_config["pipeline"]["modules"]].update(
                    (k, v)
                    for k, v in prev_solution.items()
                    if k in pipeline_config[pipeline_config["pipeline"]["modules"]]
                )
            # Execute subpipepline
            ini_filename = self.execute_pipeline(pipeline_config, n_cores)
            print("Extracting results from the run")
            reader = self.get_cosmosis_result(ini_filename)
            reader.load_observation()
            reader.load_ssp_model()
            reader.load_extinction_model()
            # TODO: remove somehow the if conditional
            if "KinDust" in reader.last_module:
                reader.load_chain(include_ssp_weights=True)
            else:
                reader.load_chain()
            solution = reader.get_maxlike_solution()
            prev_solution = solution.copy()
            print("MaxLike solution: ", solution)
            # Reconstruct best fit
            if "los_sigma" in solution:
                print("Convolvind SED with kinematic solution")
                sed, mask = kinematics.convolve_ssp(
                    reader.config, solution["los_sigma"], solution["los_vel"]
                )
            elif "los_sigma" in pipeline_config[pipeline_config["pipeline"]["modules"]]:
                print("Convolvind SED with kinematic input")
                los_sigma = pipeline_config[pipeline_config["pipeline"]["modules"]]['los_sigma']
                los_vel = pipeline_config[pipeline_config["pipeline"]["modules"]]['los_vel']
                sed, mask = kinematics.convolve_ssp(
                    reader.config, los_sigma=los_sigma, los_vel=los_vel
                )
            reader.config["ssp_sed"] = sed
            reader.config["ssp_wl"] = reader.config["wavelength"]
            reader.config["mask"] = mask

            # Dust extinction
            if "av" in solution:
                print("Reddening SED with dust solution")
                redden_ssp(reader.config, solution["av"])
            elif "av" in pipeline_config[pipeline_config["pipeline"]["modules"]]:
                print("Reddening SED with dust input")
                av = pipeline_config[pipeline_config["pipeline"]["modules"]]['av']
                redden_ssp(reader.config, av)

            flux_model = sfh.composite_stellar_population(
                reader.config, solution)
            if reader.last_module == "SFH":
                self.plot_fit(pipeline_config, reader.config, flux_model,
                              solution=solution)
            else:
                self.plot_fit(pipeline_config, reader.config, flux_model)


    def plot_fit(self, pipe_config, config, flux_model, solution=None):
        """Plot the fit."""
        fig, axs = plt.subplots(ncols=1, nrows=2, sharex=True, constrained_layout=True)
        plt.suptitle(f"Module: {pipe_config['pipeline']['modules']}")
        ax = axs[0]
        # Plot input spectra
        ax.fill_between(
            config["wavelength"],
            config["flux"] - config["cov"] ** 0.5,
            config["flux"] + config["cov"] ** 0.5,
            color="k",
            alpha=0.5,
        )
        ax.plot(config["wavelength"], config["flux"], c="k", label="Observed")
        # Show masked pixels
        ax.plot(
            config["wavelength"][~config["mask"]],
            config["flux"][~config["mask"]],
            c="b",
            marker="x",
            lw=0,
            label="Masked",
        )
        # Plot model
        ax.plot(config["wavelength"], flux_model, c="r", label="Model")
        # Plot residuals
        ax.plot(
            config["wavelength"],
            flux_model - config["flux"],
            c="lime",
            label="Residuals",
        )
        ax.axhline(0, ls="--", color="k", alpha=0.2)
        ax.set_ylabel("Flux")
        ax.legend()

        chi2 = (flux_model - config["flux"]) ** 2 / config["cov"]
        ax = axs[1]
        ax.plot(config["wavelength"], chi2, c="k", lw=0.7)
        ax.grid(visible=True)
        ax.set_ylabel(r"$\chi^2$")
        ax.set_yscale("symlog", linthresh=0.1)
        ax.set_xlabel("Wavelenth (AA)")
        inax = ax.inset_axes((1.05, 0, 0.3, 1), sharey=ax)
        inax.hist(
            chi2,
            bins=np.geomspace(0.01, 100),
            orientation="horizontal",
            color="k",
            histtype="step",
        )
        inax.set_xlabel("No. pixels")
        inax.grid(visible=True)
        inax.tick_params(labelleft=False)

        if solution is not None:
            sol_text = "Solution\n"
            for k, v in solution.items():
                if "ssp" not in k:
                    #sol_text += f"{k}={v:.3f}\n"
                    continue
                else:
                    sol_text += f"{k}={v:.1f} "
            ax.annotate(sol_text, xy=(.95, .95), xycoords='axes fraction',
                        va='top', ha='right', fontsize=7, color='Grey')

            sfh_result = sfh.reconstruct_sfh(config, solution)
            inax = axs[0].inset_axes((1.05, 0, 0.3, 1))
            mappable = inax.pcolormesh(sfh_result['metals'], sfh_result['ages'],
                            np.log10(sfh_result['ssp_mass_formed'] + 1),
                            vmin=4)
            inax.set_xscale('log')
            inax.set_yscale('log')
            plt.colorbar(mappable, ax=inax)
            inax = inax.inset_axes((0, 1.05, 1., .3))
            inax.plot(sfh_result['ages'][0, :],
                      np.log10(sfh_result['mass_formed_history'] + 1))
            inax.set_ylim(3, 11)

        fig.savefig(os.path.join(os.path.dirname(pipe_config['output']['filename']),
                    f"{pipe_config['pipeline']['modules']}_best_fit_spectra.png"),
                    bbox_inches='tight', dpi=200)
        #plt.show()

