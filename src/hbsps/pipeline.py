"""
This module contains the pipeline manager to concatenate multiple modules.
"""

import os
import subprocess
import numpy as np
from scipy.optimize import nnls

from matplotlib import pyplot as plt

from hbsps import output
from hbsps import kinematics, sfh
from hbsps.dust_extinction import deredden_spectra, redden_ssp_model


class MainPipeline(object):
    """PST-HBSPS Pipeline manager.
    
    Attributes
    ----------
    pipelines_config : list
        List of dictionaries containing the configuration parameters for each
            subpipeplie.
    n_cores_list : list, optional, default=None
        List containing the number of cores to be used on each run. If None,
        every subpipeline will use one single core during runtime.
    ini_files : list
        List of .ini filenames
    ini_values_files : list
        List of files containing the priors associated to ``ini_files``.
    """
    def __init__(self, pipeline_configuration_list, n_cores_list=None,
                 ini_files=None, ini_values_files=None):
        self.pipelines_config = pipeline_configuration_list

        if n_cores_list is None:
            self.n_cores_list = [1] * len(pipeline_configuration_list)
        else:
            self.n_cores_list = n_cores_list

        if ini_files is None:
            self.ini_files = [ini_files] * len(pipeline_configuration_list)
        else:
            self.ini_files = ini_files

        if ini_values_files is None:
            self.ini_values_files = [ini_values_files] * len(pipeline_configuration_list)
        else:
            self.ini_values_files = ini_values_files

    def run_command(self, command):
        print(f"Running command >> {command} <<")
        return subprocess.call(command, shell=True)

    def execute_pipeline(self, config, n_cores, ini_filename=None,
                         ini_values_filename=None):
        """Execute a sub-pipeline.
        
        Parameters
        ----------
        config : dict
            Dictionary containing the configuration parameters for setting up
            the subpipeline.
        n_cores : int
            Number of cores to used during runtime.
        ini_filename : str, optional, default=None
            If provided, this file is used to run cosmosis.
        ini_values_filename : str, optional, default=None
            If provided, use this file to set the prior values.
        """
        if ini_filename is None:
            ini_filename = os.path.join(
                os.path.dirname(config["output"]["filename"]),
                config["pipeline"]["modules"].replace(" ", "_") + "_auto.ini")
            output.make_ini_file(ini_filename, config)
        else:
            assert os.path.isfile(ini_filename), f"{ini_filename} not found"

        if ini_values_filename is None:
            output.make_values_file(config)
        else:
            assert os.path.isfile(ini_values_filename), f"{ini_values_filename} not found"
            config["pipeline"]["values"] = ini_values_filename

        if n_cores > 1:
            command = f"mpiexec -n {n_cores} cosmosis --mpi {ini_filename}"
        else:
            command = f"cosmosis {ini_filename}"
        return_code = self.run_command(command)
        if return_code == 0:
            print("Successful run, return code: ", return_code)
            return ini_filename
        else:
            print("Unsuccessful run, return code: ", return_code)
            return None

    def execute_all(self, plot_result=False):
        """Execute all sub-pipelines."""
        print("Executing all pipelines")
        prev_solution = None
        for subpipe_config, n_cores, ini_filename, ini_values_filename in zip(
            self.pipelines_config, self.n_cores_list,
            self.ini_files, self.ini_values_files):

            if prev_solution is not None:
                print("Updating configuration file with previus run results")
                # Update the input values
                subpipe_config[subpipe_config["pipeline"]["modules"]].update(
                    (k, v)
                    for k, v in prev_solution.items()
                    if k in subpipe_config[subpipe_config["pipeline"]["modules"]]
                )
            # Execute sub-pipepline
            ini_filename = self.execute_pipeline(subpipe_config, n_cores,
                                                 ini_filename=ini_filename,
                                                 ini_values_filename=ini_values_filename)
            # Extract best solution
            print("Extracting results from the run")
            reader = output.Reader(ini_filename)
            reader.load_results()
            solution = reader.get_maxlike_solution()
            prev_solution = solution.copy()
            print("MaxLike solution: ", solution)

            if plot_result:
                if "SFH" in reader.last_module:
                    reader.load_sfh_model()
                elif "KinDust" in reader.last_module:
                    reader.load_chain(include_ssp_extra_output=True)
                reader.load_observation()
                reader.load_extinction_model()
                reader.load_ssp_model()
                # Reconstruct best fit
                # TODO: this should be directly done by the fitting modules
                flux_model, sol_config = self.reconstruct_solution(
                    subpipe_config, reader.config.copy(), solution)

                if "SFH" in reader.last_module:
                    self.plot_fit(subpipe_config, sol_config, flux_model,
                                solution=solution)
                else:
                    self.plot_fit(subpipe_config, sol_config, flux_model)


    def reconstruct_solution(self, pipeline_config, config, solution):
        """Reconstruct a fit solution"""
        if "los_sigma" in solution:
            print("Convolvind SED with kinematic solution")
            sed, mask = kinematics.convolve_ssp(
                config, solution["los_sigma"], solution["los_vel"]
            )
        elif "los_sigma" in pipeline_config[pipeline_config["pipeline"]["modules"]]:
            print("Convolvind SED with kinematic input")
            los_sigma = pipeline_config[pipeline_config["pipeline"]["modules"]]['los_sigma']
            los_vel = pipeline_config[pipeline_config["pipeline"]["modules"]]['los_vel']
            sed, mask = kinematics.convolve_ssp(
                config, los_sigma=los_sigma, los_vel=los_vel
            )
        config["ssp_sed"] = sed
        config["ssp_wl"] = config["wavelength"]
        config["mask"] = mask

        # Dust extinction
        if "av" in solution:
            print("Reddening SED with dust solution")
            redden_ssp_model(config, solution["av"])
        elif "av" in pipeline_config[pipeline_config["pipeline"]["modules"]]:
            print("Reddening SED with dust input")
            av = pipeline_config[pipeline_config["pipeline"]["modules"]]['av']
            redden_ssp_model(config, av)

        if 'SFH' in pipeline_config['pipeline']['modules']:
            sfh_model = config['sfh_model']
            mask = config['mask']
            valid = sfh_model.parse_free_params(solution)
            if not valid:
                print("ERROR PARSING PARAMETERS")
                return
            flux_model = sfh_model.model.compute_SED(config['ssp_model'],
										     t_obs=sfh_model.today,
											 allow_negative=False).value
            flux_model *= np.mean(config['flux'][mask] / flux_model[mask])
        else:
            solution, rnorm = nnls(config["ssp_sed"].T, config['flux'], maxiter=sed.shape[0] * 10)
            flux_model = np.sum(config["ssp_sed"] * solution[:, np.newaxis], axis=0)
        return flux_model, config

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
            # Include the solution
            sol_text = "Solution\n"
            for k, v in solution.items():
                if "ssp" in k:
                    #sol_text += f"{k}={v:.3f}\n"
                    continue
                else:
                    sol_text += f"{k}={v:.3f}\n"
            ax.annotate(sol_text, xy=(.95, .95), xycoords='axes fraction',
                        va='top', ha='right', fontsize=7, color='Grey')

        fig.savefig(os.path.join(os.path.dirname(pipe_config['output']['filename']),
                    f"{pipe_config['pipeline']['modules']}_best_fit_spectra.png"),
                    bbox_inches='tight', dpi=200)
        #plt.show()

