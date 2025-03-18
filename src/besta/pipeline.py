"""
This module contains the pipeline manager to concatenate multiple modules.
"""

import os
import subprocess
import numpy as np

from matplotlib import pyplot as plt

from cosmosis import DataBlock

from besta import io
from besta import pipeline_modules


class MainPipeline(object):
    """BESTA Pipeline manager.

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

    def __init__(
        self,
        pipeline_configuration_list,
        n_cores_list=None,
        ini_files=None,
        ini_values_files=None,
    ):
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
            self.ini_values_files = [ini_values_files] * len(
                pipeline_configuration_list
            )
        else:
            self.ini_values_files = ini_values_files

    def run_command(self, command):
        print(f"Running command >> {command} <<")
        return subprocess.call(command, shell=True)

    def execute_pipeline(
        self, config, n_cores, ini_filename=None, ini_values_filename=None
    ):
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
                config["pipeline"]["modules"].replace(" ", "_") + "_auto.ini",
            )
            io.make_ini_file(ini_filename, config)
        else:
            assert os.path.isfile(ini_filename), f"{ini_filename} not found"

        if ini_values_filename is None:
            io.make_values_file(config)
        else:
            assert os.path.isfile(
                ini_values_filename
            ), f"{ini_values_filename} not found"
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
            self.pipelines_config,
            self.n_cores_list,
            self.ini_files,
            self.ini_values_files,
        ):
            if prev_solution is not None:
                print("Updating configuration file with previus run results")
                # Update the input values
                subpipe_config[subpipe_config["pipeline"]["modules"]].update(
                    (k, v)
                    for k, v in prev_solution.items()
                    if k in subpipe_config[subpipe_config["pipeline"]["modules"]]
                )
            # Execute sub-pipepline
            ini_filename = self.execute_pipeline(
                subpipe_config,
                n_cores,
                ini_filename=ini_filename,
                ini_values_filename=ini_values_filename,
            )
            # Extract best solution
            print("Extracting results from the run")
            reader = io.Reader(ini_filename)
            reader.load_results()
            solution = reader.get_maxlike_solution()
            prev_solution = solution.copy()
            print("MaxLike solution: ", solution)

            if plot_result:
                # Initialise the module to reconstruct the solution
                pipeline_module = reader.last_module
                solution_datablock = reader.solution_to_datablock(prev_solution)
                self.plot_fit(
                    pipeline_module, solution_datablock, pipe_config=subpipe_config
                )

    def plot_fit(self, module, solution: DataBlock, pipe_config):
        """Plot the fit."""
        flux_model = module.make_observable(solution, parse=True)
        if isinstance(flux_model, tuple):
            weights = flux_model[1]
            flux_model = flux_model[0]
        else:
            weights = np.ones_like(flux_model)

        fig, axs = plt.subplots(ncols=1, nrows=2, sharex=True, constrained_layout=True,
                                figsize=(np.round(flux_model.size / 300, 0), 6))
        plt.suptitle(f"Module: {module.name}")
        ax = axs[0]
        # Plot input spectra
        ax.fill_between(
            module.config["wavelength"].value,
            module.config["flux"] - module.config["cov"] ** 0.5,
            module.config["flux"] + module.config["cov"] ** 0.5,
            color="k",
            alpha=0.5,
        )
        ax.plot(
            module.config["wavelength"], module.config["flux"], c="k", label="Observed",
            lw=0.7)
        # Show masked pixels
        nan_mask = np.ones_like(module.config["flux"])
        nan_mask[(weights * module.config["weights"]) == 0] = np.nan
        ax.plot(
            module.config["wavelength"],
            module.config["flux"] * nan_mask,
            c="r",
            lw=0.7,
            label="Masked",
        )
        # Plot model
        ax.plot(module.config["wavelength"], flux_model, c="b", label="Model",
                lw=0.7)
        # Plot residuals
        residuals = flux_model - module.config["flux"]
        ax.plot(
            module.config["wavelength"],
            residuals,
            c="orange",
            label="Residuals",
            lw=0.7
        )
        ax.axhline(0, ls="--", color="k", alpha=0.2)
        ax.set_ylabel("Flux")
        ax.legend()

        p5, p95 = np.nanpercentile(module.config["flux"], [5, 95])
        p_residuals = np.nanpercentile(residuals, 5) * 0.95
        ax.set_ylim(np.min([p_residuals, p5 * 0.8]), p95 * 1.2)

        chi2 = (flux_model - module.config["flux"]) ** 2 / module.config["cov"]
        ax = axs[1]
        ax.plot(module.config["wavelength"], chi2, c="k", lw=0.7)
        ax.grid(visible=True)
        ax.set_ylabel(r"$\chi^2$")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("Wavelenth (AA)")
        inax = ax.inset_axes((1.05, 0, 0.3, 1), sharey=ax)
        inax.hist(
            chi2,
            bins=np.geomspace(0.01, 100),
            orientation="horizontal",
            color="k",
            histtype="step"
        )
        inax.annotate(f"Median: {np.nanmedian(chi2):.1f}",
                      xy=(0.05, 0.95), xycoords="axes fraction", va="top")

        inax.set_xlabel("No. pixels")
        inax.grid(visible=True)
        inax.tick_params(labelleft=False)

        figname = os.path.basename(
            pipe_config["output"].get("figurename",
                                      pipe_config["output"]["filename"]))
        output_file = os.path.join(
            os.path.dirname(pipe_config["output"]["filename"]),
            f"{pipe_config['pipeline']['modules']}_{figname}_best_fit_spectra.png",
            )

        fig.savefig(
            output_file,
            bbox_inches="tight",
        )
        print(f"Fit plot saved at: {output_file}")
        plt.close()
