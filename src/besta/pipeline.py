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
from besta.logging import get_logger, setup_logging

logger = get_logger(__name__)

class MainPipeline(object):
    """BESTA Pipeline manager.

    Attributes
    ----------
    pipelines_config : list
        List of dictionaries containing the configuration parameters for each
            subpipeline.
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
        self._parse_logger(pipeline_configuration_list)
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

    def _parse_logger(self, pipeline_configuration_list):
        for config in pipeline_configuration_list:
            # select the first module in the pipeline to configure logging (if any)
            module = config["pipeline"]["modules"].replace(",", " ").split(" ")[0]
            logging_console = config[module].get("logging_console", False)
            logging_level = config[module].get("logging_level", "INFO").upper()
            logging_overwrite = config[module].get("logging_overwrite", False)
            logging_file = config[module].get("logging_file", None)

            setup_logging(level=logging_level, log_file=logging_file,
                          overwrite=logging_overwrite, console=logging_console)
            break

    def run_command(self, command):
        logger.info(f"Running command >> {command} <<")
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
            Number of cores to use during runtime.
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
            assert os.path.isfile(os.path.expandvars(ini_filename)
                                  ), f"{os.path.expandvars(ini_filename)} not found"

        if ini_values_filename is None:
            io.make_values_file(config)
        else:
            assert os.path.isfile(
                ini_values_filename
            ), f"{ini_values_filename} not found"
            config["pipeline"]["values"] = ini_values_filename

        if n_cores == -1:
            n_cores = os.cpu_count()
            logger.info(f"Using all available cores: {n_cores}")

        if n_cores > 1:
            command = f"mpiexec -n {n_cores} cosmosis --mpi {ini_filename}"
        else:
            command = f"cosmosis {ini_filename}"
        return_code = self.run_command(command)
        if return_code == 0:
            logger.info("Successful run, return code: %s", return_code)
            return ini_filename
        else:
            logger.error("Unsuccessful run, return code: %s", return_code)
            return None

    def execute_all(self, plot_result=False):
        """Execute all sub-pipelines."""
        logger.info("Executing all pipelines")
        prev_solution = None
        for subpipe_config, n_cores, ini_filename, ini_values_filename in zip(
            self.pipelines_config,
            self.n_cores_list,
            self.ini_files,
            self.ini_values_files,
        ):
            if prev_solution is not None:
                logger.info("Updating configuration file with previous run results")
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

            if ini_filename is None:
                logger.error("Pipeline execution failed, stopping.")
                return 1

            # Extract best solution
            logger.info("Extracting results from the run")
            reader = io.Reader(ini_filename)
            reader.load_results()
            solution = reader.get_maxlike_solution()
            prev_solution = solution.copy()
            logger.info("MaxLike solution: %s", solution)

            if plot_result:
                solution_datablock = reader.solution_to_datablock(
                    prev_solution)
                    
                # Initialise the module to reconstruct the solution
                for par_module in reader.modules:
                    logger.info("Plotting results for module: %s", par_module)
                    pipeline_module = reader.get_module(par_module)
                    figname = subpipe_config["output"].get(
                        "figurename",
                        subpipe_config["output"]["filename"].replace(".txt", "")
                        + f"_{par_module}_best_fit_solution.png",
                    )
                    pipeline_module.plot_solution(solution_datablock,
                                             figname=figname)
        return 0