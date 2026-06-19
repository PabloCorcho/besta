"""
This module contains the pipeline manager to concatenate multiple modules.
"""

import os
import subprocess
import copy
import numpy as np

from typing import List, Dict

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
        """Execute a shell command and return its process exit code."""
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
            ini_filename = os.path.expandvars(ini_filename)
            if not os.path.isfile(ini_filename):
                raise FileNotFoundError(f"{ini_filename} not found")

        if ini_values_filename is None:
            io.make_values_file(config)
        else:
            ini_values_filename = os.path.expandvars(ini_values_filename)
            if not os.path.isfile(ini_values_filename):
                raise FileNotFoundError(f"{ini_values_filename} not found")
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
                module_name = subpipe_config["pipeline"]["modules"].replace(",", " ").split()[0]
                if module_name not in subpipe_config:
                    raise KeyError(
                        f"Module '{module_name}' not found in subpipeline configuration."
                    )
                # Update the input values
                subpipe_config[module_name].update(
                    (k, v)
                    for k, v in prev_solution.items()
                    if k in subpipe_config[module_name]
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
            logger.info("MAP solution: %s", solution)

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
            # Check for section postprocess
            #TODO
        return 0


# Run multiple independent pipelines in parallel
def _run_main_pipeline_job(job):
    """Worker helper used by multiprocessing to run one MainPipeline."""
    pipeline = MainPipeline(
        job["pipeline_config"],
        n_cores_list=job["n_cores_list"],
        ini_files=job["ini_files"],
        ini_values_files=job["ini_values_files"],
    )
    status = pipeline.execute_all(plot_result=job["plot_result"])
    return {"index": job["index"], "status": status}


class BatchPipeline(object):
    """Manager for running multiple independent MainPipeline instances in parallel.

    Parameters
    ----------
    pipeline_configuration_list : list[list[dict]]
        Each entry is the configuration list required to initialize a MainPipeline.
    n_cores_list : list, optional
        Per-pipeline n_cores lists to pass to each MainPipeline.
    ini_files : list, optional
        Per-pipeline ini_files lists to pass to each MainPipeline.
    ini_values_files : list, optional
        Per-pipeline ini_values_files lists to pass to each MainPipeline.
    n_jobs_parallel : int, optional
        Maximum number of parallel processes.
    """

    def __init__(
        self,
        *,
        pipeline_configuration_list: List[List[Dict]],
        n_cores_list=None,
        ini_files=None,
        ini_values_files=None,
        n_jobs_parallel=None,
    ):
        # Store the list of independent pipeline configurations.
        self.all_pipelines_config = pipeline_configuration_list

        n_pipelines = len(self.all_pipelines_config)

        self.all_n_cores_list = (
            [None] * n_pipelines if n_cores_list is None else n_cores_list
        )
        self.all_ini_files = [None] * n_pipelines if ini_files is None else ini_files
        self.all_ini_values_files = (
            [None] * n_pipelines if ini_values_files is None else ini_values_files
        )
        self.n_jobs_parallel = n_jobs_parallel

        self._validate_inputs()

    def _validate_inputs(self):
        """Validate top-level list lengths."""
        n_pipelines = len(self.all_pipelines_config)
        for attr_name in (
            "all_n_cores_list",
            "all_ini_files",
            "all_ini_values_files",
        ):
            value = getattr(self, attr_name)
            if len(value) != n_pipelines:
                raise ValueError(
                    f"{attr_name} length ({len(value)}) must match "
                    f"number of pipelines ({n_pipelines})"
                )

    def cpu_info(self):
        """Log available CPU cores and number of independent pipelines."""
        n_pipelines = len(self.all_pipelines_config)
        n_cores = os.cpu_count() or 1
        logger.info(f"Available CPU cores: {n_cores}")
        logger.info(f"Number of pipelines to run: {n_pipelines}")

    def build_pipelines(self):
        """Build all MainPipeline objects without executing them."""
        pipelines = []
        for pipeline_config, n_cores, ini_file, ini_values_file in zip(
            self.all_pipelines_config,
            self.all_n_cores_list,
            self.all_ini_files,
            self.all_ini_values_files,
        ):
            pipelines.append(
                MainPipeline(
                    pipeline_config,
                    n_cores_list=n_cores,
                    ini_files=ini_file,
                    ini_values_files=ini_values_file,
                )
            )
        return pipelines

    def _build_jobs(self, plot_result=False):
        jobs = []
        for index, (pipeline_config, n_cores, ini_file, ini_values_file) in enumerate(
            zip(
                self.all_pipelines_config,
                self.all_n_cores_list,
                self.all_ini_files,
                self.all_ini_values_files,
            )
        ):
            jobs.append(
                {
                    "index": index,
                    "pipeline_config": pipeline_config,
                    "n_cores_list": n_cores,
                    "ini_files": ini_file,
                    "ini_values_files": ini_values_file,
                    "plot_result": plot_result,
                }
            )
        return jobs

    def run_single_pipeline(self, index, plot_result=False):
        """Run one independent pipeline by index."""
        jobs = self._build_jobs(plot_result=plot_result)
        if index < 0 or index >= len(jobs):
            raise IndexError(f"Pipeline index {index} out of range")
        return _run_main_pipeline_job(jobs[index])["status"]

    def run_all_pipelines(self, plot_result=False):
        """Run all configured MainPipeline instances in parallel."""
        from multiprocessing import get_context

        self.cpu_info()
        jobs = self._build_jobs(plot_result=plot_result)
        if not jobs:
            logger.info("No pipelines to run")
            return []

        max_jobs = os.cpu_count() or 1
        if self.n_jobs_parallel is None:
            n_jobs = min(len(jobs), max_jobs)
        else:
            n_jobs = max(1, min(self.n_jobs_parallel, len(jobs), max_jobs))

        logger.info(f"Running {n_jobs} pipelines in parallel")

        if n_jobs == 1:
            raw_results = [_run_main_pipeline_job(job) for job in jobs]
        else:
            with get_context("spawn").Pool(processes=n_jobs) as pool:
                raw_results = pool.map(_run_main_pipeline_job, jobs)

        raw_results.sort(key=lambda x: x["index"])
        results = [item["status"] for item in raw_results]

        n_failed = sum(status != 0 for status in results)
        if n_failed > 0:
            logger.error(f"{n_failed}/{len(results)} pipelines failed")
        else:
            logger.info("All pipelines executed successfully")

        return results
    
    @classmethod
    def from_running_parameters(
        cls,
        pipeline_configuration: Dict,
        running_parameters: List[Dict],
        **kwargs,
    ):
        """Factory method to create a BatchPipeline from a single pipeline configuration and multiple running parameters.
        
        Parameters
        ----------
        pipeline_configuration : dict
            Base configuration for the pipeline, which will be updated with each set of running parameters.
        running_parameters : list[dict]
            List of dictionaries containing the parameters to update in the base configuration for each independent pipeline run.
        **kwargs
            Additional keyword arguments to pass to the BatchPipeline constructor (e.g., n_cores_list
            or ini_files). These will be applied to all pipelines created from the running parameters.
        
        Returns
        -------
        BatchPipeline
             An instance of BatchPipeline configured to run multiple pipelines based on the provided configuration and parameters.

        Example
        -------
           >>> from besta.pipeline import BatchPipeline
           >>> base_config = {
           ...     "pipeline": {"modules": "FullSpectralFit", "values": "./values.ini"},
           ...     "output": {"filename": "./fit_result"},
           ...     "FullSpectralFit": {"file": "/path/to/module.py", "redshift": 0.1},
           ... }
           >>> running_parameters = [
           ...     {"FullSpectralFit": {"redshift": 0.10}, "output": {"filename": "./fit_z010"}},
           ...     {"FullSpectralFit": {"redshift": 0.12}, "output": {"filename": "./fit_z012"}},
           ... ]
           >>> batch = BatchPipeline.from_running_parameters(
           ...     pipeline_configuration=base_config,
           ...     running_parameters=running_parameters,
           ...     n_jobs_parallel=2,
           ... )
           >>> results = batch.run_all_pipelines()
           >>> len(results)
           2
        """
        if not isinstance(running_parameters, list):
            raise TypeError("running_parameters must be a list of dictionaries")

        pipeline_config_list = []

        # Recursively merge nested dictionaries from one parameter set.
        def recursive_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = recursive_update(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        for params in running_parameters:
            if not isinstance(params, dict):
                raise TypeError("Each running parameter entry must be a dictionary")
            config_copy = copy.deepcopy(pipeline_configuration)
            config_copy = recursive_update(config_copy, params)
            pipeline_config_list.append([config_copy])  # Wrap in list for MainPipeline
        return cls(pipeline_configuration_list=pipeline_config_list, **kwargs)