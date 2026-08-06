.. _pipeline_manager:

Pipeline Manager
================

BESTA provides a small orchestration layer to chain several pipeline modules and run them through CosmoSIS. The key entrypoint is :class:`besta.pipeline.MainPipeline`, which:

- builds or reuses CosmoSIS ``.ini`` and ``values`` files for each sub-pipeline,
- executes CosmoSIS (optionally with MPI for parallelisation) and collects the output,
- propagates best-fit parameters between stages, and
- can plot best-fit spectra/photometry for each module.

Typical usage
-------------

.. code-block:: python

   from besta.pipeline import MainPipeline
   from besta.pipeline_modules.full_spectral_fit import FullSpectralFitModule

   configuration = {
       "runtime": {"sampler": "maxlike emcee"},
       "maxlike": {"method": "Nelder-Mead", "tolerance": 1e-3, "maxiter": 3000},
       "emcee": {"walkers": 32, "samples": 100, "nsteps": 100},
       "output": {"filename": "./full_fit_example", "format": "text"},
       "pipeline": {
           "modules": "FullSpectralFit",
           "values": "./values.ini",
           "likelihoods": "FullSpectralFit",
           "quiet": "F",
           "timing": "T",
           "debug": "T",
           "extra_output": "extra/stellar_mass",
       },
       "FullSpectralFit": {
           "file": FullSpectralFitModule.get_path(),
           "inputSpectrum": "./my_spectrum.dat",
           "SSPModel": "PopStar",
           "SSPModelArgs": "cha",
           "SSPDir": "None",
           "wlUnits": "Angstrom",
           "fluxUnits": "1e-16 erg / (s cm2 Angstrom)",
           "wlRange": [3500.0, 9000.0],
           "SFHModel": "ExponentialSFH",
           "velscale": 50.0,
           "ExtinctionLaw": "ccm89",
       },
   }

   pipeline = MainPipeline([configuration], n_cores_list=[1])
   pipeline.execute_all(plot_result=True)

Configuration files
-------------------

``MainPipeline`` will auto-generate:

- a CosmoSIS ``.ini`` file per configuration (unless you provide one), and
- a ``values`` file from the ``pipeline["values"]`` path if it does not exist.

Both files are written next to the configured output unless paths are absolute. Environment variables in paths are expanded. If CosmoSIS exits with a non-zero code, ``execute_all`` stops and returns ``1``.

Best-fit propagation
--------------------

When running multiple sub-pipelines, the maximum-likelihood solution of each run is propagated to the next configuration: parameters that appear in the next module’s section are updated with the previous best-fit values. This enables staged fitting (e.g., kinematics → SFH → photometry).

Plotting results
----------------

If ``plot_result=True``, the best-fit spectra/photometry are plotted for each module using :meth:`besta.pipeline_modules.base_module.BaseModule.plot_solution` and saved alongside the output text files. Each module is re-instantiated from the ``.ini`` file to rebuild the model before plotting.

Running Independent Pipelines In Parallel
-----------------------------------------

For independent runs (for example, fitting many galaxies with the same workflow or using IFU observations), use :class:`besta.pipeline.BatchPipeline`.

Each element of ``pipeline_configuration_list`` is one full ``MainPipeline`` input (i.e., a list of sub-pipeline configuration dictionaries):

.. code-block:: python

   from besta.pipeline import BatchPipeline

   # Two independent jobs, each one contains a single-stage MainPipeline.
   job_a = [config_a]
   job_b = [config_b]

   batch = BatchPipeline(
       pipeline_configuration_list=[job_a, job_b],
       n_jobs_parallel=2,
   )

   # Returns one status code per independent job (0 means success).
   results = batch.run_all_pipelines(plot_result=False)

You can also instantiate the internal ``MainPipeline`` objects without running:

.. code-block:: python

   pipelines = batch.build_pipelines()

Parameter sweeps with ``from_running_parameters``
-------------------------------------------------

A very common use case is when all runs share a base configuration. It is possible to create a batch from parameter updates:

.. code-block:: python

   from besta.pipeline import BatchPipeline

   base_config = {
       # full BESTA configuration dict
   }

   # Parameters that differ between runs.
   running_parameters = [
       {"FullSpectralFit": {"redshift": 0.10}},
       {"FullSpectralFit": {"redshift": 0.12}},
       {"FullSpectralFit": {"redshift": 0.14}},
   ]

   batch = BatchPipeline.from_running_parameters(
       pipeline_configuration=base_config,
       running_parameters=running_parameters,
       n_jobs_parallel=3,
   )

   results = batch.run_all_pipelines()

Notes:

- ``running_parameters`` must be a list of dictionaries.
- Each parameter set is deep-copied from the base configuration, so updates from one run do not leak into the others.
- ``BatchPipeline`` returns statuses in the same order as the input jobs.

Short checklist
---------------

- Provide absolute or relative paths for input spectra/photometry and masks.
- Ensure the ``values`` file lists all sampled parameters; constants are back-filled when creating ``DataBlock`` objects for plotting.
- Use the same ``velscale`` when preparing observed spectra and SSP templates.
- If you enable SFH transforms (``use_transforms``), remember that priors in the ``values`` file must be on the *latent* variables (see :ref:`configuration` for details).



