.. _pipeline_manager:

Pipeline Manager
================

BESTA provides a small orchestration layer to chain several pipeline modules and run them through CosmoSIS. The key entrypoint is :class:`besta.pipeline.MainPipeline`, which:

- builds or reuses CosmoSIS ``.ini`` and ``values`` files for each sub-pipeline,
- executes CosmoSIS (optionally with MPI) and collects the output,
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
           "extra_output": "parameters/stellar_mass",
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

Short checklist
---------------

- Provide absolute or relative paths for input spectra/photometry and masks.
- Ensure the ``values`` file lists all sampled parameters; constants are back-filled when creating ``DataBlock`` objects for plotting.
- Use the same ``velscale`` when preparing observed spectra and SSP templates.
- If you enable SFH transforms (``use_transforms``), remember that priors in the ``values`` file must be on the *latent* variables (see :ref:`configuration` for details).



