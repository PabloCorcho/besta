Getting started
===============

This page provides a short roadmap for new users before diving into the more
detailed reference sections.

Recommended reading order
^^^^^^^^^^^^^^^^^^^^^^^^^

1. :ref:`installation` to create a working environment.
2. :ref:`configuration` to understand the runtime options and values files.
3. :ref:`pipeline_manager` if you plan to run CosmoSIS-based fits.
4. :ref:`postprocessing` to inspect results after a run completes.
5. :ref:`grid_inference` if your use case is better served by precomputed model grids.

Two main workflows
^^^^^^^^^^^^^^^^^^

BESTA currently supports two complementary analysis modes:

- **Sampler-based pipeline fitting** with :class:`besta.pipeline.MainPipeline`.
	Use this when you need flexible forward modelling and continuous parameter
	inference, especially for spectroscopy or joint fits with nuisance parameters.
- **Grid-based inference** with :class:`besta.grid.grid.GridFitter`.
	Use this when you already have a finite model library and want high-throughput,
	repeatable inference over many objects.

Minimal pipeline example
^^^^^^^^^^^^^^^^^^^^^^^^

The smallest CosmoSIS-based run usually contains:

- a ``runtime`` section selecting the samplers,
- an ``output`` section selecting the results filename,
- a ``pipeline`` section pointing to the active module and values file, and
- one module-specific section such as ``FullSpectralFit``.

The complete structure is documented in :ref:`configuration`, and a full
orchestration example is given in :ref:`pipeline_manager`.

Tutorial material
^^^^^^^^^^^^^^^^^

The repository includes worked examples under ``tutorials/``. In particular:

- ``tutorials/fit_sdss_spectra`` demonstrates an end-to-end spectral fit,
- ``tutorials/fit_redshift`` contains redshift-oriented examples.

These tutorials complement the API reference by showing how the values file,
input spectra, LSF information, and post-processing tools fit together in a
single workflow.

Where to go next
^^^^^^^^^^^^^^^^

- For available fitting modules, see :ref:`pipeline_modules`.
- For API-level details, see :ref:`api`.
- For contributing fixes or documentation improvements, see the contributing page.

