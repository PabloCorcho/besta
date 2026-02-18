.. _grid_inference:

Grid-Based Inference
====================

BESTA includes a direct **model-grid inference** workflow that does not require
CosmoSIS samplers. This mode is useful when you already have a finite model
library and want fast posterior evaluation over that library.

Why use the grid module instead of CosmoSIS sampling?
-----------------------------------------------------

Use the grid module when:

- Your model space is already discretized (for example: precomputed SED/SSP libraries).
- You want robust, repeatable inference with no MCMC tuning.
- You need high throughput over many objects using candidate selection and parallel workers.
- You want direct control over priors/likelihoods in Python.

Prefer the CosmoSIS pipeline (see :ref:`pipeline_manager`) when:

- Your parameter space is continuous and not naturally represented by a fixed grid.
- You need sampler diagnostics and chain-level convergence checks.
- You rely on existing module wiring in the CosmoSIS runtime.


Core Concepts
-------------

- :class:`besta.grid.grid.ModelGrid`: container for model observables/targets and metadata.
- :class:`besta.grid.grid.GridFitter`: computes posterior weights over grid models.
- :mod:`besta.grid.prob`: prior and likelihood building blocks.
- :mod:`besta.grid.binning`: candidate selectors to avoid evaluating the full grid for every object.


Minimal Workflow
----------------

1. Build or load a :class:`~besta.grid.grid.ModelGrid`.
2. Create a :class:`~besta.grid.grid.GridFitter` with a likelihood and prior.
3. Evaluate posterior summaries for one object, or run :meth:`~besta.grid.grid.GridFitter.fit_batch` for many.

Example (single-object posterior on one target):

.. code-block:: python

   import numpy as np
   from besta.grid import ModelGrid, GridFitter
   from besta.grid.prob import GaussianProductLikelihood, FlatPrior

   # N models, P observables, Q targets
   grid = ModelGrid(
       observables=obs_models,                # shape (N, P)
       targets=target_models,                 # shape (N, Q)
       observable_names=["mag_g", "mag_r", "mag_i"],
       target_names=["logM", "age", "Z", "z"],
   )

   fitter = GridFitter(
       grid=grid,
       likelihood=GaussianProductLikelihood(),
       prior=FlatPrior(),
       use_standardised=True,
   )

   x = np.array([22.1, 21.5, 21.2])          # observed data (P,)
   sx = np.array([0.03, 0.03, 0.04])         # observational errors (P,)
   bins = np.linspace(7.0, 12.0, 101)        # bins for logM

   post_logM, centers = fitter.posterior_over_target(
       x_native=x,
       sigma_native=sx,
       target_col="logM",
       bins=bins,
   )


Batch Inference (many objects)
------------------------------

For catalogs, use :meth:`~besta.grid.grid.GridFitter.fit_batch`.
It supports:

- optional candidate selection (`binner=`),
- thread/process parallelism (`n_jobs`, `backend`),
- posterior truncation controls (`posterior_keep_mass`, etc.),
- optional per-target summary statistics (`stats_for`, `stats_bins`),
- optional HDF5 output (`output_hdf5_path`).

.. code-block:: python

   from besta.grid.binning import KDTreeBinner

   binner = KDTreeBinner(dims=[0, 1, 2]).fit(grid)  # select candidates in observable space

   results = fitter.fit_batch(
       X_native=X_catalog,                    # shape (M, P)
       SIG_native=SIG_catalog,                # shape (M, P)
       binner=binner,
       n_jobs=8,
       backend="thread",
       stats_for=["logM", "age"],
       stats_bins=[np.linspace(7, 12, 120), np.linspace(0, 14, 120)],
       output_hdf5_path="grid_fit_results.h5",
       output_hdf5_group="/run1",
       return_mode="iter",
   )

   for r in results:
       # r contains: m, candidates, post_models, truncation, and optional stats
       pass


Priors and Likelihoods
----------------------

The grid module is fully Bayesian; you choose the ingredients from
:mod:`besta.grid.prob`:

- priors: :class:`~besta.grid.prob.FlatPrior`, :class:`~besta.grid.prob.CompositePrior`,
  :class:`~besta.grid.prob.ObservableDependentPrior`, and others,
- likelihoods: :class:`~besta.grid.prob.GaussianProductLikelihood`,
  :class:`~besta.grid.prob.CompositeLikelihood`, etc.

This gives a similar statistical structure to sampling methods, but evaluated
directly on a finite model set instead of drawing chains.


Input/Output and Reproducibility
--------------------------------

:class:`~besta.grid.grid.ModelGrid` supports multiple formats:

- FITS tables: :meth:`~besta.grid.grid.ModelGrid.from_fits_table`,
  ``ModelGrid.to_fits_table(...)``
- HDF5: :meth:`~besta.grid.grid.ModelGrid.from_hdf5`,
  :meth:`~besta.grid.grid.ModelGrid.to_hdf5`
- Pickle: :meth:`~besta.grid.grid.ModelGrid.from_pickle`,
  :meth:`~besta.grid.grid.ModelGrid.to_pickle`
- automatic loader: :meth:`~besta.grid.grid.ModelGrid.load_auto`


Practical Tips
--------------

- Start with `FlatPrior + GaussianProductLikelihood` as a baseline.
- If the grid is large, use a binner to cut candidate counts before posterior evaluation.
- Use `return_mode="iter"` for low-memory streaming over large catalogs.
- Write batch outputs to HDF5 for reproducible downstream post-processing.
