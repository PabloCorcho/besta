.. _grid_inference:

Grid-Based Inference
====================

BESTA includes a direct **model-grid inference** workflow that does not require
CosmoSIS samplers. This mode is particularly useful when you already have a finite model
library and want fast posterior evaluation over that library or the volume of data very large.

Core Concepts
-------------

The model-grid workflow is built around several key modules/classes:

- :class:`besta.grid.grid.ModelGrid`: container for model observables/targets and metadata.
- :class:`besta.grid.grid.GridFitter`: computes posterior weights over grid models.
- :mod:`besta.grid.prob`: prior and likelihood building blocks.
- :mod:`besta.grid.binning`: candidate selectors to avoid evaluating the full grid for every object.

Working with ModelGrids
-----------------------

A :class:`~besta.grid.grid.ModelGrid` is a container of a grid of models split into ``observables`` and ``targets`` (see example below). They support multiple I/O formats:

- FITS tables: :meth:`~besta.grid.grid.ModelGrid.from_fits_table`,
  ``ModelGrid.to_fits_table(...)``
- HDF5: :meth:`~besta.grid.grid.ModelGrid.from_hdf5`,
  :meth:`~besta.grid.grid.ModelGrid.to_hdf5`
- Pickle: :meth:`~besta.grid.grid.ModelGrid.from_pickle`,
  :meth:`~besta.grid.grid.ModelGrid.to_pickle`
- automatic loader: :meth:`~besta.grid.grid.ModelGrid.load_auto`

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


Minimal Workflow
----------------

1. Build or load a :class:`~besta.grid.grid.ModelGrid`.
2. Create a :class:`~besta.grid.grid.GridFitter` with a likelihood and a set of priors.
3. Evaluate posterior summaries for one object, or run :meth:`~besta.grid.grid.GridFitter.fit_batch` for many.

Example (single-object posterior on one target):

.. code-block:: python

   import numpy as np
   from besta.grid import ModelGrid, GridFitter
   from besta.grid.prob import GaussianProductLikelihood, FlatPrior

   # N models, 3 observables, 4 targets
   grid = ModelGrid(
       observables=obs_models,                # shape (N, 3)
       targets=target_models,                 # shape (N, 3)
       observable_names=["mag_g", "mag_r", "mag_i"],
       target_names=["logM", "age", "Z", "z"],  # (stellar mass, mean age, metals, redshift)
   )

   fitter = GridFitter(
       grid=grid,
       likelihood=GaussianProductLikelihood(),
       prior=FlatPrior(),
       use_standardised=True,
   )

   x = np.array([22.1, 21.5, 21.2])          # observed data (3,)
   sigma_x = np.array([0.03, 0.03, 0.04])         # observational errors (3,)
   bins = np.linspace(7.0, 12.0, 101)        # bins for logM

   # Estimate the marginal stellar mass posterior PDF
   post_logM, centers = fitter.posterior_over_target(
       x_native=x,
       sigma_native=sigma_x,
       target_col="logM", 
       bins=bins,
   )


Batch Inference
----------------

For large samples, use :meth:`~besta.grid.grid.GridFitter.fit_batch`.
This method supports:

- optional candidate selection (`binner=`),
- thread/process parallelism (`n_jobs`, `backend`),
- posterior truncation controls (`posterior_keep_mass`, etc.),
- optional per-target summary statistics (`stats_for`, `stats_bins`),
- optional HDF5 output (`output_hdf5_path`).


.. code-block:: python

   from besta.grid.binning import KDTreeBinner

   binner = KDTreeBinner(dims=[0, 1, 2]).fit(grid)  # Use the first three observable columns

   results = fitter.fit_batch(
       X_native=X_catalog,                    # shape (M, P)
       SIG_native=SIG_catalog,                # shape (M, P)
       binner=binner,
       n_jobs=8,
       backend="thread",
       stats_for=["logM", "age"],  # the posterior statistics for these two quantities
       stats_bins=[np.linspace(7, 12, 120), np.linspace(0, 14, 120)],
       output_hdf5_path="grid_fit_results.h5",
       output_hdf5_group="/run1",
       return_mode="iter",
   )

   for r in results:
       # r contains: m, candidates, post_models, truncation, and optional stats
       pass


Candidate selection
^^^^^^^^^^^^^^^^^^^

This is an essential feature when it comes to using large high-dimensional model grids and large datasets. The binners act as model cadidate selectors, rather than using the entire grid on each evaluation, based on the observables of the input sources. This reduces significantly the number of posterior evaluations (sometimes by orders of magnitude).

Posterior analysis
^^^^^^^^^^^^^^^^^^

BESTA includes some built-in tools for post-processing the posterior PDF and estimate several key quantities such as percentiles, MAP, or covariance matrix, per source.



