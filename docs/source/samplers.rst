.. _samplers:

Sampler Extensions
==================

BESTA can use custom CosmoSIS samplers in addition to the standard sampler
set. The currently included custom samplers are:

- :class:`besta.samplers.pymc_sampler.PymcSampler`

PyMC sampler
------------

The PyMC sampler wraps a BESTA/CosmoSIS pipeline posterior inside a PyMC model
and runs gradient-free MCMC step methods suitable for black-box likelihoods.

Loading in CosmoSIS
^^^^^^^^^^^^^^^^^^^

Add the custom sampler module file in the ``[runtime]`` section of the configuration file:

.. code-block:: ini

   [runtime]
   import_samplers = /path/to/besta/src/besta/samplers/pymc_sampler.py
   sampler = pymc

Then add a dedicated sampler section:

.. code-block:: ini

   [pymc]
   samples = 1000
   burn_fraction = 0.2
   chains = 2
   step_method = demetropolisz
   progressbar = F

Supported options
^^^^^^^^^^^^^^^^^

The following configuration keys are supported by
:class:`besta.samplers.pymc_sampler.PymcSampler`:

- ``samples``: number of posterior samples to draw (default: 1000).
- ``burn_fraction``: burn-in/tuning length. If in ``[0,1)``, interpreted as a
  fraction of ``samples``; otherwise interpreted as an integer count.
- ``chains``: number of MCMC chains.
- ``step_method``: one of ``metropolis``, ``demetropolis``,
  ``demetropolisz``, or ``slice``.
- ``progressbar``: enable/disable PyMC progress bar.
- ``seed``: random seed (negative values disable explicit seeding).
- ``start_method`` and ``start_input``: standard CosmoSIS starting-point
  controls. A peak supplied by a preceding sampler takes precedence, matching
  the behavior of other CosmoSIS MCMC samplers.
- ``start_jitter``: standard deviation of the Gaussian perturbation applied in
  unit-cube coordinates to initialize additional chains (default: ``1e-3``).
- ``start_attempts``: maximum attempts to find each valid perturbed chain start
  (default: ``1000``).
- ``start_edge_buffer``: minimum distance between initial values and the unit
  cube boundaries (default: ``1e-6``).
- ``start_min_posterior``: minimum posterior accepted for a chain start
  (default: ``-1e19``), chosen to reject BESTA's ``-1e20`` invalid-model
  sentinel.

Notes
^^^^^

- Gradient-based methods such as NUTS/HMC are intentionally rejected by this
  sampler because the wrapped pipeline posterior currently does not provide gradients.
- Samples are drawn in normalized unit-cube coordinates and denormalized
  through the CosmoSIS pipeline before evaluating the posterior.
- Each chain starts from the standard CosmoSIS estimate. When samplers are
  chained as ``maxlike pymc``, this is the MaxLike peak. Additional chains are
  initialized with small, independently validated perturbations.
