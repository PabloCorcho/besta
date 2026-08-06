.. _cli:

Command-Line Interface
======================

BESTA installs two console commands:

- ``besta`` for post-processing and diagnostics.
- ``besta-run`` for executing fits through :class:`besta.pipeline.MainPipeline`.

Basic usage
-----------

.. code-block:: bash

   besta <file> [options]

where ``<file>`` is either:

- a BESTA/CosmoSIS results file, or
- an ``.ini`` file (when ``--from_ini`` is set).

Main commands
-------------

Generate best-fit plots from a run:

.. code-block:: bash

   besta path/to/run.ini --from_ini --make_best_fit

Generate summary statistics FITS from a results table:

.. code-block:: bash

   besta path/to/results.txt --make_summary_statistics --output summary.fits

Available options
-----------------

- ``--from_ini``: treat input ``file`` as an ini file.
- ``--make_best_fit``: produce best-fit plots for each configured module.
- ``--make_corner_plot``: request a corner plot.
- ``--make_summary_statistics``: compute and write summary statistics.
- ``--output``: output FITS path used with ``--make_summary_statistics``.

Current limitations
-------------------

At present, ``--make_corner_plot`` is a placeholder in the CLI implementation
and does not yet generate output.


Running fits from the CLI
-------------------------

Use ``besta-run`` to launch a fit directly from configuration files.

Basic usage
^^^^^^^^^^^

.. code-block:: bash

   besta-run <config.ini>

Examples
^^^^^^^^

Run a single-configuration fit using 4 cores and create best-fit plots:

.. code-block:: bash

   besta-run run.ini --n-cores 4 --plot-result

Options
^^^^^^^

- ``--n-cores``: number of cores per subpipeline (default: 1, ``-1`` for all).
- ``--n-cores-list``: comma-separated per-subpipeline core counts.
- ``--ini-files``: comma-separated existing ini files (one per subpipeline).
- ``--ini-values-files``: comma-separated values files (one per subpipeline).
- ``--plot-result``: generate module best-fit plots after each run.
