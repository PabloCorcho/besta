Post-processing and Posterior Analysis
======================================

This module provides a high-level interface to **summarise, analyse, and export**
the results produced by BESTA / CosmoSIS runs. It is designed to sit *after*
sampling and focuses on:

* Robust posterior summaries (MAP, mean, covariance, correlation)
* Credible intervals (percentiles and HDIs)
* One- and two-dimensional posterior PDFs
* Diagnostic plots (PDFs, corner plots, chains)
* Bayesian evidence estimation
* Machine-readable outputs (FITS and JSON)

The core abstraction is the :class:`besta.postprocess.ResultsSummary` object, which collects all
derived quantities in a single, reusable container.

Overview
--------

Typical workflow:

1. Load a CosmoSIS results table (or pass one directly).
2. Call :func:`besta.postprocess.summarize_results`.
3. Inspect or plot results via :class:`besta.postprocess.ResultsSummary`.
4. Export summaries to FITS and/or JSON.

The module is **sampler-agnostic** and works with any posterior sample stored
as a table with a log-posterior column.

Input requirements
------------------

The input table must contain:

* A **log-posterior** column (default name: ``post``)
* One column per sampled parameter, typically named using the convention::

    section--parameter

For evidence estimation, the table must also contain:

* A **log-prior** column (e.g. ``prior``)

If a separate log-likelihood column is not present, it is automatically
reconstructed as::

    loglike = post - prior

Basic usage
-----------

Summarising a results table:

.. code-block:: python

    from besta.postprocess import summarize_results

    summary = summarize_results(
        table,
        output_fits="besta_summary.fits",
        output_json="besta_summary.json",
    )

The returned object is an instance of :class:`besta.postprocess.ResultsSummary`.

Selecting parameters
--------------------

By default, all columns containing ``"--"`` are interpreted as sampled
parameters. You can override this behaviour:

.. code-block:: python

    summary = summarize_results(
        table,
        parameter_keys=[
            "sfh--tau",
            "sfh--age",
            "dust--av",
        ],
    )

Posterior summaries
-------------------

The following quantities are always computed:

* Maximum-a-posteriori (MAP) parameter vector
* Posterior-weighted mean
* Covariance matrix
* Correlation matrix

They are accessible as NumPy arrays:

.. code-block:: python

    summary.map
    summary.mean
    summary.covariance
    summary.correlation

Credible intervals
------------------

Percentiles
~~~~~~~~~~~

Percentiles are computed using **weighted quantiles**:

.. code-block:: python

    summary.percentiles
    summary.percentiles_values

By default, the percentiles are:

``[0.05, 0.16, 0.5, 0.84, 0.95]``

This can be customised:

.. code-block:: python

    summary = summarize_results(
        table,
        percentiles=[0.025, 0.16, 0.5, 0.84, 0.975],
    )

Highest-Density Intervals (HDI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to percentiles, the module computes **highest-density intervals**
(HDIs), which are often more informative for skewed or multi-modal posteriors.

By default, a 68% HDI is computed:

.. code-block:: python

    summary.hdi_intervals["tau"]

HDIs may contain **multiple disjoint intervals** if the posterior is multi-modal.

One-dimensional PDFs
--------------------

For each parameter, the module computes:

* A histogram-based PDF
* An optional KDE-based PDF

These are stored in:

.. code-block:: python

    summary.pdf_1d["tau"]["grid"]
    summary.pdf_1d["tau"]["hist_pdf"]
    summary.pdf_1d["tau"]["kde_pdf"]

You can disable KDE estimation:

.. code-block:: python

    summary = summarize_results(
        table,
        kde_1d=False,
    )

Two-dimensional PDFs
--------------------

Joint PDFs for parameter pairs can be computed as:

.. code-block:: python

    summary = summarize_results(
        table,
        compute_2d=True,
        parameter_key_pairs=[
            ("sfh--tau", "sfh--age"),
            ("dust--av", "sfh--age"),
        ],
    )

For each pair, the following are stored:

* Regular grid in both parameters
* Joint PDF
* Enclosed-fraction (HPD) map

Accessible as:

.. code-block:: python

    pdf = summary.pdf_2d[("tau", "age")]["pdf"]
    frac = summary.pdf_2d[("tau", "age")]["enclosed_fraction"]

The enclosed-fraction map allows easy extraction of 68% / 95% credible contours.

Plotting
--------

1D PDFs
~~~~~~~

.. code-block:: python

    summary.plot_1d_pdfs("plots/pdf1d")

2D PDFs
~~~~~~~

.. code-block:: python

    summary.plot_2d_pdfs("plots/pdf2d")

Corner plots
~~~~~~~~~~~~

A lightweight built-in corner plot is provided:

.. code-block:: python

    summary.corner_plot("corner.png")

This avoids external dependencies, but more advanced diagnostics can be added
later (e.g. via ``corner`` or ``arviz``).

Evidence estimation
-------------------

If the results table contains:

* ``post``  – log posterior
* ``prior`` – log prior

then the Bayesian evidence can be estimated.

Laplace approximation (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    summary = summarize_results(
        table,
        estimate_evidence=True,
        evidence_method="laplace",
        logprior_key="prior",
    )

This computes:

.. math::

    Z \approx \mathcal{L}(\theta_\mathrm{MAP}) \,
              \pi(\theta_\mathrm{MAP}) \,
              (2\pi)^{d/2} \, |\Sigma|^{1/2}

where :math:`\Sigma` is the posterior covariance matrix.

Robust harmonic-mean estimator (sanity check)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    summary.estimate_evidence_from_table(
        table,
        method="hme_robust",
        logprior_key="prior",
    )

.. warning::

    The harmonic-mean estimator is **statistically unstable** and should only
    be used as a rough cross-check. The Laplace approximation is preferred.

The estimated evidence is available as:

.. code-block:: python

    summary.evidence.logz
    summary.evidence.method
    summary.evidence.details

Outputs
-------

FITS
~~~~~

The FITS output contains:

* Covariance and correlation matrices
* Percentile tables
* 1D PDFs
* 2D PDFs and enclosed-fraction maps
* Evidence (stored in the primary header)

JSON
~~~~

The JSON output provides a **fully serialised summary**, including:

* Posterior statistics
* Credible intervals
* PDFs
* Evidence
* User-provided metadata

This format is intended for:

* Dashboards
* Pipeline aggregation
* Downstream inference and validation

Design philosophy
-----------------

This module intentionally separates:

* **Sampling** (CosmoSIS / BESTA)
* **Inference** (this module)
* **Visualisation / reporting**

This makes posterior analysis reproducible, testable, and reusable across
projects and datasets.
