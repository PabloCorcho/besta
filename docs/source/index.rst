.. Bayesian Estimator for Stellar Population Analysis documentation master file, created by
   sphinx-quickstart on Tue Nov  5 16:21:59 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bayesian Estimator for Stellar Population Analysis (BESTA)
==========================================================

What is the purpose of BESTA?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

BESTA is a framework devoted to infer physical properties of galaxies or other astronomical objects from observational data such as spectroscopy or photometry.
This is achived by combining forward modelling and Bayesian inference techniques, broadly known in the field as SED fitting.

By analyzing both galaxy spectra and photometric data, BESTA provides robust and statistically rigorous estimates of key astrophysical parameters, such as stellar mass, star formation history, metallicity, and dust extinction.

What can BESTA do?
^^^^^^^^^^^^^^^^^^

BESTA is a project built around several key principles.

1) To support multi-wavelength modelling of galaxies.

The next generatoin of optical and near-IR wide-field imaging surveys (Euclid, Roman, or LSST), the new spectroscopic facilities (WEAVE, 4MOST, PFS), and the future radio observatories such as SKA.

2) To provide a flexible Bayesian inference framework that can handle different levels of data volume and complexity.

This allows to use the same pipeline and underlying models for different datasets while accounting for model uncertainties and degeneracies in a Bayesian fashion.
For example, ground-based and Euclid photometry to infer the recent variation on the star-formation history of galaxies, 

How does BESTA work?
^^^^^^^^^^^^^^^^^^^^

BESTA is designed for both expert researchers and beginners in the field of extragalactic astronomym. It features a user-friendly interface and highly customizable workflows, enabling users to tailor their analyses to specific datasets and scientific goals. Whether you're studying individual galaxies or large cosmological surveys, BESTA ensures accurate and reproducible results.

BESTA supports two complementary inference workflows:

- **CosmoSIS-based sampling pipelines**, documented in :ref:`pipeline_manager`.

This is the preferred method for fitting complex models (e.g. stellar kinematics) using moderate volumes of data. The model parameters are explored using the CosmoSIS library [Zunt+15]_, which provide a large number of 
different Monte Carlo sampling methods (e.g. emcee, multinest, pocoMC, etc.).

- **Direct grid-based inference** (no sampler runtime required), documented in :ref:`grid_inference`.

This method is useful when inferring properties from coarse SEDs and large datasets (e.g. Euclid).
The main strategy is to use pre-computed grids of models in combination with candidate-selection tools from :mod:`besta.grid.binning` to allow the inference of physical properties for thousands of galaxies in seconds.

For more information, you can read the section listed below.

Acknowledgements
^^^^^^^^^^^^^^^^

If you use BESTA in your research, please cite the following papers:

- BESTA itself, including the package version used for the analysis.
- The Population Synthesis Toolkit (PST) [PSTPaper]_.
- CosmoSIS for sampler-based inference workflows [Zunt+15]_.




.. toctree::
   :maxdepth: 1
   :caption: Contents

   quick_guide
   installation
   modules
   configuration
   manager
   postprocessing
   grid
   contributing
   api  

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. [Zunt+15] `CosmoSIS: Modular cosmological parameter estimation <https://ui.adsabs.harvard.edu/abs/2015A%26C....12...45Z/abstract>`_
.. [PSTDocs] `Population Synthesis Toolkit documentation <https://population-synthesis-toolkit.readthedocs.io/en/latest/>`_
.. [PSTPaper] `Population Synthesis Toolkit: A Python package for stellar population synthesis modelling <https://ui.adsabs.harvard.edu/abs/2025JOSS...10.8203C/abstract>`_