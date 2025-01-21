.. Bayesian Estimator for Stellar Population Analysis documentation master file, created by
   sphinx-quickstart on Tue Nov  5 16:21:59 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Bayesian Estimator for Stellar Population Analysis (BESTA)
==========================================================

BESTA (Bayesian Estimator for Stellar Populations Analysis) is an advanced software tool designed to infer the physical properties of galaxies using state-of-the-art Bayesian inference and Monte Carlo techniques [Zunt+15]_. By analyzing both galaxy spectra and photometric data, BESTA provides robust and statistically rigorous estimates of key astrophysical parameters, such as stellar mass, star formation history, metallicity, and dust extinction.

Modern astrophysical observations demand sophisticated tools to extract meaningful insights from complex datasets. BESTA addresses this challenge by combining:
- Bayesian Framework: Offers a probabilistic approach to quantify uncertainties and model degeneracies inherent in the analysis of galaxy properties.
- Monte Carlo Simulations: Employs efficient sampling techniques to explore high-dimensional parameter spaces and deliver posterior probability distributions.

BESTA is designed for both expert researchers and beginners in the field of extragalactic astronomy. It features a user-friendly interface and highly customizable workflows, enabling users to tailor their analyses to specific datasets and scientific goals. Whether you're studying individual galaxies or large cosmological surveys, BESTA ensures accurate and reproducible results.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   installation
   modules
   configuration
   manager
   contributing
   api  

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. [Zunt+15] `CosmoSIS: Modular cosmological parameter estimation <https://ui.adsabs.harvard.edu/abs/2015A%26C....12...45Z/abstract>`_
