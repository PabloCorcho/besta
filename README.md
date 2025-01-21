# Bayesian Estimator for Stellar Population Analysis (BESTA)

This package provides a set of tools devoted to inferring physical properties from spectroscopic and/or photometric data by means of Bayesian inference and Monte Carlo techniques.

At the moment, its use is primarily focused on deriving the properties of the underlying stellar populations in unresolved spectroscopic or photometric data.

## Status
[![Documentation Status](https://readthedocs.org/projects/besta/badge/?version=latest)](https://besta.readthedocs.io/en/latest/?badge=latest)

# Introduction

## The framework

BESTA leverages on two main libraries.

### [Population Synthesis Toolkit](https://github.com/paranoya/population-synthesis-toolkit/tree/main)

A highly-flexible framework for performing stellar population synthetis.

### [Cosmosis](https://cosmosis.readthedocs.io/en/latest/)

A library for parameter estimation via Monte Carlo methods.

## The philosophy

BESTA is designed for building arbitrary SED-fitting pipelines, with different fitting strategies, and complex models.


### Kinematics and dust extinction

The current version assumes a single dust screen and single LOSVD for all SSP's that contribute to a given SED.

### Star formation histories

The star formation histories used to predict the SED can be divided into two groups.

Analytic SFHs

- Exponential
- Delayed-tau
- Lognormal

Piece-wise SFHs

- Fixed time bins
- Fixed mass fraction

## What's new

### Pipeline modules

The default modules used to perform the fitting using either spectra or photometry, are now located within `src/besta/pipeline_modules`. All module names are now in lower case format.

It is now possible to obtain the path to a given module (an include it on the cosmosis .ini file) by using the `get_path` method:

```python
from besta.pipeline_modules.kin_dust import KinDustModule

path_to_module = KinDustModule.get_path()
print("In the .ini file now you should plug")
print(f"file = {path_to_module}")
print("Right after the section [KinDust]")
```

In order to facilitate the validation of the fits, every default module can now easily initialised in the exact same fashion as it is done during the cosmosis run. This makes things much more easy in terms of visualizing the fit

```python

kin_dust_configuration {
    # Include all the details from the section [KinDust] in the .ini file
} 
block = DataBlock()
block['parameters', 'av'] = 0
block['parameters', 'los_vel'] = 0
block['parameters', 'los_sigma'] = 100.
block['parameters', 'los_h3'] = 0
block['parameters', 'los_h4'] = 0

kindust_module = KinDustModule(kin_configuration)
# Obtain the spectra that results from the solution stored in `block`
flux_model, weights = kindust_module.make_observable(block)
```

#### Kinematics

The kinematic fit now also accepts higher moments (`h3` and `h4`).

### Pipeline runs

Following the previous changes, now setting a pipeline and tracking the results is much easier.