# Bayesian Estimator for Stellar Population Analysis (BESTA)

[![Documentation Status](https://readthedocs.org/projects/besta/badge/?version=latest)](https://besta.readthedocs.io/en/latest/?badge=latest)
[![test](https://github.com/PabloCorcho/besta/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/PabloCorcho/besta/actions/workflows/test.yml)

BESTA is a Python package for inferring stellar population properties from spectroscopic and/or photometric data using Bayesian inference and Monte Carlo methods.

Full documentation is available at [besta.readthedocs.io](https://besta.readthedocs.io).

## Framework

BESTA is built on top of two core libraries:

- **[Population Synthesis Toolkit (PST)](https://github.com/paranoya/population-synthesis-toolkit)** — flexible stellar population synthesis models.
- **[CosmoSIS](https://cosmosis.readthedocs.io)** — modular parameter estimation via Monte Carlo sampling.

## Features

### Pipeline modules

Ready-to-use CosmoSIS pipeline modules are provided in `besta.pipeline_modules`:

| Module | Description |
|---|---|
| `GalaxySpectraModule` | Full spectroscopic SED fitting |
| `FullSpectralFitModule` | Joint stellar population and kinematics fit from spectra |
| `GalaxyPhotometryModule` | Broadband photometric SED fitting |
| `SFHPhotometryGridModule` | SFH inference from photometry via model grid |
| `SFHPhotometryEmulatorModule` | SFH inference from photometry via emulator |

### Star formation history models

Analytic: `ExponentialSFH`, `DelayedTauSFH`, `DelayedTauQuenchedSFH`, `LogNormalSFH`, `LogNormalQuenchedSFH`

Piece-wise: `FixedTimeSFH`, `FixedTime_sSFR_SFH`, `FixedMassFracSFH`

## Installation

```bash
pip install besta
```

Requires Python ≥ 3.10. See the [installation guide](https://besta.readthedocs.io) for CosmoSIS setup instructions.
