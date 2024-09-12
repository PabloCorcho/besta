# 

This package, inspired by [HBSPS](https://github.com/mdries/HBSPS) 
(Hierarchical Bayesian Stellar Population Synthesis), is devoted to inferring physical properties from spectroscopic and/or photometric data by performing SED fitting on a hierarchical basis.

# Package contents

## [Population Synthesis Toolkit](https://github.com/paranoya/population-synthesis-toolkit/tree/main)

Highly-flexible framework for performing stellar population synthetis.

## [Cosmosis](https://cosmosis.readthedocs.io/en/latest/)

A library for parameter estimation via Monte Carlo methods.

## Kinematics and dust extinction

The current version assumes a single dust screen and single LOSVD for all SSP's that contribute to a given SED.

## Star formation histories

The star formation histories used to predict the SED can be divided into two groups.

### Analytic

- Exponential
- Delayed-tau
- Lognormal

### Numerical

- Fixed time bins
- Fixed mass fraction