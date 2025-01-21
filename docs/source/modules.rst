.. _pipeline_modules:

Pipeline Modules
================

BESTA performs the inference of physical parameters from observables by means of pipeline modules. This serve as recipes where the input observable is already predefined and target a particular task (e.g. the inference of stellar kinematics from spectroscopic data). 

Technically, each pipeline module of ``besta`` consists of a Module class with are based on the  ``cosmosis.ClassModule`` class (see the :ref:`BaseModule <base_module>` API documentation for details).
In essence, any pipeline module includes:

- A set of instructions in the constructor to set up the required data and models used during the fit.
- A method ``make_observable`` that takes as arguments a ``DataBlock`` with the parameters used in the model and returns the model observable.
- An ``execute`` method used by ``CosmoSIS`` during runtime.

Available modules
*****************

kin_dust
^^^^^^^^
Module devoted for the inference of the stellar kinematics from spectroscopic data

sfh_spectra
^^^^^^^^^^^
Inference of parameters of a given Chemical Evolution Model using spectroscopic data.

full_spectral_fit
^^^^^^^^^^^^^^^^^
Kinematics and CEM model properties using spectroscopic data. 

sfh_photometry
^^^^^^^^^^^^^^
Inference of parameters of a given Chemical Evolution Model using photometric data.
