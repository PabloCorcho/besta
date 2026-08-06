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

full_spectral_fit
^^^^^^^^^^^^^^^^^
Kinematics and CEM model properties using spectroscopic data. 

spectra_redshift_fit
^^^^^^^^^^^^^^^^^^^^
Measure spectroscopic redshifts.

galaxy_spectra
^^^^^^^^^^^^^^
Spectroscopic fitting with a galaxy (multi-component) emission model.

galaxy_photometry
^^^^^^^^^^^^^^^^^
Photometric fitting with a galaxy (multi-component) emission model.