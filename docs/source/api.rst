.. _api:

API Reference
=============

Package
^^^^^^^

.. automodule:: besta
   :members:

CosmoSIS-based Inference
^^^^^^^^^^^^^^^^^^^^^^^^

Core modules
************

.. automodule:: besta.config
   :members:
   :show-inheritance:

.. automodule:: besta.logging
   :members:
   :show-inheritance:

.. automodule:: besta.io
   :members:
   :show-inheritance:

.. automodule:: besta.pipeline
   :members:
   :show-inheritance:

Pipeline Modules
****************

.. toctree::
   :maxdepth: 1

   besta.pipeline_modules

Kinematics
**********

.. automodule:: besta.kinematics
   :members:
   :show-inheritance:

.. automodule:: besta.spectrum
   :members:
   :show-inheritance:

Star formation histories
************************

.. automodule:: besta.sfh
   :members:
   :show-inheritance:

Utilities
*********

.. automodule:: besta.utils
   :members:
   :show-inheritance:

.. automodule:: besta.visualization
   :members:
   :show-inheritance:

Model grid-based inference
^^^^^^^^^^^^^^^^^^^^^^^^^^

For a user-oriented guide (without CosmoSIS sampling), see
:ref:`grid_inference`.

Basics
******

.. automodule:: besta.grid

.. autoclass:: besta.grid.grid.ModelGrid
   :members:
   :show-inheritance:

Candidate selection
*******************

.. automodule:: besta.grid.binning
   :members:
   :show-inheritance:

.. automodule:: besta.grid.prob
   :members:
   :show-inheritance:

.. automodule:: besta.grid.transforms
   :members:
   :show-inheritance:

.. automodule:: besta.grid.generator
   :members:
   :show-inheritance:

.. automodule:: besta.grid.emulator
   :members:
   :show-inheritance:

Model fitting
*************

.. autoclass:: besta.grid.grid.GridFitter
   :members:
   :show-inheritance:

.. autoclass:: besta.grid.grid.GridFitHDF5Writer
   :members:
   :show-inheritance:

Postprocessing
^^^^^^^^^^^^^^

.. automodule:: besta.postprocess
   :members:
   :show-inheritance:
