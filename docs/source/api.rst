.. _api:

API
===

CosmoSIS-based Inference
^^^^^^^^^^^^^^^^^^^^^^^^


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

Input/Output
************

.. automodule:: besta.io
   :members:
   :show-inheritance:

Pipeline Manager
****************
.. _api_pipeline:

.. automodule:: besta.pipeline
   :members:
   :show-inheritance:


Star formation histories
************************

.. automodule:: besta.sfh
   :members:
   :show-inheritance:

Model grid-based inference
^^^^^^^^^^^^^^^^^^^^^^^^^^

For a user-oriented guide (without CosmoSIS sampling), see
:ref:`grid_inference`.

Basics
******

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
   :no-index:
   :show-inheritance:

.. automodule:: besta.grid.transforms
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
