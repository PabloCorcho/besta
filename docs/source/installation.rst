.. _installation:

==================
Installation Guide
==================

System Requirements
===================

- Python >=3.10

Installing BESTA
================

Recommended installation
^^^^^^^^^^^^^^^^^^^^^^^^

The recommended way of installing ``cosmosis`` and ``besta`` is by means of a ``conda`` environment. Here I provide the basic instruction for setting it up.

Create a new conda environment and include relevant channels (NOTE: with python >3.12 the installation does not work)

.. code-block:: bash

    conda create --name besta-env
    conda activate besta-env

You can install BESTA directly from PyPI using `pip`:

.. code-block:: bash

    pip install besta

Installing from Source
======================
To install the package from the source repository:

.. code-block:: bash

    git clone https://github.com/PabloCorcho/besta.git
    cd besta
    python3 -m pip install -r requirements.txt
    python3 -m pip install .
