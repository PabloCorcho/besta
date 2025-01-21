.. _installation:

==================
Installation Guide
==================

System Requirements
===================

- Python 3.8 or <3.12

Installing CosmoSIS
*******************

BESTA uses the ``cosmosis`` library for performing the Monte Carlo sampling operations and therefore it is a core component of this package. Users can follow the official documentation `page <https://cosmosis.readthedocs.io/en/latest/intro/installation.html>`_.

Recommended installation
^^^^^^^^^^^^^^^^^^^^^^^^

The recommended way of installing ``cosmosis`` and ``besta`` is by means of a ``conda`` environment. Here I provide the basic instruction for setting it up.

Create a new conda environment and include relevant channels (NOTE: with python >3.12 the installation does not work)

.. code-block:: bash

    conda create --name cosmo-env
    conda config --add channels conda-forge
    conda config --set channel_priority strict


To speed up the installation of CosmoSIS, it is recommended to install first `mamba`

.. code-block:: bash

    conda install -y mamba


Now we can install ``cosmosis`` (this will create some directories in your current working directory)

.. code-block:: bash

    mamba install -y cosmosis cosmosis-build-standard-library

Once installed, every time a terminal is open run:

.. code-block:: bash

    conda activate cosmo-env
    source cosmosis-configure


Installing BESTA
================

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
