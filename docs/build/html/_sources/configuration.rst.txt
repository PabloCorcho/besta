.. _configuration:

Configuring BESTA
=================

CosmosSIS config files
**********************

For details on CosmosSIS parameter files, refer to the `official documentation <https://cosmosis.readthedocs.io/en/latest/usage/parameter_files.html>`_.

BESTA configuration file
************************

BESTA's general configuration is managed through a YAML file (besta-config.yml).
By default, BESTA uses the configuration file located in the source directory of
the package. However, users can specify their own configuration file by setting
the environment variable ``$besta_config`` to the path of the new file.

You can view the contents of the default configuration file
`here <https://github.com/PabloCorcho/besta/blob/main/src/besta/besta-config.yml>`_.


Pipeline modules configuration
******************************

Single Stellar Population models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here's an exaustive list of the possible configuration parameters that can be set.

- ``SSPModel``

Specifies the SSP model from the `PST <https://population-synthesis-toolkit.readthedocs.io/en/latest/>`_ library.

- ``SSPDir``

Specifies an alternative directory for SSP data if it is not located in the
default PST directory.

- ``SSPModelArgs``

Additional arguments passed to the SSP model during instantiation. If multiple
arguments are provided, they should be separated by a comma

**Example:**

If ``SSPModel="EMILES"``, you can select the isochrone and IMF models like this:

.. code-block:: yaml

    SSPModelArgs: "BASTI,KROUPA_UNIVERSAL"

- ``SSP-NMF-N``

Reduces the SSP model to a smaller subsample using non-negative matrix
factorization (NMF), which can decrease computation time. The value of
``SSP-NMF-N`` specifies the number of components.

**Note**: Reducing the model in this way results in non-physical metallicity and age
for each component.

- ``SaveSSPModel``

Specifies the path to a pickle file where the fully configured SSP model can be
stored. This reduces overhead when fitting multiple objects, such as when using
IFS data.

**Tip**: This is especially useful when the model needs to be convolved with an
instrumental line spread function that remains constant across measurements.

- ``SSPModelFromPickle``

Specifies the path to a pre-saved pickle file containing an SSP model, allowing
reuse of previously configured models.
