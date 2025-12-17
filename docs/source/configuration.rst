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

This sections describes the configuration parameters used by BESTA fitting
modules (see :class:`besta.pipeline.base.BaseModule` and its specialised subclasses).

Single Stellar Population models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These options control how the SSP templates are loaded and prepared in
:meth:`besta.pipeline.base.BaseModule.prepare_ssp_model`.

- ``SSPModel``

  Name of the SSP model class from the `PST <https://population-synthesis-toolkit.readthedocs.io/en/latest/>`_
  library (e.g. ``EMILES``, ``BC03``, ...). The class is instantiated via
  ``getattr(pst.SSP, SSPModel)(...)``.

- ``SSPDir``

  Directory containing the SSP data. If set to a value containing ``"none"``
  (case-insensitive), the default PST path is used.

- ``SSPModelArgs``

  Additional arguments passed to the SSP model constructor. Multiple arguments
  must be provided as a comma-separated string.

  **Example**

  If ``SSPModel="EMILES"``, you can select the isochrone and IMF models like this:

  .. code-block:: yaml

      SSPModel: EMILES
      SSPDir: /path/to/pst_data
      SSPModelArgs: "BASTI,KROUPA_UNIVERSAL"

- ``velscale``

  Velocity scale (km/s) used to log-rebin the SSP spectra onto a constant
  :math:`\Delta\ln\lambda` grid. This must match the spectral preparation used
  for the observed data when fitting spectra.

  The logarithmic bin width is computed as:

  .. code-block:: python

      dlnlam = velscale / c_kms

  where ``c_kms`` is the speed of light in km/s.

- ``SSPLSF``

  Optional file describing the intrinsic SSP spectral resolution as a function
  of wavelength. The file must contain at least two columns: wavelength and
  FWHM. If provided, the SSP resolution is combined with the instrumental line
  spread function to compute the effective broadening required to match the
  observed data.

  .. note::

     BESTA currently assumes Gaussian LSFs. Ensure that the wavelength units
     in the LSF file are consistent with your templates/observations.

- ``SSP-NMF-N``

  Reduce the SSP template matrix using non-negative matrix factorisation (NMF)
  to decrease computation time. The value of ``SSP-NMF-N`` sets the number of
  components retained.

  .. warning::

     NMF components do **not** correspond to physical SSPs at specific ages and
     metallicities, and should only be used as a computational acceleration.

- ``SaveSSPModel``

  Path to a pickle file where the fully configured SSP model object is stored.
  This can reduce overhead when fitting multiple objects (e.g. IFS data), in
  particular when templates are convolved with a wavelength-dependent LSF.

- ``SSPModelFromPickle``

  Path to a pickle file containing a previously saved SSP model. If set, the
  model is loaded directly and the configuration steps (including interpolation
  and LSF convolution) are skipped.


Spectral data preparation
^^^^^^^^^^^^^^^^^^^^^^^^^

These options are used by
:meth:`besta.pipeline.base.SpectraFitModule.prepare_observed_spectra`.

- ``inputSpectrum``

  Path to a plain-text file containing three columns: wavelength, flux, and
  flux uncertainty.

- ``wlUnits`` *(optional)*

  Wavelength unit of the first column (e.g. ``Angstrom``, ``nm``, ``micron``).
  If not provided, Angstroms are assumed.

- ``fluxUnits`` *(optional)*

  Flux-density unit of the second and third columns. If not provided, BESTA
  assumes ``1e-16 erg / (s cm2 Angstrom)`` (see module source for details).

- ``redshift`` *(optional)*

  Input redshift used to shift the observed wavelength array to the rest frame.
  If not provided, it defaults to 0.

- ``wlRange`` *(optional)*

  Two-element array defining the wavelength range (in ``wlUnits``) included in
  the fit. If not provided, the full wavelength range is used.

- ``wlNormRange`` *(optional)*

  Two-element array defining the wavelength range (in ``wlUnits``) used to
  normalise the spectrum when ``normalize=True`` is enabled.

- ``mask`` *(optional)*

  Path to a text file containing a per-pixel weight/mask array. If not provided,
  all pixels are assigned weight 1. The mask size must match the spectrum length.

- ``lsf`` *(optional)*

  Path to a text file describing the instrumental line spread function. The file
  must contain wavelength and FWHM columns. The FWHM is interpolated onto the
  observed wavelength grid and stored as ``config["lsf"]``.

- ``velscale`` *(optional)*

  If provided, the observed spectrum is log-binned to a constant velocity scale
  (km/s). If omitted, the spectrum is kept on its native wavelength grid.


Photometric data preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These options are used by
:meth:`besta.pipeline.base.PhotometryFitModule.prepare_observed_photometry`.

- ``inputPhotometry``

  Path to a plain-text file with one filter per row. The file must contain:
  filter identifier, flux, and flux uncertainty.

  Filter identifiers can be:
  - paths to local filter throughput files (loaded with
    :meth:`pst.observables.Filter.from_text_file`), or
  - SVO filter names (loaded with :meth:`pst.observables.Filter.from_svo`).

- ``fluxUnits`` *(optional)*

  Flux unit of the photometry columns. If provided, BESTA converts values to
  nanomaggies. If not provided, BESTA assumes the input is already in nanomaggies.

- ``flux_to_lum`` *(optional)*

  If set to ``True``, scales the fluxes by a distance-based factor using the
  provided ``redshift``. This is intended for producing an absolute-flux-like
  normalisation (see code for current implementation).

  .. note::

     This option assumes a cosmology defined in :mod:`besta.config`.


Star formation history models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These options are used by :meth:`besta.pipeline.base.BaseModule.prepare_sfh_model`.

- ``SFHModel``

  Name of the SFH model class from :mod:`besta.sfh`. The model is instantiated
  via ``getattr(besta.sfh, SFHModel)(*SFHArgs, **config)``.

- ``SFHArgs1``, ``SFHArgs2``, ...

  Optional positional arguments passed to the SFH model constructor. Arguments
  are read sequentially until a key is missing.

  String values containing commas are interpreted as arrays of floats. For example:

  .. code-block:: yaml

      SFHModel: SomeSFHModel
      SFHArgs1: 1.5
      SFHArgs2: "0.1,0.3,1.0,3.0"


Dust extinction law
^^^^^^^^^^^^^^^^^^^

These options are used by :meth:`besta.pipeline.base.BaseModule.prepare_extinction_law`.

- ``ExtinctionLaw`` *(optional)*

  Name of the dust attenuation/extinction law passed to
  :class:`pst.dust.DustScreen`. If not provided, no extinction law is applied.


Multiplicative polynomials
^^^^^^^^^^^^^^^^^^^^^^^^^^

When fitting spectra, a multiplicative polynomial can be included in the model to
absorb smooth flux-calibration residuals and/or large-scale continuum mismatches.
This is configured in
:meth:`besta.pipeline.base.SpectraFitModule.prepare_legendre_polynomials`.

To enable the polynomial, set:

- ``legendre_deg``

  Maximum degree of the Legendre polynomial basis.

Optional parameters:

- ``legendre_bounds``

  Two-element array defining the minimum and maximum wavelength used to
  renormalise the wavelength vector and evaluate the polynomials. If not
  provided, the observed spectral edges are used.

- ``legendre_scale``

  Characteristic scale (Angstrom) that sets the lowest polynomial order to be
  included. If provided, the minimum order is estimated as:

  .. code-block:: python

      min_order = np.round((wl_max - wl_min) / scale)

  The polynomial orders used range from ``min_order`` to
  ``min_order + legendre_deg``.

- ``legendre_clip_first_zero``

  If set, values of each polynomial below the first zero and above the last zero
  are set to 0 to reduce edge oscillations.
