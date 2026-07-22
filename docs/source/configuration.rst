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
modules (see :class:`besta.pipeline_modules.base_module.BaseModule` and its specialised subclasses).

Single Stellar Population models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These options control how the SSP templates are loaded and prepared in
:meth:`besta.pipeline_modules.base_module.BaseModule.prepare_ssp_model`.

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
:meth:`besta.pipeline_modules.base_module.SpectraFitModule.prepare_observed_spectra`.

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

- ``is_variance`` / ``is_inverse_variance`` *(optional)*

  Control how the third input column is interpreted:

  - ``is_variance=True``: the third column is treated as variance,
  - ``is_inverse_variance=True``: the third column is treated as inverse variance,
  - otherwise: the third column is treated as 1-sigma uncertainty.

  .. note::

     Only one of these flags should be enabled at a time.

- ``snr_clip`` *(optional)*

  Applies an upper S/N clipping by inflating uncertainties where
  ``S/N > snr_clip``. This can prevent a few very high-S/N pixels from
  dominating the likelihood.

- ``mask_non_finite`` *(optional, default: True)*

  If enabled, non-finite flux/error values are automatically masked (weight set
  to zero) before fitting.

- ``lsf`` *(optional)*

  Path to a text file describing the instrumental line spread function. The file
  must contain wavelength and FWHM columns. The FWHM is interpolated onto the
  observed wavelength grid and stored as ``config["lsf"]``.

- ``mask_telluric`` *(optional)*

  If set, masks atmospheric telluric absorption bands using
  :func:`besta.spectrum.mask_telluric_regions`. You can expand the masked
  windows with ``telluric_pad``.

- ``mask_sky_lines`` *(optional)*

  If set, masks sky-emission lines using
  :func:`besta.spectrum.mask_sky_emission_lines`. You can expand windows with
  ``sky_line_pad``.

- ``mask_emission_lines`` *(optional)*

  If set, masks strong emission lines using
  :func:`besta.spectrum.mask_strong_emission_lines`.

- ``use_features`` *(optional, default: False)*

  Enables feature-driven spectral weighting. Set ``use_features_type`` to:

  - ``"auto"``: continuum-based feature weighting,
  - ``"atlas"``: BESTA uses an atlas of optical absorption features to mask redundant regions without
  significant stellar information.

- ``velscale`` *(optional)*

  If provided, the observed spectrum is log-binned to a constant velocity scale
  (km/s). If omitted, the spectrum is kept on its native wavelength grid. This is
  required for using LOSVD model fitting.


Photometric data preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These options are used by
:meth:`besta.pipeline_modules.base_module.PhotometryFitModule.prepare_observed_photometry`.

- ``inputPhotometry``

  Path to a plain-text file with one filter per row. The file must contain:
  filter identifier, flux, and flux uncertainty.

  Filter identifiers can be:

  - paths to local filter throughput files (loaded with
    ``Filter.from_text_file``), or
  - SVO filter names (loaded with ``Filter.from_svo``).

  An optional fourth column can be used to encode limits per band:

  - ``upper`` for upper limits,
  - ``lower`` for lower limits.

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

These options are used by :meth:`besta.pipeline_modules.base_module.BaseModule.prepare_sfh_model`.

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

- ``use_transforms`` *(optional, default: False)*

  When set to ``True``, SFH models that support it internally transform free
  parameters to enforce physicality (e.g. softmax for mass fractions to ensure
  they sum to 1, sigmoid or cumulative positive deltas to enforce monotonic
  times). This does **not** change parameter names, but it changes their
  meaning: priors in the ``values`` file should then be defined on the
  unconstrained latent variables (typically centred around 0 with symmetric
  ranges, e.g. ``-3 0 3``) rather than on the final fractions/times.

  Supported models: ``FixedTimeSFH``, ``FixedCosmicTimeSFH``, ``FlexibleCosmicTimeSFH``,
  ``FixedMassFracSFH``, and ``FixedTime_sSFR_SFH``. See :mod:`besta.sfh` for the
  precise mappings and their inverses (``to_physical``/``to_latent`` hooks).


Likelihood configuration options
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All modules share the following likelihood-related options.

- ``likelihood_kind`` *(optional)*

  Selects the observable family used by the module likelihood wrapper.
  Allowed values are:

  - ``spectra``
  - ``photometry``

  If omitted, BESTA infers the kind from the module name/class.

- ``likelihood_method`` *(optional, default: ``auto``)*

  Selects the implementation backend used by the module likelihood wrapper. Allowed values are:
  - ``auto`` *(default)*: automatically select the backend based on the observable kind,
  - ``numba``: use the Numba likelihood implementation.

- ``save_chi2`` *(optional, default: False)*

  Enables storage of chi-squared diagnostics in the module outputs when
  supported by the module.

- ``NoiseModel`` *(optional, spectra modules, default: ``NoiseModel``)*

  Selects the noise model used to build the effective inverse-variance vector
  for spectral likelihood evaluation.

  Available values:

  - ``NoiseModel``: use the observed inverse variance unchanged.
  - ``MultiplicativeNoiseModel``: scale inverse variance by a sampled
    parameter ``noise/beta``.

  If ``MultiplicativeNoiseModel`` is selected, define the parameter in the
  values file, for example:

  .. code-block:: ini

     [noise]
     beta = 0.1 1.0 10.0

Values/prior file tips
----------------------

Priors are defined in the CosmoSIS ``values`` file. A typical snippet:

.. code-block:: ini

   [parameters]
   av = 0 0.1 1.0
   los_vel = -500 0 500
   los_sigma = 50 100 400
   logtau = -1 0.5 1.7
   alpha_powerlaw = 0 1 10
   ism_metallicity_today = 0.005 0.01 0.08
   legendre_1 = -0.2 0 0.2

If ``use_transforms=True`` for your SFH, set priors on the latent variables:

- softmax latents: symmetric ranges around 0 (e.g. ``coeff_1 = -3 0 3``),
- sigmoid latents: broad symmetric ranges (e.g. ``coeff_at_2.5 = -3 0 3``),
- exp/delta latents: symmetric ranges around 0 (e.g. ``t_at_frac_0.5 = -3 0 3``).

CosmoSIS will treat single numbers as fixed parameters; missing parameters are
back-filled with fixed values when converting solutions to CosmoSIS ``DataBlock``
objects for plotting or post-processing.

Configuration examples
**********************

Photometry-only fit
^^^^^^^^^^^^^^^^^^^

.. code-block:: ini

   [runtime]
   sampler = maxlike

   [maxlike]
   method = Powell

   [output]
   filename = ./photometry_fit
   format = text

   [pipeline]
  modules = GalaxyPhotometry
   values = ./values.ini
  likelihoods = GalaxyPhotometry

  [GalaxyPhotometry]
  file = path/to/besta/pipeline_modules/galaxy_photometry.py
   inputPhotometry = ./photometry.dat
   fluxUnits = nanomaggie
   redshift = 0.05
   SFHModel = ExponentialSFH
   SSPModel = PopStar
   SSPModelArgs = "cha"
   SSPDir = None
   ExtinctionLaw = ccm89

Single-spectrum fit
^^^^^^^^^^^^^^^^^^^

.. code-block:: ini

   [runtime]
   sampler = maxlike emcee

   [maxlike]
   method = Powell

   [emcee]
   nwalkers = 16
   nsteps = 500

   [output]
   filename = ./spectral_fit
   format = text

   [pipeline]
   modules = FullSpectralFit
   values = ./values.ini
   likelihoods = FullSpectralFit
   extra_output = parameters/stellar_mass

   [FullSpectralFit]
   file = path/to/besta/pipeline_modules/full_spectral_fit.py
   inputSpectrum = ./spectrum.dat
   wlUnits = Angstrom
   fluxUnits = 1e-16 erg / (s cm2 Angstrom)
   wlRange = 3500.0 9000.0
   velscale = 50.0
   SSPModel = PopStar
   SSPModelArgs = cha
   SSPDir = None
   SFHModel = ExponentialSFH
   ExtinctionLaw = ccm89
   legendre_deg = 4

Two spectra of the same object (joint fit)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a given object, you can fit as many observables as you want by including multiple
``FullSpectralFit`` modules in the pipeline. For example, to fit blue and red
spectra of the same galaxy simultaneously with shared parameters:

.. code-block:: python

   from besta.pipeline import MainPipeline
   from besta.pipeline_modules.full_spectral_fit import FullSpectralFitModule

   cfg = {
    
      "runtime": {"sampler": "maxlike emcee"},
         "maxlike": {"method": "Nelder-Mead", "tolerance": 1e-3, "maxiter": 3000},
         "emcee": {"walkers": 32, "samples": 100, "nsteps": 100},

       "output": {"filename": "./fit_all", "format": "text"},
       "pipeline": {
           "modules": "FullSpectralFit_blue FullSpectralFit_red",
           "values": "./values_blue.ini",
           "likelihoods": "FullSpectralFit_blue FullSpectralFit_red",
       },
       "FullSpectralFit_blue": {
           "file": FullSpectralFitModule.get_path(),
           "inputSpectrum": "./spectrum_blue.dat",
           "wlRange": [3500.0, 5500.0],
           "wlUnits": "Angstrom",
           "fluxUnits": "1e-16 erg / (s cm2 Angstrom)",
           "velscale": 70.0,
           "SSPModel": "PopStar",
           "SSPModelArgs": "cha",
           "SSPDir": "None",
           "SFHModel": "ExponentialSFH",
           "ExtinctionLaw": "ccm89",
           "like_name": "FullSpectralFit_blue",
       },
       "FullSpectralFit_red": {
           "file": FullSpectralFitModule.get_path(),
           "inputSpectrum": "./spectrum_red.dat",
           "wlRange": [5500.0, 9000.0],
           "wlUnits": "Angstrom",
           "fluxUnits": "1e-16 erg / (s cm2 Angstrom)",
           "velscale": 50.0,
           "SSPModel": "PopStar",
           "SSPModelArgs": "cha",
           "SSPDir": "None",
           "SFHModel": "ExponentialSFH",
           "ExtinctionLaw": "ccm89",
           "like_name": "FullSpectralFit_red",
       }
   }

   pipeline = MainPipeline([cfg], n_cores_list=[1])
   pipeline.execute_all(plot_result=True)

Note that, while using the same SSP model, the resolution and wavelength range
can differ between spectra.

.. note::

    It is important to set different ``like_name`` values for each module to avoid name clashes in the likelihood calculation.



Dust extinction law
^^^^^^^^^^^^^^^^^^^

These options are used by :meth:`besta.pipeline_modules.base_module.BaseModule.prepare_extinction_law`.

- ``ExtinctionLaw`` *(optional)*

  Name of the dust attenuation/extinction law passed to
  ``pst.dust.DustScreen``. If not provided, no extinction law is applied.


Multiplicative polynomials
^^^^^^^^^^^^^^^^^^^^^^^^^^

When fitting spectra, a multiplicative polynomial can be included in the model to
absorb smooth flux-calibration residuals and/or large-scale continuum mismatches.
This is configured in
:meth:`besta.pipeline_modules.base_module.SpectraFitModule.prepare_legendre_polynomials`.

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
