"""Equivalent width fitting pipeline module."""

from besta.pipeline_modules.base_module import SpectraFitModule, ml_amplitude
import numpy as np
from astropy import units as u
from astropy.constants import c as _SPEED_OF_LIGHT
from matplotlib import pyplot as plt
from scipy.special import ndtr

from cosmosis.datablock import names as section_names
from cosmosis.datablock import SectionOptions
from besta.logging import get_logger
from besta.visualization import draw_dict_in_axes
from pst.observables import EquivalentWidth

logger = get_logger(__name__)

# Balmer lines: rest-frame (air) wavelength in Angstrom and flux relative to
# H-alpha for Case B recombination (T = 1e4 K, n_e = 100 cm^-3)
_BALMER_WAVELENGTH = np.array([6562.80, 4861.33, 4340.47, 4101.74, 3970.07])
_BALMER_RATIO = np.array([1.0, 1.0 / 2.86, 0.468 / 2.86, 0.259 / 2.86,
                          0.159 / 2.86])
# From Madau & Dickinson (2014)
_LHA_PER_SFR = {"salpeter": 1.0 / 7.9e-42,  # Kennicutt (1998)
                "kroupa": 1.0 / 5.37e-42,  # Murphy et al. (2011)
                "chabrier": 1.0 / 5.37e-42
                }

# Case B: L(Halpha) [erg/s] = 1.37e-12 * Q(H) [photons/s]
_LHA_PER_Q = 1.37e-12
# erg/s -> luminosity units of BaseModule.luminosity_values (1e-16 erg/s)
_LUMINOSITY_PER_ERG_S = 1e16


class EquivalentWidthFitModule(SpectraFitModule):
    """Fit stellar populations to a set of spectral indices.

    Each index is fitted in one of two ways, chosen with the option
    ``fit_mode``:

    - ``ew`` *(default)*: the equivalent widths are measured once on the
      observed spectrum and then compared with those of the model spectrum at
      every sample. To be fast, the EW of each index is estimated from the mean
      flux in its three windows,

      ``EW = (1 - F_C / F_cont) * (lambda_C,max - lambda_C,min)``,

      where ``F_C`` is the mean flux in the central window and ``F_cont`` is
      the linear interpolation of the mean fluxes of the side windows at the
      centre of the central window. Indices that are not fully covered by the
      observed spectrum or that touch masked pixels are discarded.
    - ``spectra``: the pseudo-continuum normalised spectrum in the central
      window of each index, ``F(lambda) / F_cont(lambda)``, is compared pixel
      by pixel with the observed one. The residuals of all indices are
      concatenated into a single likelihood. Masked pixels in the central
      window are excluded (the side windows must be clean). It requires a
      ``kinematics`` section in the parameters (the model is convolved with
      the LOSVD and trimmed to the observed pixels). Pixels shared by
      overlapping indices are counted once per index.

    ``fit_mode`` can be a single mode, applied to every index, or a
    comma-separated list with one mode per index, either in the order of
    ``ew_names`` / ``ew_list`` (``fit_mode = ew, spectra, ew``) or as
    ``name:mode`` pairs (``fit_mode = Mgb:spectra, Fe5270:spectra``), where
    the indices that are not named are fitted as ``ew``. ``ew_list`` can also
    have a ``mode`` column (see below). An index is never fitted in both ways,
    since its EW is a function of its central pixels.

    The model is computed once, but the EW indices and the normalised-spectra
    indices give two independent likelihood terms when the modes come from a
    list or from the ``mode`` column: ``<like_name>_ew_like`` and
    ``<like_name>_spectra_like`` in the data block, where ``<like_name>`` is
    the module likelihood name without its ``_like`` suffix (by default
    ``EquivalentWidthFit``). Both must be listed in the ``likelihoods`` option
    of the ``[pipeline]`` section, for example ``likelihoods =
    EquivalentWidthFit_ew EquivalentWidthFit_spectra``, even if one of them has
    no index. With a single mode there is one term, ``<like_name>_like``.

    Module options (on top of the ones needed to read the spectrum and set up
    the SSP, SFH and dust models):

    - ``ew_names``: comma-separated names of the indices, as understood by
      :meth:`pst.observables.EquivalentWidth.from_name`.
    - ``ew_list`` *(alternative to* ``ew_names`` *)*: text file with one index
      per row and 7 columns, ``name left_min left_max central_min central_max
      right_min right_max`` (Angstrom). It takes precedence if both are given.
      An optional 8th column, ``mode`` (``ew`` or ``spectra``), sets how each
      index is fitted. If ``fit_mode`` is also given, they must agree.
    - ``fit_mode`` *(optional, default* ``ew`` *)*: a mode or a list of modes,
      as described above.
    - ``ew_like_weight``, ``spectra_like_weight`` *(optional, default 1;
      only with two likelihood terms)*: factors multiplying each
      log-likelihood term. Values different from 1 temper the likelihood, so
      the resulting evidence is no longer a strict Bayesian evidence. They are
      useful when the many (correlated) pixels of the spectra indices would
      dominate the few EWs.
    - ``ew_error_floor`` *(optional, default 0)*: error (in Angstrom) added in
      quadrature to the measured uncertainty of each index, to account for
      model systematics (``ew`` mode).
    - ``spectra_error_floor`` *(optional, default 0)*: error, relative to the
      continuum, added in quadrature to every normalised pixel (``spectra``
      mode).
    - ``balmer_emission`` *(optional, default* ``none`` *)*: add the Balmer
      emission (H-alpha to H-epsilon, Case B ratios) to the model.
      ``ionising`` derives H-alpha from the number of ionising photons of the
      SSPs (needs an SSP model that provides it), ``kennicutt`` from the SFR
      averaged over ``balmer_tau`` through the Kennicutt (1998) relation. The
      lines are attenuated by the same ``a_v`` as the stars. Do not mask
      Balmer lines in the observed spectrum when using it.
    - ``balmer_tau`` *(optional, default 10)*: Myr over which the SFR is
      averaged (``kennicutt`` only).
    - ``balmer_sigma`` *(optional, default 2)*: Gaussian sigma (Angstrom) of
      the emission lines (instrumental plus intrinsic).
    """

    name = "EquivalentWidthFit"

    def __init__(self, options, **kwargs):
        """
        Set up the equivalent width fit module.

        Parameters
        ----------
        options : dict or DataBlock
            Options from the startup configuration.
        **kwargs : dict
            Extra keyword arguments forwarded to ``SpectraFitModule``.
        """
        super().__init__(options, likelihood_kind="spectra", **kwargs)
        options = self.parse_options(options)

        if not options.has_value("velscale"):
            raise ValueError(
                "Option 'velscale' is required for setting up EquivalentWidthFitModule.")
        if not (options.has_value("ew_names") or options.has_value("ew_list")):
            raise ValueError("Option 'ew_names' or 'ew_list' is required.")

        # Select how each index is fitted (EW or normalised spectra). The
        # default is registered by get_string, so check before reading it
        self.config["fit_mode_given"] = options.has_value("fit_mode")
        self.config["fit_mode"] = self._parse_fit_mode(
            options.get_string("fit_mode", default="ew"))
        logger.info("Equivalent width module fitting mode: %s",
                    self.config["fit_mode"])

        if options.get_string("balmer_emission", default="none") == "ionising":
            include_ionising_q = True
        else:
            include_ionising_q = False

        self.prepare_observed_spectra(options)
        self.prepare_ssp_model(options, include_ionising_q=include_ionising_q)
        self.prepare_sfh_model(options)
        self.prepare_extinction_law(options)
        self.prepare_equivalent_widths(options)
        self.prepare_likelihood_terms(options)
        self.prepare_balmer_emission(options)
        self.prepare_losvd_kernel(options)

    @staticmethod
    def _window_matrix(wavelength, ranges):
        """Matrix whose product with a spectrum gives the mean in each range."""
        matrix = np.zeros((len(ranges), wavelength.size))
        for row, (low, high) in zip(matrix, ranges):
            inside = (wavelength >= low) & (wavelength <= high)
            if not inside.any():
                raise ValueError(
                    f"No pixels between {low} and {high} Angstrom.")
            row[inside] = 1.0 / inside.sum()
        return matrix

    def _ew_from_means(self, means, rows):
        """EW (Angstrom) of the indices ``rows`` from the window means."""
        left, centre, right = means.reshape(-1, 3)[rows].T
        t = self.config["ew_t"][rows]
        continuum = (1 - t) * left + t * right
        return (1 - centre / continuum) * self.config["ew_dlam"][rows]

    def _ew_error_from_means(self, means, var_means, rows):
        """Propagated error (Angstrom) of :meth:`_ew_from_means`."""
        left, centre, right = means.reshape(-1, 3)[rows].T
        var_left, var_centre, var_right = var_means.reshape(-1, 3)[rows].T
        t, dlam = self.config["ew_t"][rows], self.config["ew_dlam"][rows]
        continuum = (1 - t) * left + t * right
        d_centre = dlam / continuum
        d_left = d_centre * centre * (1 - t) / continuum
        d_right = d_centre * centre * t / continuum
        return np.sqrt(d_centre ** 2 * var_centre + d_left ** 2 * var_left
                       + d_right ** 2 * var_right)

    def _set_indices(self, indices, is_spectra):
        """Store the indices and the quantities derived from their windows.

        ``is_spectra`` flags the indices fitted as normalised spectra; the
        rest are fitted as EWs. ``ew_rows`` and ``spectra_rows`` are the
        positions of each kind in the list of indices.
        """
        # (n_index, window, [min, max]) with windows ordered left, centre, right
        ranges = np.array([[ew.left_wl_range.value, ew.central_wl_range.value,
                            ew.right_wl_range.value] for ew in indices])
        mid = ranges.mean(axis=2)
        is_spectra = np.asarray(is_spectra, dtype=bool)
        self.config["ew_list"] = indices
        self.config["ew_is_spectra"] = is_spectra
        self.config["ew_rows"] = np.flatnonzero(~is_spectra)
        self.config["spectra_rows"] = np.flatnonzero(is_spectra)
        self.config["ew_names"] = [ew.name for ew in indices]
        self.config["ew_ranges"] = ranges.reshape(-1, 2)
        self.config["ew_t"] = (mid[:, 1] - mid[:, 0]) / (mid[:, 2] - mid[:, 0])
        self.config["ew_dlam"] = ranges[:, 1, 1] - ranges[:, 1, 0]

    def _keep_indices(self, keep):
        """Keep only the indices flagged in ``keep``, with their modes."""
        self._set_indices(
            [ew for ew, k in zip(self.config["ew_list"], keep) if k],
            self.config["ew_is_spectra"][keep])

    def _set_pixels(self):
        """Select the central-window pixels of the indices fitted as spectra.

        Stores the pixel positions on the observed grid (``pix_idx``), the
        index each one belongs to (``pix_owner``, a position in the list of
        indices) and its weight ``t`` for the pseudo-continuum interpolation
        (``pix_t``). Masked pixels are skipped.
        """
        wl = self.config["wavelength"].value
        good = (self.config["weights"] > 0) & np.isfinite(self.config["flux"])
        ranges = self.config["ew_ranges"].reshape(-1, 3, 2)
        mid = ranges.mean(axis=2)
        idx, owner = [], []
        for ith in self.config["spectra_rows"]:
            low, high = ranges[ith, 1]
            pixels = np.flatnonzero((wl >= low) & (wl <= high) & good)
            idx.append(pixels)
            owner.append(np.full(pixels.size, ith))
        idx = np.concatenate(idx) if idx else np.empty(0, dtype=int)
        owner = np.concatenate(owner) if owner else np.empty(0, dtype=int)
        self.config["pix_idx"] = idx
        self.config["pix_owner"] = owner
        self.config["pix_wl"] = wl[idx]
        self.config["pix_t"] = ((wl[idx] - mid[owner, 0])
                                / (mid[owner, 2] - mid[owner, 0]))

    def _continuum_pixels(self, means):
        """Pseudo-continuum at each fitted pixel from the window means."""
        owner, t = self.config["pix_owner"], self.config["pix_t"]
        return (1 - t) * means[3 * owner] + t * means[3 * owner + 2]

    @staticmethod
    def _parse_fit_mode(value):
        """Parse ``fit_mode``.

        Returns a mode (``ew`` or ``spectra``) for all the indices, a list of
        modes (one per index) or a dictionary ``{index name: mode}``.
        """
        tokens = [token.strip().strip("\"'").strip() for token in value.split(",")]
        tokens = [token for token in tokens if token]
        if not tokens:
            raise ValueError("Option 'fit_mode' is empty.")
        named = [":" in token for token in tokens]
        if any(named) and not all(named):
            raise ValueError(
                "fit_mode cannot mix 'name:mode' pairs with plain modes.")
        if all(named):
            parsed = {}
            for token in tokens:
                name, mode = (part.strip() for part in token.rsplit(":", 1))
                if name in parsed:
                    raise ValueError(f"fit_mode has the index {name!r} twice.")
                parsed[name] = mode.lower()
            modes = list(parsed.values())
        else:
            parsed = modes = [token.lower() for token in tokens]
        for mode in modes:
            if mode not in {"ew", "spectra"}:
                raise ValueError(
                    f"Unknown fit_mode {mode!r}; expected ew or spectra.")
        return parsed[0] if isinstance(parsed, list) and len(parsed) == 1 else parsed

    def _index_modes(self, names, column, options):
        """Whether each index is fitted as a normalised spectrum (not as EW).

        ``column`` are the values of the ``mode`` column of ``ew_list``, if it
        has one. The modes come from ``fit_mode`` and the column, which must
        agree; an index with neither is fitted as EW. Also decides whether the
        likelihood is split in two terms (``split_terms``): it is when the
        modes come from a list, or from the column without an explicit
        ``fit_mode``.
        """
        fit_mode = self.config["fit_mode"]
        explicit = self.config["fit_mode_given"]
        # Modes set by fit_mode
        given = np.zeros(len(names), dtype=bool)
        spectra = np.zeros(len(names), dtype=bool)
        if isinstance(fit_mode, str):
            given[:] = explicit
            spectra[:] = fit_mode == "spectra"
        elif isinstance(fit_mode, list):
            if len(fit_mode) != len(names):
                raise ValueError(f"fit_mode has {len(fit_mode)} modes for "
                                 f"{len(names)} indices: {list(names)}")
            given[:] = True
            spectra[:] = [mode == "spectra" for mode in fit_mode]
        else:
            unknown = sorted(set(fit_mode) - set(names))
            if unknown:
                raise ValueError(f"fit_mode has unknown indices: {unknown}")
            for ith, name in enumerate(names):
                given[ith] = name in fit_mode
                spectra[ith] = fit_mode.get(name) == "spectra"
        if column is not None:
            from_column = np.array([mode == "spectra" for mode in column])
            conflict = given & (spectra != from_column)
            if conflict.any():
                raise ValueError(
                    "fit_mode and the mode column of ew_list disagree for "
                    f"{list(np.array(names)[conflict])}")
            spectra = np.where(given, spectra, from_column)
        self.config["split_terms"] = (
            not isinstance(fit_mode, str) or (column is not None and not explicit))
        return spectra

    def prepare_likelihood_terms(self, options):
        """Set the likelihood terms written to the data block.

        A single mode has one term, ``<like_name>``. When the modes come from
        a list or the ``mode`` column there are two, ``<base>_ew_like`` and
        ``<base>_spectra_like``, where ``<base>`` is ``like_name`` without its
        ``_like`` suffix, with optional weights.
        """
        if not self.config["split_terms"]:
            self.config["like_terms"] = (self.config["fit_mode"],)
            self.config["like_keys"] = (self.like_name,)
            self.config["like_weights"] = (1.0,)
            return
        base = self.like_name[:-len("_like")]
        self.config["like_terms"] = ("ew", "spectra")
        self.config["like_keys"] = (base + "_ew_like", base + "_spectra_like")
        self.config["like_weights"] = (
            options.get_double("ew_like_weight", default=1.0),
            options.get_double("spectra_like_weight", default=1.0))
        logger.info("Two likelihood terms: the pipeline must list 'likelihoods = %s'",
                    " ".join(key[:-len("_like")] for key in self.config["like_keys"]))

    def prepare_equivalent_widths(self, options):
        """Measure the observed equivalent widths and select the usable ones."""

        modes = None
        if options.has_value("ew_list"):
            # load from a list containing 7 columns, plus an optional mode
            fields = [("name", "U40"), ("bl", "f4"), ("br", "f4"),
                      ("cl", "f4"), ("cr", "f4"), ("rl", "f4"), ("rr", "f4")]
            with open(options["ew_list"]) as f:
                rows = (line.split("#")[0].split() for line in f)
                first = next((row for row in rows if row), [])
            if len(first) == 8:
                fields.append(("mode", "U10"))
            ew_list = np.loadtxt(options["ew_list"], dtype=fields, ndmin=1)
            if "mode" in ew_list.dtype.names:
                modes = [str(mode).strip().lower() for mode in ew_list["mode"]]
                for mode in modes:
                    if mode not in {"ew", "spectra"}:
                        raise ValueError(
                            f"Unknown index mode {mode!r}; expected ew or spectra.")
            ews = []
            for name, bl, br, cl, cr, rl, rr in zip(
                ew_list["name"],
                ew_list["bl"], ew_list["br"],
                ew_list["cl"], ew_list["cr"],
                ew_list["rl"], ew_list["rr"]):
                ew = EquivalentWidth(left_wl_range=[bl, br],
                                     central_wl_range=[cl, cr],
                                     right_wl_range=[rl, rr],
                                     name=name)
                ews.append(ew)

        elif options.has_value("ew_names"):
            names = options["ew_names"]
            if isinstance(names, str):
                names = names.split(",")

            ews = []
            for name in names:
                if str(name).strip():
                    ew = EquivalentWidth.from_name(str(name).strip())
                    # Indices loaded from JSON files carry no name
                    ew.name = str(name).strip()
                    ews.append(ew)

        is_spectra = self._index_modes([ew.name for ew in ews], modes, options)

        wl = self.config["wavelength"].value
        bad_pixel = (self.config["weights"] <= 0) | ~np.isfinite(self.config["flux"])

        selected, selected_spectra = [], []
        for ew, spectra in zip(ews, is_spectra):
            windows = (ew.left_wl_range.value, ew.central_wl_range.value,
                       ew.right_wl_range.value)
            low, high = windows[0][0], windows[2][1]
            # Fitted as spectra only the side windows (continuum) must be clean
            checked = (windows[0], windows[2]) if spectra else [(low, high)]
            if low < wl[0] or high > wl[-1]:
                logger.warning("Dropping index %s: outside the observed range", ew.name)
            elif any(bad_pixel[(wl >= a) & (wl <= b)].any() for a, b in checked):
                logger.warning("Dropping index %s: it includes masked pixels", ew.name)
            elif any(not ((wl >= a) & (wl <= b)).any() for a, b in windows):
                logger.warning("Dropping index %s: a window has no pixels", ew.name)
            elif spectra and not (
                    (wl >= windows[1][0]) & (wl <= windows[1][1])
                    & ~bad_pixel).any():
                logger.warning("Dropping index %s: no valid central pixels", ew.name)
            else:
                selected.append(ew)
                selected_spectra.append(spectra)
        if not selected:
            raise ValueError("None of the requested equivalent widths can be measured.")

        # Store the indices that will be used. Dropping spectra indices keeps
        # the order of the EW ones, so their observables stay valid.
        self._set_indices(selected, selected_spectra)
        self._prepare_observed_ews(options)
        self._prepare_observed_spectra_windows(options)
        self.config["n_ew_obs"] = self.config["obs_ew"].size
        self.config["obs_vector"] = np.concatenate(
            [self.config["obs_ew"], self.config["obs_spec"]])
        self.config["obs_ivar"] = np.concatenate(
            [self.config["ivar_ew"], self.config["ivar_spec"]])

        # Model window means: on the SSP grid, or on the observed grid if the
        # SED is trimmed after the LOSVD convolution
        ssp_wl = self.config["ssp_model"].wavelength.to_value("Angstrom")
        self.config["ssp_wl"] = ssp_wl
        self.config["ew_matrix"] = self._window_matrix(
            ssp_wl, self.config["ew_ranges"])
        self.config["ew_matrix_trimmed"] = self._window_matrix(
            wl, self.config["ew_ranges"])
        logger.info("Fitting %d indices (%d as EW and %d as spectra): %s",
                    len(self.config["ew_names"]), self.config["ew_rows"].size,
                    self.config["spectra_rows"].size, self.config["ew_names"])

    def _prepare_observed_ews(self, options):
        """Measure the observed EWs of the indices fitted as EW."""
        rows = self.config["ew_rows"]
        if rows.size == 0:
            self.config["obs_ew"] = np.empty(0)
            self.config["ivar_ew"] = np.empty(0)
            return
        wl = self.config["wavelength"].value
        # Compute a matrix for estimating mean fluxes across all wavelength ranges
        matrix = self._window_matrix(wl, self.config["ew_ranges"])
        means = matrix @ self.config["flux"]
        var_means = (matrix ** 2) @ self.config["var"]
        ew_obs = self._ew_from_means(means, rows)
        ew_err = np.hypot(self._ew_error_from_means(means, var_means, rows),
                          # This can be relevant is the EW is close to 0
                          options.get_double("ew_error_floor", default=0.0))

        good = np.isfinite(ew_obs) & np.isfinite(ew_err) & (ew_err > 0)
        if not good.all():
            logger.warning("Dropping indices with undefined EW or error: %s",
                           [self.config["ew_names"][row] for row in rows[~good]])
            keep = np.ones(len(self.config["ew_list"]), dtype=bool)
            keep[rows[~good]] = False
            if not keep.any():
                raise ValueError("No equivalent width has a finite value and error.")
            self._keep_indices(keep)

        self.config["obs_ew"] = ew_obs[good]
        self.config["ivar_ew"] = 1.0 / ew_err[good] ** 2

    def _observed_pixels(self):
        """Window means of the data and pseudo-continuum at the fitted pixels."""
        self._set_pixels()
        wl = self.config["wavelength"].value
        means = self._window_matrix(wl, self.config["ew_ranges"]) @ self.config["flux"]
        return means, self._continuum_pixels(means)

    def _prepare_observed_spectra_windows(self, options):
        """Normalise the observed central windows of the spectra indices."""
        means, continuum = self._observed_pixels()
        # An index needs a positive continuum in all its pixels
        spectra = self.config["ew_is_spectra"]
        owner = self.config["pix_owner"]
        n_pix = np.bincount(owner, minlength=spectra.size)
        n_bad = np.bincount(owner, weights=~(continuum > 0), minlength=spectra.size)
        good = ~spectra | ((n_pix > 0) & (n_bad == 0))
        if not good.all():
            logger.warning("Dropping indices with a non-positive continuum: %s",
                           [n for n, g in zip(self.config["ew_names"], good) if not g])
            if not good.any():
                raise ValueError("No index has a positive continuum.")
            self._keep_indices(good)
            means, continuum = self._observed_pixels()

        pix = self.config["pix_idx"]
        flux, var = self.config["flux"], self.config["var"]
        floor = options.get_double("spectra_error_floor", default=0.0)
        self.config["obs_spec"] = flux[pix] / continuum
        self.config["ivar_spec"] = 1.0 / (var[pix] / continuum ** 2 + floor ** 2)
        self.config["obs_means"] = means

    def prepare_balmer_emission(self, options):
        """Configure the optional Balmer emission of the model."""
        mode = options.get_string("balmer_emission", default="none")
        mode = mode.strip().strip("\"'").strip().lower()
        if mode not in {"none", "ionising", "kennicutt"}:
            raise ValueError(
                f"Unknown balmer_emission={mode!r}; expected none, ionising or kennicutt.")
        self.config["balmer_emission"] = None if mode == "none" else mode
        if mode == "none":
            return

        if mode == "ionising":
            if self.config["ssp_model"].log_ionising_HI_photons is None:
                raise ValueError(
                    "The SSP model does not provide ionising photon rates; "
                    "use balmer_emission = kennicutt.")
        else:
            # Default is Kennicutt (1998)
            tau = options.get_double("balmer_tau", default=20.0) << u.Myr
            if tau > self.config["sfh_model"].today:
                raise ValueError("balmer_tau cannot exceed the age of the Universe "
                                 "at the source.")
            self.config["balmer_tau"] = tau
            ssp_imf = self.config["ssp_model"].imf
            if "cha" in ssp_imf.lower():
                ssp_imf = "chabrier"
            elif "kro" in ssp_imf.lower():
                ssp_imf = "kroupa"
            elif "sal" in ssp_imf.lower():
                ssp_imf = "salpeter"
            else:
                raise ValueError(
                    f"Unknown SSP IMF {ssp_imf!r}; expected Salpeter, Kroupa or Chabrier.")
            self.config["LHa_per_SFR"] = _LHA_PER_SFR[ssp_imf.lower()]
            logger.info("Using Balmer emission from SFR averaged over %s, "
                        "IMF %s", tau, ssp_imf)

        # Fraction of each line inside each window, per Angstrom of window
        sigma = options.get_double("balmer_sigma", default=2.0)
        self.config["balmer_sigma"] = sigma
        low, high = self.config["ew_ranges"].T
        fraction = (ndtr((high[:, None] - _BALMER_WAVELENGTH) / sigma)
                    - ndtr((low[:, None] - _BALMER_WAVELENGTH) / sigma))
        self.config["balmer_matrix"] = fraction / (high - low)[:, None]
        if self.config["spectra_rows"].size:
            # Same, for every fitted pixel (mean flux density within the pixel)
            half_pixel = 0.5 * self.config["velscale"] / _SPEED_OF_LIGHT.to_value("km/s")
            pix_low = self.config["pix_wl"] * np.exp(-half_pixel)
            pix_high = self.config["pix_wl"] * np.exp(half_pixel)
            fraction = (ndtr((pix_high[:, None] - _BALMER_WAVELENGTH) / sigma)
                        - ndtr((pix_low[:, None] - _BALMER_WAVELENGTH) / sigma))
            self.config["balmer_pix_matrix"] = fraction / (pix_high - pix_low)[:, None]
        logger.info("Adding Balmer emission to the model (%s)", mode)

    def balmer_luminosities(self, block):
        """Attenuated Balmer line luminosities, in units of 1e-16 erg/s.

        The SFR and the ionising photon rate come from the same SFH model as
        the SED, so the line to continuum ratio does not depend on the mass
        normalization.
        """
        mode = self.config["balmer_emission"]
        if mode is None:
            return None
        sfh_model = self.config["sfh_model"]
        today = sfh_model.today
        if mode == "ionising":
            log_q = sfh_model.model.ionising_photon_rate_hi(
                self.config["ssp_model"], today).to_value(u.dex(u.s ** -1))
            l_halpha = _LHA_PER_Q * 10 ** log_q
        else:
            tau = self.config["balmer_tau"]
            times = u.Quantity([(today - tau).to_value(u.Gyr),
                                today.to_value(u.Gyr)], u.Gyr)
            mass = sfh_model.model.stellar_mass_formed(times)
            sfr = (mass[1] - mass[0]).to_value(u.Msun) / tau.to_value(u.yr)
            l_halpha = self.config["LHa_per_SFR"] * max(sfr, 0.0)
        lines = _LUMINOSITY_PER_ERG_S * l_halpha * _BALMER_RATIO

        dust_model, a_v = self._get_dust_for_block(block)
        if dust_model is not None:
            lines = dust_model.apply_extinction(
                _BALMER_WAVELENGTH << u.Angstrom, lines, a_v=a_v)
        return lines

    def _get_dust_for_block(self, block):
        """Dust model and ``a_v`` of the current sample (``None`` if no dust)."""
        dust_model = self.config["extinction_law"]
        if dust_model is None:
            return None, None
        if not block.has_value("dust.attenuation", "a_v"):
            raise ValueError(
                "Dust attenuation parameter 'a_v' is missing in the data block.")
        if block.has_value("dust.attenuation", "r"):
            dust_model = self._get_dust_model(
                dust_model.extinction_law_name, block["dust.attenuation", "r"])
        return dust_model, block["dust.attenuation", "a_v"]

    def _model_grid(self, sed):
        """Wavelength grid and window matrix matching a model SED.

        The SED is on the SSP grid, or on the observed grid if it was trimmed
        after the LOSVD convolution.
        """
        if sed.size == self.config["ew_matrix"].shape[1]:
            return self.config["ssp_wl"], self.config["ew_matrix"]
        return (self.config["wavelength"].value,
                self.config["ew_matrix_trimmed"])

    def model_sed(self, block, parse=False):
        """Stellar SED on the SSP grid (1e-16 erg/s/AA), including dust."""
        sfh_model = self.config["sfh_model"]
        if parse:
            sfh_model.parse_datablock(block)
        ssp_model = self.config["ssp_model"]
        sed = self.luminosity_values(sfh_model.model.compute_SED(
            ssp_model, t_obs=sfh_model.today, allow_negative=False))
        dust_model, a_v = self._get_dust_for_block(block)
        if dust_model is not None:
            sed = dust_model.apply_extinction(ssp_model.wavelength, sed, a_v=a_v)
        # Include kinematics if the block has a kinematics section
        # This is particularly relevant when combining with other modules
        if block.has_section("kinematics"):
            self.config["losvd_kernel"].parse_parameters(block)
            sed = self.convolve_losvd_and_trim(sed)
        elif self.config["spectra_rows"].size:
            raise ValueError(
                "Fitting indices as spectra needs a 'kinematics' section in "
                "the parameters to broaden the model and match the observed pixels.")
        return sed

    def make_observable(self, block, parse=False):
        """Compute the model observable.

        The equivalent widths in Angstrom of the indices fitted as EW,
        followed by the normalised flux at every pixel fitted as spectra
        (``obs_vector`` has the same layout).
        """
        sed = self.model_sed(block, parse=parse)
        means = self._model_grid(sed)[1] @ sed
        lines = self.balmer_luminosities(block)
        if lines is not None:
            means = means + self.config["balmer_matrix"] @ lines

        # Save SFH mass-fraction times and sSFRs (one mass-history evaluation)
        self.save_sfh_extras(block, self.config["sfh_model"])

        ew_model = self._ew_from_means(means, self.config["ew_rows"])
        flux = sed[self.config["pix_idx"]]
        if lines is not None:
            flux = flux + self.config["balmer_pix_matrix"] @ lines
        return np.concatenate([ew_model, flux / self._continuum_pixels(means)])

    def _term_likelihoods(self, model):
        """Weighted log-likelihood of each term (see ``like_terms``)."""
        n_ew = self.config["n_ew_obs"]
        parts = {"ew": slice(None, n_ew), "spectra": slice(n_ew, None)}
        terms = []
        for kind, weight in zip(self.config["like_terms"], self.config["like_weights"]):
            part = parts[kind]
            obs, ivar = self.config["obs_vector"][part], self.config["obs_ivar"][part]
            like = self.log_like(obs, model[part], ivar, include_norm=True) \
                if obs.size else 0.0
            terms.append(weight * like)
        return terms

    def _write_likelihoods(self, block, values):
        """Write the likelihood terms, in the order of ``like_keys``."""
        for key, value in zip(self.config["like_keys"], values):
            block[section_names.likelihoods, key] = value

    def execute(self, block):
        """Function executed by sampler.

        The likelihood resulting from this function is the evidence on the
        basis of which the parameter space is sampled. With a list of modes
        it is split in two terms, one for the EWs and one for the normalised
        spectra, that the sampler adds up. Every term must be written for
        every sample, so the prior penalty goes to the first one and the
        others are set to 0 for invalid samples.
        """
        n_terms = len(self.config["like_keys"])
        valid, prior_penalty = self.config["sfh_model"].parse_datablock(block)
        if prior_penalty is None:
            prior_penalty = 0.0
        if not valid:
            logger.debug("Invalid sample: %s", block)
            self._write_likelihoods(block, [prior_penalty] + [0.0] * (n_terms - 1))
            return 0

        model = self.make_observable(block)
        if not np.isfinite(model).all():
            logger.debug("Non-finite model observables: %s", model)
            self._write_likelihoods(block, [-1e20] + [0.0] * (n_terms - 1))
            return 0

        terms = self._term_likelihoods(model)
        self._write_likelihoods(block, [terms[0] + prior_penalty] + terms[1:])

        if self.config.get("save_chi2", False):
            for key, term in zip(self.config["like_keys"], terms):
                block["extra", key + "_chi2"] = -2 * term
        return 0

    def plot_solution(self, block, figname=None, ncols=3):
        """Plot the observed and model indices.

        Each panel shows one index. Indices fitted as EW: the observed
        spectrum, the model spectrum scaled to the data over the index
        windows, and the left, central and right windows, with the observed
        and model EWs in the annotation. Indices fitted as spectra: the
        observed and model spectra normalised by their pseudo-continuum, with
        the fitted pixels highlighted and the chi2 of the index in the
        annotation.

        Parameters
        ----------
        block : :class:`cosmosis.DataBlock`
            Solution to plot.
        figname : str, optional
            If given, the figure is saved at this path.
        ncols : int, optional
            Maximum number of panels per row.

        Returns
        -------
        fig : :class:`matplotlib.figure.Figure`
        solution_data : dict
            For the EW indices: names, observed EWs and errors, model EWs and
            residuals in units of the error. For the spectra indices: for
            every fitted pixel, the index name and number, wavelength,
            observed and model normalised flux, error and residual in units
            of the error. In ``ew`` and ``spectra`` modes it is the dictionary
            of that kind; with two likelihood terms, a dictionary with both
            under the keys ``ew`` and ``spectra``.
        """
        model = self.make_observable(block, parse=True)
        sed = self.model_sed(block)
        model_wl = self._model_grid(sed)[0]
        lines = self.balmer_luminosities(block)
        if lines is not None:
            sigma = self.config["balmer_sigma"]
            sed = sed + np.sum(
                lines / (sigma * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((model_wl[:, None] - _BALMER_WAVELENGTH) / sigma) ** 2),
                axis=1)
        names = np.array(self.config["ew_names"])
        is_spectra = self.config["ew_is_spectra"]
        ew_rows, spectra_rows = self.config["ew_rows"], self.config["spectra_rows"]
        # The observables are the EWs followed by the fitted pixels
        n_ew = self.config["n_ew_obs"]
        obs = self.config["obs_vector"]
        err = self.config["obs_ivar"] ** -0.5
        chi = (model - obs) / err
        terms = self._term_likelihoods(model)
        loglike = sum(terms)

        wl = self.config["wavelength"].value
        flux = self.config["flux"]
        flux_err = np.sqrt(self.config["var"])

        n_index = names.size
        owner = self.config["pix_owner"]
        # Position of each EW index among the EW observables
        ew_pos = np.cumsum(~is_spectra) - 1
        if spectra_rows.size:
            pix_chi2 = np.bincount(owner, weights=chi[n_ew:] ** 2, minlength=n_index)
            pix_count = np.bincount(owner, minlength=n_index)
            ranges = self.config["ew_ranges"].reshape(-1, 3, 2)
            mid = ranges.mean(axis=2)
            window_matrix = self.config["ew_matrix_trimmed"]
            means_obs = window_matrix @ flux
            means_model = window_matrix @ np.interp(wl, model_wl, sed)
        summary = []
        if ew_rows.size:
            kind = "" if not spectra_rows.size else " (EW)"
            summary.append(f"chi2{kind} = {np.sum(chi[:n_ew] ** 2):.1f} "
                           f"({ew_rows.size} indices)")
        if spectra_rows.size:
            kind = "" if not ew_rows.size else " (spectra)"
            summary.append(f"chi2{kind} = {np.sum(chi[n_ew:] ** 2):.1f} "
                           f"({owner.size} pixels, {spectra_rows.size} indices)")
        title = ", ".join(summary)
        ncols = min(ncols, n_index)
        nrows = int(np.ceil(n_index / ncols))
        # Indices on the left, solution parameters and fit summary on the right
        fig = plt.figure(figsize=(5 * ncols + 4, 3.5 * nrows),
                         constrained_layout=True)
        panels, info = fig.subfigures(1, 2, width_ratios=[5 * ncols, 4])
        axs = panels.subplots(nrows, ncols, squeeze=False)
        panels.suptitle(f"Module: {self.name}   {title}   "
                        f"log-likelihood = {loglike:.1f}")

        fit_info = {"Indices (EW / spectra)":
                    f"{ew_rows.size} / {spectra_rows.size}"}
        if spectra_rows.size:
            fit_info["Fitted pixels"] = owner.size
        if self.config["split_terms"]:
            fit_info["ln L (EW)"], fit_info["ln L (spectra)"] = terms
        fit_info["ln L"] = loglike
        ax_info = info.subplots()
        draw_dict_in_axes(
            ax_info, list(self.solution_sections(block).items()) + [("Fit", fit_info)],
            section_spacing=1, title_style="underline")
        ax_info.axis("off")

        for ith, (ax, ew) in enumerate(zip(axs.flat, self.config["ew_list"])):
            left, centre, right = (ew.left_wl_range.value,
                                   ew.central_wl_range.value,
                                   ew.right_wl_range.value)
            pad = 0.1 * (right[1] - left[0])
            region = (wl >= left[0] - pad) & (wl <= right[1] + pad)
            window = (wl >= left[0]) & (wl <= right[1])
            model_on_obs = np.interp(wl, model_wl, sed)

            if is_spectra[ith]:
                # Both spectra normalised by their own pseudo-continuum
                t_line = (wl[region] - mid[ith, 0]) / (mid[ith, 2] - mid[ith, 0])
                cont_obs = ((1 - t_line) * means_obs[3 * ith]
                            + t_line * means_obs[3 * ith + 2])
                cont_model = ((1 - t_line) * means_model[3 * ith]
                              + t_line * means_model[3 * ith + 2])
                norm_obs = flux[region] / cont_obs
                norm_err = flux_err[region] / cont_obs
                ax.fill_between(wl[region], norm_obs - norm_err,
                                norm_obs + norm_err, color="k", alpha=0.3)
                ax.plot(wl[region], norm_obs, c="k", lw=0.7, label="Observed")
                ax.plot(wl[region], model_on_obs[region] / cont_model, c="b",
                        lw=0.7, label="Model")
                fitted = owner == ith
                ax.plot(self.config["pix_wl"][fitted], obs[n_ew:][fitted], "k.", ms=3)
                ax.plot(self.config["pix_wl"][fitted], model[n_ew:][fitted], "b.", ms=3)
                text = (f"{names[ith]}\n"
                        f"chi2 / N = {pix_chi2[ith]:.1f} / {pix_count[ith]}")
                ylabel = "Normalised flux"
            else:
                # Scale the model to the data over the index windows
                pos = ew_pos[ith]
                scale = ml_amplitude(flux[window], model_on_obs[window],
                                     self.config["ivar"][window])
                ax.fill_between(wl[region], (flux - flux_err)[region],
                                (flux + flux_err)[region], color="k", alpha=0.3)
                ax.plot(wl[region], flux[region], c="k", lw=0.7, label="Observed")
                ax.plot(wl[region], scale * model_on_obs[region], c="b", lw=0.7,
                        label="Model")
                text = (f"{names[ith]}\n"
                        f"obs: {obs[pos]:.2f} $\\pm$ {err[pos]:.2f} $\\AA$\n"
                        f"model: {model[pos]:.2f} $\\AA$ ($\\Delta$ = {chi[pos]:.1f}$\\sigma$)")
                ylabel = "Flux"
            for span, color in ((left, "tab:blue"), (centre, "tab:green"),
                                (right, "tab:red")):
                ax.axvspan(*span, color=color, alpha=0.2)
            ax.set_xlim(wl[region][[0, -1]])
            ax.annotate(text, xy=(0.03, 0.97), xycoords="axes fraction",
                        va="top", fontsize=8,
                        bbox=dict(boxstyle="round", fc="w", alpha=0.8))
            ax.set_ylabel(ylabel)
            if ith == 0:
                ax.legend(loc="lower right", fontsize=7)
        for ax in axs.flat[n_index:]:
            ax.axis("off")
        for ax in axs[-1]:
            ax.set_xlabel("Wavelength (AA)")

        if figname is not None:
            fig.savefig(figname, bbox_inches="tight", dpi=200)
            logger.info("Fit plot saved at: %s", figname)

        ew_data = {
            "name": names[ew_rows],
            "ew_obs": obs[:n_ew],
            "ew_err": err[:n_ew],
            "ew_model": model[:n_ew],
            "chi": chi[:n_ew],
        }
        spectra_data = {
            "name": names[owner],
            "index": owner,
            "wavelength": self.config["pix_wl"],
            "obs": obs[n_ew:],
            "obs_err": err[n_ew:],
            "model": model[n_ew:],
            "chi": chi[n_ew:],
        }
        if self.config["split_terms"]:
            solution_data = {"ew": ew_data, "spectra": spectra_data}
        else:
            solution_data = ew_data if self.config["fit_mode"] == "ew" else spectra_data
        plt.close(fig)
        return fig, solution_data

    def cleanup(self):
        """Release resources after an equivalent width fit run."""
        pass


def setup(options):
    """Create the CosmoSIS-facing module instance."""

    options = SectionOptions(options)
    mod = EquivalentWidthFitModule(options)
    return mod


def execute(block, mod):
    """Run one likelihood evaluation for the configured module."""

    mod.execute(block)
    return 0


def cleanup(mod):
    """Release module resources after sampling."""

    mod.cleanup()

module = EquivalentWidthFitModule
