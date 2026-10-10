"""Equivalent width fitting pipeline module."""

from besta.pipeline_modules.base_module import SpectraFitModule, ml_amplitude
import numpy as np
from astropy import units as u
from matplotlib import pyplot as plt
from scipy.special import ndtr

from cosmosis.datablock import names as section_names
from cosmosis.datablock import SectionOptions
from besta.logging import get_logger
from pst.observables import EquivalentWidth

logger = get_logger(__name__)

# Balmer lines: rest-frame (air) wavelength in Angstrom and flux relative to
# H-alpha for Case B recombination (T = 1e4 K, n_e = 100 cm^-3)
_BALMER_WAVELENGTH = np.array([6562.80, 4861.33, 4340.47, 4101.74, 3970.07])
_BALMER_RATIO = np.array([1.0, 1.0 / 2.86, 0.468 / 2.86, 0.259 / 2.86,
                          0.159 / 2.86])
# Kennicutt (1998): L(Halpha) [erg/s] = SFR [Msun/yr] / 7.9e-42 (Salpeter IMF)
_LHA_PER_SFR = 1.0 / 7.9e-42
# Case B: L(Halpha) [erg/s] = 1.37e-12 * Q(H) [photons/s]
_LHA_PER_Q = 1.37e-12
# erg/s -> luminosity units of BaseModule.luminosity_values (1e-16 erg/s)
_LUMINOSITY_PER_ERG_S = 1e16


class EquivalentWidthFitModule(SpectraFitModule):
    """Fit stellar populations to a set of spectral equivalent widths.

    The equivalent widths are measured once on the observed spectrum and then
    compared with those of the model spectrum at every sample. To be fast, the
    EW of each index is estimated from the mean flux in its three windows,

    ``EW = (1 - F_C / F_cont) * (lambda_C,max - lambda_C,min)``,

    where ``F_C`` is the mean flux in the central window and ``F_cont`` is the
    linear interpolation of the mean fluxes of the side windows at the centre
    of the central window. Indices that are not fully covered by the observed
    spectrum or that touch masked pixels are discarded.

    Module options (on top of the ones needed to read the spectrum and set up
    the SSP, SFH and dust models):

    - ``ew_names``: comma-separated names of the indices, as understood by
      :meth:`pst.observables.EquivalentWidth.from_name`.
    - ``ew_error_floor`` *(optional, default 0)*: error (in Angstrom) added in
      quadrature to the measured uncertainty of each index, to account for
      model systematics.
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
        if not options.has_value("ew_names"):
            raise ValueError("Option 'ew_names' is required.")

        self.prepare_observed_spectra(options)
        self.prepare_ssp_model(options)
        self.prepare_sfh_model(options)
        self.prepare_extinction_law(options)
        self.prepare_equivalent_widths(options)
        self.prepare_balmer_emission(options)

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

    def _ew_from_means(self, means):
        """EW (Angstrom) from the window means (left, central, right)."""
        left, centre, right = means.reshape(-1, 3).T
        t = self.config["ew_t"]
        continuum = (1 - t) * left + t * right
        return (1 - centre / continuum) * self.config["ew_dlam"]

    def _ew_error_from_means(self, means, var_means):
        """Propagated error (Angstrom) of :meth:`_ew_from_means`."""
        left, centre, right = means.reshape(-1, 3).T
        var_left, var_centre, var_right = var_means.reshape(-1, 3).T
        t, dlam = self.config["ew_t"], self.config["ew_dlam"]
        continuum = (1 - t) * left + t * right
        d_centre = dlam / continuum
        d_left = d_centre * centre * (1 - t) / continuum
        d_right = d_centre * centre * t / continuum
        return np.sqrt(d_centre ** 2 * var_centre + d_left ** 2 * var_left
                       + d_right ** 2 * var_right)

    def _set_indices(self, indices):
        """Store the indices and the quantities derived from their windows."""
        # (n_index, window, [min, max]) with windows ordered left, centre, right
        ranges = np.array([[ew.left_wl_range.value, ew.central_wl_range.value,
                            ew.right_wl_range.value] for ew in indices])
        mid = ranges.mean(axis=2)
        self.config["ew_list"] = indices
        self.config["ew_names"] = [ew.name for ew in indices]
        self.config["ew_ranges"] = ranges.reshape(-1, 2)
        self.config["ew_t"] = (mid[:, 1] - mid[:, 0]) / (mid[:, 2] - mid[:, 0])
        self.config["ew_dlam"] = ranges[:, 1, 1] - ranges[:, 1, 0]

    def prepare_equivalent_widths(self, options):
        """Measure the observed equivalent widths and select the usable ones."""
        names = options["ew_names"]
        if isinstance(names, str):
            names = names.split(",")
        names = [str(name).strip() for name in names if str(name).strip()]

        wl = self.config["wavelength"].value
        bad_pixel = (self.config["weights"] <= 0) | ~np.isfinite(self.config["flux"])

        selected = []
        for name in names:
            ew = EquivalentWidth.from_name(name)
            ew.name = name
            windows = (ew.left_wl_range.value, ew.central_wl_range.value,
                       ew.right_wl_range.value)
            low, high = windows[0][0], windows[2][1]
            if low < wl[0] or high > wl[-1]:
                logger.warning("Dropping index %s: outside the observed range", name)
            elif bad_pixel[(wl >= low) & (wl <= high)].any():
                logger.warning("Dropping index %s: it includes masked pixels", name)
            elif any(not ((wl >= a) & (wl <= b)).any() for a, b in windows):
                logger.warning("Dropping index %s: a window has no pixels", name)
            else:
                selected.append(ew)
        if not selected:
            raise ValueError("None of the requested equivalent widths can be measured.")

        self._set_indices(selected)
        matrix = self._window_matrix(wl, self.config["ew_ranges"])
        means = matrix @ self.config["flux"]
        var_means = (matrix ** 2) @ self.config["var"]
        ew_obs = self._ew_from_means(means)
        ew_err = np.hypot(self._ew_error_from_means(means, var_means),
                          options.get_double("ew_error_floor", default=0.0))

        good = np.isfinite(ew_obs) & np.isfinite(ew_err) & (ew_err > 0)
        if not good.all():
            logger.warning("Dropping indices with undefined EW or error: %s",
                           [n for n, g in zip(self.config["ew_names"], good) if not g])
        if not good.any():
            raise ValueError("No equivalent width has a finite value and error.")
        if not good.all():
            self._set_indices([ew for ew, g in zip(selected, good) if g])

        self.config["ew_obs"] = ew_obs[good]
        self.config["ew_obs_ivar"] = 1.0 / ew_err[good] ** 2
        # Model window means are taken on the SSP grid
        ssp_wl = self.config["ssp_model"].wavelength.to_value("Angstrom")
        self.config["ew_matrix"] = self._window_matrix(
            ssp_wl, self.config["ew_ranges"])
        logger.info("Fitting %d equivalent widths: %s",
                    len(self.config["ew_names"]), self.config["ew_names"])

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
            tau = options.get_double("balmer_tau", default=10.0) << u.Myr
            if tau > self.config["sfh_model"].today:
                raise ValueError("balmer_tau cannot exceed the age of the Universe "
                                 "at the source.")
            self.config["balmer_tau"] = tau

        # Fraction of each line inside each window, per Angstrom of window
        sigma = options.get_double("balmer_sigma", default=2.0)
        self.config["balmer_sigma"] = sigma
        low, high = self.config["ew_ranges"].T
        fraction = (ndtr((high[:, None] - _BALMER_WAVELENGTH) / sigma)
                    - ndtr((low[:, None] - _BALMER_WAVELENGTH) / sigma))
        self.config["balmer_matrix"] = fraction / (high - low)[:, None]
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
            l_halpha = _LHA_PER_SFR * max(sfr, 0.0)
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
                dust_model.name, block["dust.attenuation", "r"])
        return dust_model, block["dust.attenuation", "a_v"]

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
        return sed

    def make_observable(self, block, parse=False):
        """Compute the model equivalent widths (Angstrom)."""
        means = self.config["ew_matrix"] @ self.model_sed(block, parse=parse)
        lines = self.balmer_luminosities(block)
        if lines is not None:
            means = means + self.config["balmer_matrix"] @ lines

        # Save SFH mass-fraction times and sSFRs (one mass-history evaluation)
        self.save_sfh_extras(block, self.config["sfh_model"])

        return self._ew_from_means(means)

    def execute(self, block):
        """Function executed by sampler.

        The likelihood resulting from this function is the evidence on the
        basis of which the parameter space is sampled.
        """
        valid, prior_penalty = self.config["sfh_model"].parse_datablock(block)
        if prior_penalty is None:
            prior_penalty = 0.0
        if not valid:
            logger.debug("Invalid sample: %s", block)
            block[section_names.likelihoods, self.like_name] = prior_penalty
            return 0

        ew_model = self.make_observable(block)
        if not np.isfinite(ew_model).all():
            logger.debug("Non-finite model equivalent widths: %s", ew_model)
            block[section_names.likelihoods, self.like_name] = -1e20
            return 0

        like = self.log_like(self.config["ew_obs"],
                             ew_model,
                             self.config["ew_obs_ivar"],
                             include_norm=True)
        block[section_names.likelihoods, self.like_name] = like + prior_penalty

        if self.config.get("save_chi2", False):
            block["extra", self.like_name + "_chi2"] = -2 * like
        return 0

    def plot_solution(self, block, figname=None, ncols=3):
        """Plot the observed and model equivalent widths.

        Each panel shows one index: the observed spectrum, the model spectrum
        scaled to the data over the index windows, and the left, central and
        right windows. The annotation compares the observed and model EWs.

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
            Names, observed EWs and errors, model EWs and residuals in units
            of the error.
        """
        ssp_wl = self.config["ssp_model"].wavelength.to_value("Angstrom")
        ew_model = self.make_observable(block, parse=True)
        sed = self.model_sed(block)
        lines = self.balmer_luminosities(block)
        if lines is not None:
            sigma = self.config["balmer_sigma"]
            sed = sed + np.sum(
                lines / (sigma * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((ssp_wl[:, None] - _BALMER_WAVELENGTH) / sigma) ** 2),
                axis=1)
        names = np.array(self.config["ew_names"])
        ew_obs = self.config["ew_obs"]
        ew_err = self.config["ew_obs_ivar"] ** -0.5
        chi = (ew_model - ew_obs) / ew_err
        loglike = self.log_like(ew_obs, ew_model, self.config["ew_obs_ivar"],
                                include_norm=True)

        wl = self.config["wavelength"].value
        flux = self.config["flux"]
        flux_err = np.sqrt(self.config["var"])

        n_ew = names.size
        ncols = min(ncols, n_ew)
        nrows = int(np.ceil(n_ew / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
                                squeeze=False, constrained_layout=True)
        fig.suptitle(f"Module: {self.name}   "
                     f"chi2 = {np.sum(chi ** 2):.1f} ({n_ew} indices)   "
                     f"log-likelihood = {loglike:.1f}")

        for ith, (ax, ew) in enumerate(zip(axs.flat, self.config["ew_list"])):
            left, centre, right = (ew.left_wl_range.value,
                                   ew.central_wl_range.value,
                                   ew.right_wl_range.value)
            pad = 0.1 * (right[1] - left[0])
            region = (wl >= left[0] - pad) & (wl <= right[1] + pad)
            window = (wl >= left[0]) & (wl <= right[1])

            # Scale the model to the data over the index windows
            model_on_obs = np.interp(wl, ssp_wl, sed)
            scale = ml_amplitude(flux[window], model_on_obs[window],
                                 self.config["ivar"][window])

            ax.fill_between(wl[region], (flux - flux_err)[region],
                            (flux + flux_err)[region], color="k", alpha=0.3)
            ax.plot(wl[region], flux[region], c="k", lw=0.7, label="Observed")
            ax.plot(wl[region], scale * model_on_obs[region], c="b", lw=0.7,
                    label="Model")
            for span, color in ((left, "tab:blue"), (centre, "tab:green"),
                                (right, "tab:red")):
                ax.axvspan(*span, color=color, alpha=0.2)
            ax.set_xlim(wl[region][[0, -1]])
            ax.annotate(
                f"{names[ith]}\n"
                f"obs: {ew_obs[ith]:.2f} $\\pm$ {ew_err[ith]:.2f} $\\AA$\n"
                f"model: {ew_model[ith]:.2f} $\\AA$ ($\\Delta$ = {chi[ith]:.1f}$\\sigma$)",
                xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=8,
                bbox=dict(boxstyle="round", fc="w", alpha=0.8))
            if ith == 0:
                ax.legend(loc="lower right", fontsize=7)
        for ax in axs.flat[n_ew:]:
            ax.axis("off")
        for ax in axs[-1]:
            ax.set_xlabel("Wavelength (AA)")
        for ax in axs[:, 0]:
            ax.set_ylabel("Flux")

        if figname is not None:
            fig.savefig(figname, bbox_inches="tight", dpi=300)
            logger.info("Fit plot saved at: %s", figname)

        solution_data = {
            "name": names,
            "ew_obs": ew_obs,
            "ew_err": ew_err,
            "ew_model": ew_model,
            "chi": chi,
        }
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
