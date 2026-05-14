"""Full spectral fitting pipeline module."""

import os
import numpy as np

from besta.pipeline_modules.base_module import SpectraFitModule
from cosmosis.datablock import names as section_names
from cosmosis.datablock import SectionOptions
from besta import spectrum
from besta.logging import get_logger
from numba import njit, prange
logger = get_logger(__name__)


@njit(parallel=True, fastmath=False)
def compute_redshift_chi2_from_slices(
    flux_model,
    target_flux,
    candidate_weights,
    slice_starts,
    slice_stops,
    good_idx,
):
    n_z = len(slice_starts)

    z_chi2 = np.full(n_z, np.inf)
    z_scales = np.full(n_z, np.nan)

    # Loop over all redshift steps in parallel
    for i in prange(n_z):
        start = slice_starts[i]

        numerator = 0.0
        denominator = 0.0

        for jj in range(len(good_idx)):
            k = start + good_idx[jj]

            f = flux_model[k]
            t = target_flux[jj]
            w = candidate_weights[jj]

            numerator += w * f * t
            denominator += w * f * f

        if denominator <= 0.0:
            continue

        scale = numerator / denominator
        z_scales[i] = scale

        chi2 = 0.0

        for jj in range(len(good_idx)):
            k = start + good_idx[jj]

            f = flux_model[k]
            t = target_flux[jj]
            w = candidate_weights[jj]

            residual = scale * f - t
            chi2 += w * residual * residual

        z_chi2[i] = chi2

    return z_chi2, z_scales

class SpectraRedshiftFitModule(SpectraFitModule):
    """Fit stellar populations and kinematics directly from galaxy spectra."""

    name = "SpectraRedshiftFit"

    def __init__(self, options, **kwargs):
        """
        Set up the full spectral fit module.

        Parameters
        ----------
        options : dict or DataBlock
            Options from the startup configuration.
        **kwargs : dict
            Extra keyword arguments forwarded to ``SpectraFitModule``.
        """
        super().__init__(options, **kwargs)
        options = self.parse_options(options)

        if options.has_value("redshift") and options["redshift"] != 0:
            raise ValueError(
                "SpectraRedshiftFitModule requires 'redshift = 0.0' because it "
                "searches for the absolute redshift directly from the observed spectrum."
            )
        self.prepare_observed_spectra(options)
        # Remove this variable to let the SSP wavelength range stay native
        del self.config["ln_wave"]
        self.prepare_ssp_model(options)

        if getattr(self.config["ssp_model"], "_has_gas_model", False):
            logger.info("The SSP model includes a gas emission component.")

        self.prepare_sfh_model(options)
        self.prepare_extinction_law(options)
        self.prepare_legendre_polynomials(options)

        logger.info(f"{self.config['ssp_model'].wavelength[0]:.1f} - "
                    + f"{self.config['ssp_model'].wavelength[-1]:.1f} AA (rest frame)")
        # Precompute the model wavelength grid in the observed frame for redshift
        norm = np.nanmedian(self.config["flux"][self.config["weights"] > 0])
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Observed flux normalization is not finite; cannot perform redshift fit.")
        self.config["norm_obs_flux_scale"] = norm
        self.config["norm_obs_flux"] = self.config["flux"] / norm
        self.config["norm_obs_var"] = self.config["var"] / norm**2

        n_pix = self.config["flux"].size

        if options.has_value("z_max"):
            z_max = options["z_max"]
            wl_start_rest = self.config["wavelength"][0].to_value("AA") / (1.0 + z_max)
        else:
            wl_start_rest = self.config['ssp_model'].wavelength[0].to_value("AA")
            z_max = self.config["wavelength"][0].to_value("AA") / wl_start_rest - 1.0

        model_start = np.searchsorted(
            self.config["ssp_model"].wavelength.to_value("AA"),
            wl_start_rest)
        if model_start == self.config["ssp_model"].wavelength.size:
            raise ValueError(f"Model wavelength range does not extend to {wl_start_rest:.1f} AA. Check the SSP model and wlRange configuration.")
        logger.info(f"Maximum redshift: {z_max:.1f} (model wavelength range in rest frame starts at {self.config['ssp_model'].wavelength[model_start].to_value('AA'):.1f} AA)")

        if options.has_value("z_min"):
            z_min = options["z_min"]
            wl_stop_rest = self.config["wavelength"][-1].to_value("AA") / (1.0 + z_min)
        else:
            wl_stop_rest = self.config['ssp_model'].wavelength[-1].to_value("AA")
            z_min = self.config["wavelength"][-1].to_value("AA") / wl_stop_rest - 1.0
        
        model_stop = np.searchsorted(
            self.config["ssp_model"].wavelength.to_value("AA"),
            wl_stop_rest,
            side="right")
        if model_stop == 0:
            raise ValueError(f"Model wavelength range does not extend to {wl_stop_rest:.1f} AA. Check the SSP model and wlRange configuration.")
        logger.info(f"Minimum redshift: {z_min:.1f} (model wavelength range in rest frame ends at {self.config['ssp_model'].wavelength[model_stop-1].to_value('AA'):.1f} AA)")
        observed_wavelength = self.config["wavelength"].to_value("AA")
        model_wavelength = self.config["ssp_model"].wavelength.to_value("AA")
        observed_dlog_wavelength = np.diff(np.log(observed_wavelength))

        # Create slices for every iteration. The sweep compares pixels by index,
        # so the observed grid must match the corresponding rest-frame model grid
        # up to a single redshift factor.
        model_slices = []
        for start in range(model_start, model_stop - n_pix + 1):
            stop = start + n_pix
            model_dlog_wavelength = np.diff(np.log(model_wavelength[start:stop]))
            if np.allclose(
                observed_dlog_wavelength,
                model_dlog_wavelength,
                rtol=1e-3,
                atol=1e-3,
            ):
                model_slices.append(slice(start, stop))
        if len(model_slices) == 0:
            raise ValueError(
                "No valid redshift steps found. The observed wavelength grid must "
                "match a shifted SSP wavelength-grid window. Provide a matching "
                "'velscale' configuration or resample the input spectrum onto the "
                "model grid."
            )
        self.config["model_slices"] = model_slices
        self.config["model_start"] = np.array([s.start for s in model_slices])
        self.config["model_stop"] = np.array([slc.stop for slc in model_slices], dtype=np.int64)

        # Likelihood values of all redshift stepsg.
        slice_redshifts = observed_wavelength[0] / model_wavelength[self.config["model_start"]] - 1.0
        self.config["slice_redshifts"] = slice_redshifts
        self.z_loglike = np.full(len(model_slices), fill_value=-1e20, dtype=np.float32)

        self.save_z_loglike = False
        if options.has_value("save_z_loglike"):
            self.save_z_loglike = True
            self.z_loglike_path = options["save_z_loglike"]
            logger.info(f"Redshift log-likelihood profile will be saved to {self.z_loglike_path}")

            # Check if the output file already exists to avoid overwriting previous results.
            if os.path.exists(self.z_loglike_path):
                logger.info("Loading existing redshift log-likelihood profile from file.")
                z, loglike = np.loadtxt(self.z_loglike_path, unpack=True)
                if np.array_equal(z, slice_redshifts):
                    self.z_loglike = loglike
                else:
                    logger.warning("Redshift grid in existing log-likelihood file does not match the current configuration. Starting with an empty log-likelihood profile.")

        logger.info(
            f"Model wavelength range in rest frame: {self.config['ssp_model'].wavelength[model_start].to_value('AA'):.1f} - {self.config['ssp_model'].wavelength[model_stop-1].to_value('AA'):.1f} AA")
        logger.info(f"Number of redshift steps: {len(model_slices)} (z={z_max:.1f} to {z_min:.1f})")

        if options.get_bool("use_features", default=False):
            self.get_feature_weights(options)
        else:
            logger.info("Using original weights for redshift fitting.")

        w = self.config.get("feature_weights",
                            np.ones_like(self.config["flux"], dtype=np.float32))
        self.config["sweep_weights"] = np.where(
            (w > 0) & np.isfinite(self.config["norm_obs_var"]),
            w * self.config["weights"] / self.config["norm_obs_var"],
            0.0
        )
        # This are used for the likelihood in combination with the variance
        self.config["weights_orig"] = self.config["weights"].copy()
        self.config["weights"] *= w
        self.config["good"] = self.config["sweep_weights"] > 0
        self.config["good_idx"] = np.flatnonzero(self.config["good"]).astype(np.int64)


    @spectrum.legendre_decorator
    def make_observable(self, block, parse=False):
        """Create the spectra model from the input parameters"""
        # Stellar population synthesis
        sfh_model = self.config["sfh_model"]
        if parse:
            sfh_model.parse_datablock(block)
        
        # Here we compute the luminosity by we call it flux and rescale later
        flux_model = sfh_model.model.compute_SED(
            self.config["ssp_model"], t_obs=sfh_model.today, allow_negative=False
        ).to_value(self._default_luminosity_units)

        # Apply dust extinction
        dust_model = self.config["extinction_law"]
        if dust_model is not None:
            flux_model = dust_model.apply_extinction(
                self.config["ssp_model"].wavelength, flux_model,
                a_v=block["dust.extinction", "a_v"]
            ).value

        w = self.config["sweep_weights"]
        good = self.config["good"]
        good_idx = self.config["good_idx"]
        candidate_weights = w[good]
        slc_starts = self.config["model_start"]
        slc_stops = self.config["model_stop"]

        norm_obs_flux = self.config["norm_obs_flux"]
        target_flux = norm_obs_flux[good]

        # z_chi2 = np.full(len(self.config["model_slices"]), np.inf)
        # z_scales = np.full(len(self.config["model_slices"]), np.nan)

        z_chi2, z_scales = compute_redshift_chi2_from_slices(
                            flux_model,
                            target_flux,
                            candidate_weights,
                            slc_starts,
                            slc_stops,
                            good_idx,
                        )
        
        # Sweep over all target slices using the same weighted least-squares
        # scale that is applied to the selected model below.
        # for i, slc in enumerate(self.config["model_slices"]):
        #     candidate_flux = flux_model[slc][good]
        #     denominator = np.nansum(candidate_weights * candidate_flux**2)
        #     if denominator <= 0:
        #         logger.warning(f"Denominator for redshift step {i} is non-positive; skipping this step.")
        #         continue
        #     scale = np.nansum(candidate_weights * candidate_flux * target_flux) / denominator
        #     z_scales[i] = scale
        #     z_chi2[i] = np.nansum(
        #         candidate_weights * (candidate_flux * scale - target_flux) ** 2
        #     )

        # Keep track of the likelihood values for all redshift steps.
        self.z_loglike = np.maximum(self.z_loglike, -0.5 * z_chi2)
        if not np.any(np.isfinite(z_chi2)):
            raise ValueError("No finite redshift likelihood values found.")
        # Best fit
        best_fit_slice_index = np.argmin(z_chi2)
        best_fit_index = self.config["model_start"][best_fit_slice_index]
        z_best = self.config["slice_redshifts"][best_fit_slice_index]
        block["redshift", "redshift"] = z_best

        logger.debug(
            "Best fit redshift: %.4f (chi2=%.2f)",
            z_best,
            z_chi2[best_fit_slice_index],
        )
        # Re-scale model flux to match the observed flux level, and convert to physical units for
        # luminosity distance calculation. The stellar mass is then inferred from the normalization.
        # dl_sq = cosmology.luminosity_distance(z_best).to_value("cm")**2
        # flux_model /= 4 * np.pi * dl_sq
        # Use the original weights to estimate the normalization
        w = self.config["weights_orig"]
        candidate_flux = flux_model[best_fit_index : best_fit_index + w.size]
        denominator = np.nansum(w[good] * candidate_flux[good]**2)

        if denominator <= 0:
            logger.warning(f"Denominator for redshift step {i} is non-positive; skipping this step.")
            return np.full_like(candidate_flux, np.nan), w

        scale = np.nansum(w[good] * candidate_flux[good] * target_flux) / denominator
        normalization = scale * self.config["norm_obs_flux_scale"]        
        #block["extra", "stellar_mass"] = np.log10(normalization) + 10
        return candidate_flux * normalization, self.config["weights"]

    def execute(self, block):
        """Function executed by sampler
        This is the function that is executed many times by the sampler. The
        likelihood resulting from this function is the evidence on the basis
        of which the parameter space is sampled.
        """
        valid, penalty = self.config["sfh_model"].parse_datablock(block)
        if not valid:
            # To track invalid samples users can set debug=T in the .ini file
            block[section_names.likelihoods, self.like_name] = -1e20 * penalty
            block["extra", "stellar_mass"] = np.nan
            return 0
        # Obtain parameters from setup
        var = self.config["var"]
        flux_model, weights = self.make_observable(block)
        # Calculate likelihood-value of the fit
        good_pixels = weights > 0
        like = self.log_like(self.config["flux"][good_pixels],
                             flux_model[good_pixels],
                             var[good_pixels],
                             weights=weights[good_pixels])
        # Final posterior for sampling
        block[section_names.likelihoods, self.like_name] = like
        return 0

    def cleanup(self):
        """Persist the redshift likelihood profile if requested."""
        if self.save_z_loglike:
            logger.info(f"Saving redshift log-likelihood profile to {self.z_loglike_path}")
            np.savetxt(
                self.z_loglike_path,
                np.column_stack(
                    (self.config["slice_redshifts"], self.z_loglike)),
                    header="redshift log_likelihood")



def setup(options):
    """Create the CosmoSIS-facing module instance."""

    options = SectionOptions(options)
    mod = SpectraRedshiftFitModule(options)
    return mod


def execute(block, mod):
    """Run one likelihood evaluation for the configured module."""

    mod.execute(block)
    return 0


def cleanup(mod):
    """Release module resources after sampling."""

    mod.cleanup()

module = SpectraRedshiftFitModule
