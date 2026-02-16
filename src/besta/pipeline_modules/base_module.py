"""
Base pipeline module. This module contains the base class for creating new
pipeline modules in BESTA.
"""
from __future__ import annotations

from abc import abstractmethod
import os
import pickle
import sys

import importlib.util
import pathlib
import types
from typing import Callable, Optional

from matplotlib import pyplot as plt
from besta.visualization import draw_dict_in_axes

import numpy as np
from scipy.stats import norm
from sklearn.decomposition import NMF
from astropy import units as u
from astropy.io import ascii

from cosmosis import ClassModule
from cosmosis import DataBlock
from cosmosis.datablock import SectionOptions, option_section

from pst.utils import flux_conserving_interpolation
from pst.observables import Filter, FilterList
from pst import SSP, dust, sed
from pst.galaxy import GalaxySED

from besta import spectrum
from besta import kinematics
from besta import sfh
from besta import io
from besta.grid import ModelGrid
from besta.config import cosmology, memory
from besta.logging import get_logger

logger = get_logger(__name__)


def _log(*args):
    logger.info(" ".join(str(arg) for arg in args))


class BaseModule(ClassModule):
    """BESTA Pipeline module base class."""

    def __init__(self, options, *, alias=None):
        """Set-up the COSMOSIS sampler.
        Args:
            options: options from startup file (i.e. .ini file)
        """
        if alias is None:
            self.alias = self.name
        else:
            self.alias = alias
        options = self.parse_options(options)
        self.config = {}
        # Likelihood name
        if options.has_value("like_name"):
            self.like_name = options["like_name"]
            if "_like" not in self.like_name:
                self.like_name += "_like"
        else:
            self.like_name = self.name + "_like"
            _log("Setting module likelihood name to default: ", self.like_name)

    @abstractmethod
    def make_observable(self, *args, **kwargs):
        """Create an observable from an input set of model parameters."""

    @abstractmethod
    def execute(self, block: DataBlock, *args, **kwargs):
        """Execute the pipeline."""

    @abstractmethod
    def plot_solution(self, *args, **kwargs):
        """Plot the fit results."""
        pass

    @classmethod
    def get_path(cls):
        """Get the path to the module file."""
        return sys.modules[cls.__module__].__file__

    def parse_options(self, options: dict | DataBlock):
        """Parse the input setup options.

        Convert the input options into a :class:`SectionOptions`

        Parameters
        ----------
        options : dict or :class:`DataBlock`
            Module setup options.

        Returns
        -------
        options : :class:`SectionOptions` or :class:`DataBlock`
        """
        if isinstance(options, dict):
            options = DataBlock.from_dict(options)
            if options.has_section(option_section):
                options._delete_section(option_section)
            keys = options.keys(self.alias)
            if not keys:
                raise ValueError(f"No options found for module alias {self.alias}")
            for section, name in keys:
                options[option_section, name] = options[section, name]
            options = SectionOptions(options)
        return options

    def prepare_ssp_model(self, options, normalize=False, velocity_buffer=800.0):
        """Prepare the SSP data.

        Parameters
        ----------
        options : :class:`DataBlock`
            Input options to initialise the model.
        normalize : bool, optional
            If ``True``, normalizes the spectra using the given wavelength range.
        velocity_buffer : float
            Buffer offset (in terms of velocity) to keep extra wavelength
            elements. This reduced the corruption of the spectra at the edges
            during convolution. The buffer is applied to both sides of the
            SSP spectra.
        """
        _log("\n-> Configuring SSP model")

        if options.has_value("SSPModelFromPickle"):
            _log("\n-> Loading preconfigured SSP model from pickle")
            if not os.path.isfile(
                os.path.expandvars(options["SSPModelFromPickle"])):
                raise FileNotFoundError(
                    f"Input pickle file {options['SSPModelFromPickle']} not found")

            # Load the SSP model
            with open(
                os.path.expandvars(options["SSPModelFromPickle"]), 'rb') as file:
                ssp = pickle.load(file)

            self.config["ssp_model"] = ssp
            self.config["ssp_sed"] = ssp.L_lambda.value.reshape(
            (ssp.L_lambda.shape[0] * ssp.L_lambda.shape[1],
             ssp.L_lambda.shape[2]))
            self.config["ssp_wl"] = ssp.wavelength.to_value("Angstrom")
            # Grid parameters
            velscale = options["velscale"]
            dlnlam = velscale / spectrum.constants.c.to("km/s").value
            extra_offset_pixel = int(velocity_buffer / velscale)
            self.config["velscale"] = velscale
            self.config["extra_pixels"] = extra_offset_pixel
            _log("-> Configuration done.")
            return

        ssp_name = options["SSPModel"]
        if options.has_value("SSPDir"):
            ssp_dir = options["SSPDir"]
            if "none" in ssp_dir.lower():
                ssp_dir = None
        else:
            ssp_dir = None

        # Additional arguments to be passed to the SSP model
        if options.has_value("SSPModelArgs"):
            ssp_args = options.get_string("SSPModelArgs")
            ssp_args = ssp_args.split(",")
            _log("SSP Model extra arguments: ", ssp_args)
        else:
            ssp_args = []

        ssp = getattr(SSP, ssp_name)(*ssp_args, path=ssp_dir)

        # For photometric analyses stop here
        #TODO: loose check
        if not options.has_value("velscale"):
            self.config["ssp_model"] = ssp
            return
        # Parameters to format the templates to the input spectra
        velscale = options["velscale"]

        if options.has_value("SSP-NMF-N"):
            n_nmf = options.get_int("SSP-NMF-N")
        else:
            n_nmf = None

        # Rebin the spectra
        dlnlam = velscale / spectrum.constants.c.to("km/s").value
        extra_offset_pixel = int(velocity_buffer / velscale)
        _log(
            "Log-binning SSP spectra to velocity scale: ",
            velscale,
            " km/s",
            f"\nKeeping {extra_offset_pixel} extra pixels at both edges",
        )

        if "ln_wave" in self.config:
            ln_wl_edges = self.config["ln_wave"][[0, -1]]
            # Add extra pixels at the edges to prevent corruption during convolution
        else:
            ln_wl_edges = np.log(ssp.wavelength[[0, -1]].to_value("angstrom"))
            extra_offset_pixel = 0

        lnlam_bin_edges = np.arange(
            ln_wl_edges[0] - 0.5 * dlnlam - dlnlam * extra_offset_pixel,
            ln_wl_edges[-1] + dlnlam * (1 + extra_offset_pixel),
            dlnlam,
        )
        lnlam_bins = (lnlam_bin_edges[:-1] + lnlam_bin_edges[1:]) / 2

        # Resample the SED
        ssp.interpolate_sed(np.exp(lnlam_bins), method="binfrac")
        _log("SSP Model SED dimensions (met, age, lambda): ", ssp.L_lambda.shape)

        # Convolve with instrumental LSF
        if "lsf" in self.config:
            _log("Convolving SSP model with instrumental LSF")
            inst_lsf = np.interp(ssp.wavelength, self.config["wavelength"],
                                 self.config["lsf"])

            if options.has_value("SSPLSF"):
                _log("Including SSP resolution")
                ssp_lsf_wl, ssp_lsf_fwhm = np.loadtxt(
                    os.path.expandvars(options["SSPLSF"]),
                    unpack=True, usecols=(0, 1))
                ssp_lsf_fwhm = np.interp(ssp.wavelength,
                                         ssp_lsf_wl << u.AA, ssp_lsf_fwhm)
            else:
                ssp_lsf_fwhm = np.zeros(ssp.wavelength.size, dtype=float)
            # Assume both LSF are Gaussian
            effective_lsf_disp = (inst_lsf / 2.355)**2 - (ssp_lsf_fwhm / 2.355)**2

            if (effective_lsf_disp <= 0).any():
                raise ValueError("Effective SSP LSF cannot be negative!"
                                 + "SSP models do not have enough resolution")
            effective_lsf = np.sqrt(effective_lsf_disp)
            # Convert to pixels
            lsf_sigma_pixels = effective_lsf / np.diff(np.exp(lnlam_bin_edges))
            _log("Starting convolution of SSP models with wavelength-dependent",
                  f"LSF [min sigma={lsf_sigma_pixels.min():.2},"
                  f" max sigma={lsf_sigma_pixels.max():.2} pix]")
            try:
                io.check_array_memory(
                (ssp.L_lambda.shape[0], ssp.L_lambda.shape[1],
                 ssp.L_lambda.shape[2], ssp.L_lambda.shape[2]),
                dtype=ssp.L_lambda[0, 0, 0].dtype, unit='GB',
                safety_margin=memory["ram_safety_margin"])
                _log("Convolving full SSP model at once")
                ssp.L_lambda = kinematics.convolve_variable_gaussian_kernel(
                    ssp.L_lambda, lsf_sigma_pixels)
            except MemoryError:
                # Do a loop along metallicity axis to prevent memory overflows
                _log("Insufficient RAM memory for full SSP SED convolution")
                _log("Looping along metallicity axis")
                io.check_array_memory(
                (ssp.L_lambda.shape[1], ssp.L_lambda.shape[2],
                 ssp.L_lambda.shape[2]),
                dtype=ssp.L_lambda[0, 0, 0].dtype, unit='GB',
                safety_margin=memory["ram_safety_margin"])

                for ith in range(ssp.L_lambda.shape[0]):
                    ssp.L_lambda[ith] = kinematics.convolve_variable_gaussian_kernel(
                    ssp.L_lambda[ith], lsf_sigma_pixels)

        # Reshape the SSP model from (metal, age, wave) -> (metal * age, wave)
        ssp_sed = ssp.L_lambda.value.reshape(
            (ssp.L_lambda.shape[0] * ssp.L_lambda.shape[1], ssp.L_lambda.shape[2])
        )
        # Apply Non-negative Matrix Factorisation for reducing dimensionality
        if n_nmf is not None:
            _log(
                "Reducing SSP model dimensionality with Non-negative Matrix Factorisation",
                "\nNo. of components: ",
                n_nmf,
            )
            # TODO: hard-coded parameters
            pca = NMF(n_components=n_nmf, alpha_H=1.0, max_iter=n_nmf * 1000)
            pca.fit(ssp_sed)
            ssp_sed = pca.components_

        self.config["ssp_model"] = ssp
        self.config["ssp_wl"] = ssp.wavelength.to_value("Angstrom")
        # Grid parameters
        self.config["velscale"] = velscale
        self.config["extra_pixels"] = extra_offset_pixel
        if options.has_value("SaveSSPModel"):
            _log("Saving photometry grid to ", options["SaveSSPModel"])
            with open(os.path.expandvars(options["SaveSSPModel"]), 'wb') as file:
                pickle.dump(ssp, file, pickle.HIGHEST_PROTOCOL)
        _log("-> Configuration done.")
        return

    def prepare_extinction_law(self, options):
        """Prepare a dust extinction model.

        options : :class:`DataBlock`
            Input options to initialise the model.
        """
        _log("\n -> Configuring Dust extinction model")
        if not options.has_value("ExtinctionLaw"):
            self.config["extinction_law"] = None
            return
        ext_law = options.get_string("ExtinctionLaw")
        _log("Extinction law: ", ext_law)
        # TODO: add more extinction laws
        self.config["extinction_law"] = dust.DustScreen(ext_law)
        _log("-> Configuration is done.")

    def prepare_sfh_model(self, options):
        """Prepare the SFH model.

        Parameters
        ----------
        options : :class:`DataBlock`
            Input options to initialise the model.
        """
        _log("\n-> Configuring SFH model")
        sfh_model_name = options["SFHModel"]
        sfh_args = []
        key = "SFHArgs1"
        i = 1
        while options.has_value(f"SFHArgs{i}"):
            key = f"SFHArgs{i}"
            i += 1
            value = options[key]
            if isinstance(value, str):
                if "," in value:
                    value = np.array(value.split(","), dtype=float)
            sfh_args.append(value)
        # Optional: enable parameter transforms inside SFH models
        self.config["use_transforms"] = False
        if options.has_value("use_transforms"):
            self.config["use_transforms"] = bool(options["use_transforms"])
        if self.config["use_transforms"]:
            _log("Enabling parameter transforms inside SFH model")
        _log("SFH model name: ", sfh_model_name)
        sfh_model = getattr(sfh, sfh_model_name)
        sfh_model = sfh_model(*sfh_args, **self.config)
        self.config["sfh_model"] = sfh_model
        _log("-> Configuration done")

    def log_like(self, data, model, var, weights=None, is_upper=None, is_lower=None, include_norm=True):
        """Compute log-likelihood between data and model.

        Parameters
        ----------
        data : np.ndarray
            For detections: measured values.
            For limits: the limit value (upper or lower).
        model : np.ndarray
            Model prediction for each datum.
        var : np.ndarray
            Data variance. Must match shape of data/model.
        weights : np.ndarray, optional
            Data weights. If provided, returns weighted mean log-likelihood.
        is_upper : np.ndarray[bool], optional
            Mask for upper limits (x < data).
        is_lower : np.ndarray[bool], optional
            Mask for lower limits (x > data).
        include_norm : bool, optional, default=True
            If True, include Gaussian normalization terms for detections:
            -0.5*log(2*pi*var).

        Returns
        -------
        loglike : float
            Total (or weighted-mean) log-likelihood.
        """
        if data.shape != model.shape or data.shape != var.shape:
            raise ValueError("data, model, var must have the same shape (var is per-datum variance).")
        if np.any(var <= 0):
            raise ValueError("All var entries must be > 0 (variance).")

        if is_upper is None:
            is_upper = np.zeros_like(data, dtype=bool)
        else:
            is_upper = np.asarray(is_upper, dtype=bool)

        if is_lower is None:
            is_lower = np.zeros_like(data, dtype=bool)
        else:
            is_lower = np.asarray(is_lower, dtype=bool)

        if is_upper.shape != data.shape or is_lower.shape != data.shape:
            raise ValueError("is_upper and is_lower must have the same shape as data/model.")
        if np.any(is_upper & is_lower):
            raise ValueError("A data point cannot be both an upper and a lower limit.")

        if weights is None:
            weights = np.ones_like(data, dtype=float)
            normalize = False
        else:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != data.shape:
                raise ValueError("weights must have the same shape as data/model.")
            if np.any(weights < 0):
                raise ValueError("weights must be non-negative.")
            normalize = True

        sigma = np.sqrt(var)

        logp = np.empty_like(data, dtype=float)
        det = ~(is_upper | is_lower)

        # For detections: Gaussian logpdf
        if np.any(det):
            if include_norm:
                logp[det] = norm.logpdf(data[det], loc=model[det], scale=sigma[det])
            else:
                z = (data[det] - model[det]) / sigma[det]
                logp[det] = -0.5 * z**2

        # Upper limits: P(x < L)
        if np.any(is_upper):
            z_u = (data[is_upper] - model[is_upper]) / sigma[is_upper]
            logp[is_upper] = norm.logcdf(z_u)

        # Lower limits: P(x > L) = 1 - P(x < L)
        if np.any(is_lower):
            z_l = (data[is_lower] - model[is_lower]) / sigma[is_lower]
            logp[is_lower] = norm.logsf(z_l)  # 1 - logcdf

        # User-provided weighted mean
        if normalize:
            wsum = np.sum(weights)
            if wsum <= 0:
                raise ValueError("Sum of weights must be > 0.")
            return np.sum(logp * weights) / wsum

        return np.sum(logp * weights)



class SpectraFitModule(BaseModule):
    """Base class for spectral fitting modules in BESTA."""

    def prepare_observed_spectra(
        self, options: DataBlock, normalize=False):
        """Prepare the input spectra data.

        Parameters
        ----------
        options : :class:`DataBlock`
        normalize : bool, optional
            If ``True``, normalizes the spectra using the given wavelength range.
        """
        _log("\n-> Configuring input observed spectra")
        filename = os.path.expandvars(options["inputSpectrum"])
        # Read wavelength and spectra
        _log("Loading observed spectra from input file: ", filename)
        wavelength, flux, error = np.loadtxt(filename, unpack=True)
        _log("Wavelength coverage: ", wavelength[[0, -1]])
        _log("Size: ", wavelength.size)

        # Convert units if needed
        if options.has_value("wlUnits"):
            _log("Converting wavelength units to Angstrom")
            wl_units = u.Unit(options["wlUnits"])
            wavelength = (wavelength << wl_units).to("Angstrom").value
        else:
            _log("Assuming input wavelength units are in Angstrom")
            wl_units = u.angstrom

        if options.has_value("fluxUnits"):
            _log("Converting flux units to 1e-16 erg/s/cm^2/Angstrom")
            flux_units = u.Unit(options["fluxUnits"])
            flux = (flux << flux_units).to(
                "1e-16 erg / (s cm2 Angstrom)").value
            error = (error << flux_units).to(
                "1e-16 erg / (s cm2 Angstrom)").value
        else:
            _log("Assuming input flux units are in 1e-16 erg/s/cm^2/Angstrom")
            flux_units = u.Unit("1e-16 erg / (s cm2 Angstrom)")

        # Wavelength range to include in the fit
        if options.has_value("wlRange"):
            wl_range = (np.asarray(options["wlRange"]) << wl_units
            ).to("Angstrom").value
        else:
            wl_range = wavelength[[0, -1]]
        # Wavelength range to renormalize the spectra
        if options.has_value("wlNormRange"):
            wl_norm_range = (np.asarray(options["wlNormRange"]) << wl_units
            ).to("Angstrom").value
        else:
            wl_norm_range = wavelength[[0, -1]]
        # Input redshift (initial guess)
        if options.has_value("redshift"):
            redshift = options["redshift"]
        else:
            _log("No input redshift value provided (defaulting to 0)")
            redshift = 0.0
        # Load mask
        if options.has_value("mask"):
            weights = np.array(
                np.loadtxt(os.path.expandvars(options["mask"])), dtype=float)
        else:
            weights = np.ones_like(flux)
        
        if weights.size != flux.size:
            raise ValueError(
                "Input mask size does not match the input spectrum size.")
        # Load the instrumental LSF
        if options.has_value("lsf"):
            lsf_wl, lsf_fwhm = np.loadtxt(os.path.expandvars(options["lsf"]),
                                          unpack=True)
            instrumental_lsf = np.array(np.interp(wavelength, lsf_wl, lsf_fwhm),
                                        dtype=float)
        else:
            instrumental_lsf = np.zeros_like(wavelength)
        
        # Optional masking of telluric regions
        if options.has_value("mask_telluric") and options["mask_telluric"]:
            telluric_pad = options.get_double("telluric_pad", default=0.0)
            telluric_pad = (telluric_pad << wl_units).to("Angstrom").value
            _log(f"Masking telluric regions with pad={telluric_pad} Angstrom")
            weights, tell_mask = spectrum.mask_telluric_regions(
                wavelength, flux, error, weights,
                pad=telluric_pad,
                return_mask=True)
            _log("Number of masked pixels: ", np.count_nonzero(tell_mask))
            self.config["telluric_mask"] = tell_mask

        # Optional masking of emission lines
        if options.has_value("mask_emission_lines") and options["mask_emission_lines"]:
            weights, line_mask = spectrum.mask_strong_emission_lines(
                wavelength, flux, error, weights,
                redshift=redshift,
                # line_list=emission_line_list,
                # half_width=line_half_width,
                return_mask=True)
            _log("Number of masked pixels: ", np.count_nonzero(line_mask))
            self.config["emission_lines_mask"] = line_mask

        # Apply redshift
        _log(f"Setting wavelength array to restframe (redshift: {redshift})")
        wavelength /= 1.0 + redshift
        _log("Constraining fit to wavelength range: ", wl_range)
        good_idx = np.where(
            (wavelength >= wl_range[0]) & (wavelength <= wl_range[1]))[0]
        if len(good_idx) == 0:
            raise ValueError("No wavelength points found within the given"
                             "wavelength range.")
        wavelength = wavelength[good_idx]
        flux = flux[good_idx]
        cov = error[good_idx] ** 2
        weights = weights[good_idx]
        instrumental_lsf = instrumental_lsf[good_idx]
        # Check error
        if (cov <= 0).any():
            raise ValueError("Input flux error contains negative or null values.")
        if np.nansum(weights) <= 0:
            raise ValueError("All input weights are zero; cannot perform fit.")

        _log("Number of selected pixels within wavelength range: ", good_idx.size)
        if options.has_value("velscale"):
            velscale = options["velscale"]
        else:
            # Set velscale to None
            velscale = None
        _log("Log-binning spectra to velocity scale: ", velscale, " (km/s)")
        # Update the value of velscale
        if velscale is not None:
            dlnlam = velscale / spectrum.constants.c.to("km/s").value
            ln_wave = np.arange(np.log(wl_range[0]), np.log(wl_range[1]) + dlnlam,
                            dlnlam)

            flux = flux_conserving_interpolation(ln_wave, np.log(wavelength), flux)
            cov = flux_conserving_interpolation(ln_wave, np.log(wavelength), cov)
            weights = np.interp(ln_wave, np.log(wavelength), weights)
            instrumental_lsf = np.interp(ln_wave, np.log(wavelength), instrumental_lsf)
    
            new_wavelength = np.exp(ln_wave)
            weights[(new_wavelength < wavelength[0]) | (new_wavelength > wavelength[-1])] = 0.0
            wavelength = new_wavelength
        else:
            ln_wave = np.log(wavelength)

        _log("Number of pixels after interpolation: ", wavelength.size)
        # Normalize spectra
        if normalize:
            _log("Spectra normalized using wavelength range: ", wl_norm_range)
            norm_idx = np.where(
                (wavelength >= wl_norm_range[0]) & (wavelength <= wl_norm_range[1])
            )[0]
            norm_flux = np.nanmedian(flux[norm_idx])
            flux /= norm_flux
            cov /= norm_flux**2
            flux_units = u.dimensionless_unscaled
        else:
            norm_flux = 1.0

        if redshift > 0:
            dl_sq = cosmology.luminosity_distance(redshift).to("cm").value ** 2
            dl_sq = 4 * np.pi * dl_sq * (1 + redshift)
        else:
            dl_sq = (10 * u.pc).to("cm").value ** 2 * 4 * np.pi

        self.config["flux"] = flux
        self.config["var"] = cov
        self.config["redshift"] = redshift
        self.config["wlUnits"] = wl_units
        self.config["fluxUnits"] = flux_units
        self.config["dl_sq"] = dl_sq
        self.config["norm_flux"] = norm_flux
        self.config["wavelength"] = wavelength << u.angstrom
        self.config["ln_wave"] = ln_wave
        self.config["weights"] = weights
        if not (instrumental_lsf == 0).all():
            self.config["lsf"] = instrumental_lsf

        _log("-> Configuration done.")

    def prepare_galaxy(self, options):
        """Build and configure a :class:`pst.galaxy.GalaxySED` model.

        The method initializes stellar, attenuation, and optional dust emission
        components using the already prepared configuration and stores the
        resulting galaxy model and parameter index in ``self.config``.
        """
        
        # Stellar emissions
        ssp_model = self.config.get("ssp_model")
        if ssp_model is None:
            self.prepare_ssp_model(options)
            ssp_model = self.config["ssp_model"]

        sfh_model = self.config.get("sfh_model")
        if sfh_model is None:
            self.prepare_sfh_model(options)
            sfh_model = self.config["sfh_model"].model

        ## Create stellar emission model
        _log("Setting up stellar emission component")
        stars = sed.StellarComponent(ssp=ssp_model, sfh=sfh_model)
        
        # Dust extinction and emission
        dust_attenuation = None
        dust_emission = None
        if options.get_bool("DustAttenuation", False):
            _log("Setting up dust attenuation")
            att_name = options.get_string(
                "DustAttenuationModel", "DustScreenAttenuation")
            ext_law = options.get_string("ExtinctionLaw", "ccm89")
            dust_curve = dust.ExtinctionLibCurve(law=ext_law)
            _log("DurstAttenuationModel: ", att_name)
            # TODO: implement interface for other models
            if att_name == "DustScreenAttenuation":
                dust_attenuation = dust.DustScreenAttenuation(curve=dust_curve)

            if options.get_bool("DustEmission", False):
                _log("Setting up dust emission model")
                dust_sed = dust.Casey2012DustComponent()
                if options.get_bool("DustCalorimetric", True):
                    _log("Setting energy balance approximation")
                    dust_emission = dust.CalorimetricDustComponent(
                        attenuation=dust_attenuation,
                        dust_sed_component=dust_sed
                    )

        _log("Setting up galaxy model")
        galaxy = GalaxySED(stellar_model=stars,
                           dust_attenuation_model=dust_attenuation,
                           dust_model=dust_emission,
                           target_wavelength=self.config["ssp_model"].wavelength,
                           redshift=self.config.get("redshift", 0.0),
                           cosmology=cosmology)

        params = galaxy.build_param_index(include_fixed=False, prefix="")
        sections = [s.rsplit(".", 1) for s in params]
        self.config["galaxy-params"] = params
        self.config["galaxy-sections"] = sections
        self.config["galaxy"] = galaxy

    def prepare_legendre_polynomials(self, options):
        """Prepare the set of Legendre polynomials used during the fit.

        Parameters
        ----------
        options : :class:`DataBlock`
            Input options to initialise the model.
        """
        _log("\n-> Configuring multiplicative polynomial")
        if options.has_value("legendre_deg"):
            kwargs = {}
            if options.has_value("legendre_bounds"):
                kwargs["bounds"] = options["legendre_bounds"]
            if options.has_value("legendre_scale"):
                kwargs["scale"] = options["legendre_scale"]
            if options.has_value("legendre_clip_first_zero"):
                kwargs["clip_first_zero"] = options["legendre_clip_first_zero"]
            _log(f"Using Legendre polynomials up to degree {options['legendre_deg']}",
                  "\nAdditional arguments: ", kwargs)
            self.config["legendre_pol"] = spectrum.get_legendre_polynomial_array(
                self.config["wavelength"], options["legendre_deg"], **kwargs)
        else:
            _log(f"Not using multiplicative Legendre polynomials")
        _log("-> Configuration done")

    def plot_solution(self, solution: DataBlock, figname=None):
        """Plot the fit."""
        flux_model = self.make_observable(solution, parse=True)
        if isinstance(flux_model, tuple):
            weights = flux_model[1]
            flux_model = flux_model[0]
        else:
            weights = np.ones_like(flux_model)
        # Include input weights
        weights *= self.config["weights"]

        # Grab the solution values (visualuzation purpose only)
        sol_keys = solution.keys()
        sol_sections = {}
        for (sec, name) in sol_keys:
            if sec not in sol_sections:
                sol_sections[sec] = {name: solution[sec, name]}
            else:
                sol_sections[sec].update({name: solution[sec, name]})

        fig, axs = plt.subplots(ncols=2, nrows=2, sharex="col", sharey="row",
                                constrained_layout=True,
                                width_ratios=[4, 1],
                                height_ratios=[2, 1],
                                figsize=(16, 9))
        plt.suptitle(f"Module: {self.name}")

        # Display the information
        ax = axs[0, 1]
        # Pixel masking information
        mask_info = {"Total pixels": self.config["flux"].size,
                     "Masked pixels (w=0)": np.sum(weights <= 0),
                     " - Telluric": np.sum(
                        self.config.get("telluric_mask", 0)),
                     " - Emission lines": np.sum(
                        self.config.get("emission_lines_mask", 0))}
        
        sol_sections["settings"] = {"use_transforms": self.config.get("use_transforms", False)}

        if sol_sections["settings"]["use_transforms"]:
            sfh_params_latent = self.config["sfh_model"].get_sfh_parameters_array(solution)
            sfh_params_phys = self.config["sfh_model"].to_physical(sfh_params_latent)

            for key, value in zip(self.config["sfh_model"].sfh_bin_keys,
                                  sfh_params_phys):
                sol_sections[self.config["sfh_model"].sect_name][key] = value

        sections = [(k, v) for k, v in sol_sections.items()]
        sections = sections + [("Masking", mask_info)]
        _ = draw_dict_in_axes(ax, sections, section_spacing=1,
                          title_style="underline")
        ax.axis("off")
        # Plot input spectra and best-fit model
        ax = axs[0, 0]
        snr = np.nanpercentile(
            self.config["flux"] / np.sqrt(self.config["var"]),
            (16, 50, 84)
        )
        ax.annotate(f"SNR (16, 50, 84 percentiles): "
                    f"{snr[0]:.1f}, {snr[1]:.1f}, {snr[2]:.1f}",
                    xy=(0.02, 0.98), xycoords="axes fraction", va="top",
                    fontsize=8)
        ax.fill_between(
            self.config["wavelength"].value,
            self.config["flux"] - self.config["var"] ** 0.5,
            self.config["flux"] + self.config["var"] ** 0.5,
            color="k",
            alpha=0.5,
        )
        ax.plot(
            self.config["wavelength"], self.config["flux"], c="k",
            label="Observed", lw=0.7)
        # Show masked pixels
        nan_mask = np.ones_like(self.config["flux"])
        nan_mask[weights <= 0] = np.nan
        ax.plot(
            self.config["wavelength"],
            self.config["flux"] * nan_mask,
            c="r",
            lw=0.7,
            label="Non-zero weights",
        )
        # Plot model
        ax.plot(self.config["wavelength"], flux_model, c="b", label="Model",
                lw=0.7)
        # Plot residuals
        residuals = flux_model - self.config["flux"]
        ax.plot(
            self.config["wavelength"],
            residuals,
            c="grey",
            label="Residuals",
            lw=0.7
        )
        ax.plot(
            self.config["wavelength"],
            residuals * nan_mask,
            c="orange",
            label="Residuals (w>0)",
            lw=0.7
        )
        ax.axhline(0, ls="--", color="k", alpha=0.2)
        ax.set_ylabel("Flux")
        ax.legend(bbox_to_anchor=(0.5, 1.01), loc="lower center",
                  ncols=5, fontsize=8)

        p5, p95 = np.nanpercentile(self.config["flux"], [5, 95])
        p_residuals = np.nanpercentile(residuals, 5) * 0.95
        ax.set_ylim(np.min([p_residuals, p5 * 0.8]), p95 * 1.2)

        # Plot chi2
        good_pixels = weights > 0
        chi2 = (flux_model - self.config["flux"]) ** 2 / self.config["var"]
        mean_chi2 = np.nanmean(chi2[good_pixels])
        median_chi2 = np.nanmedian(chi2[good_pixels])
        nmad_chi2 = 1.4826 * np.nanmedian(
            np.abs(chi2[good_pixels] - median_chi2))
        loglike = self.log_like(self.config["flux"][good_pixels],
                                flux_model[good_pixels],
                                self.config["var"][good_pixels],
                                weights=weights[good_pixels])
        ax = axs[1, 0]
        ax.plot(self.config["wavelength"], chi2, c="k", lw=0.7)
        ax.grid(visible=True)
        ax.set_ylabel(r"$\chi^2$")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xlabel("Wavelength (AA)")
        
        ax = axs[1, 1]
        ax.hist(
            chi2,
            bins=np.geomspace(0.01, 100),
            orientation="horizontal",
            color="k",
            histtype="step"
        )
        ax.annotate(f"Median chi2: {np.nanmedian(chi2):.1f}"
                    + f"\nMean chi2: {mean_chi2:.1f}"
                    + f"\nNMAD chi2: {nmad_chi2:.1f}"
                    + f"\nLog-likelihood: {loglike:.1f}",
                    xy=(0.05, 0.95), xycoords="axes fraction", va="top",
                    fontsize=8)
        ax.set_xlabel("No. pixels")
        ax.grid(visible=True)
        ax.tick_params(labelleft=False)

        if figname is not None:
            fig.savefig(figname, bbox_inches="tight",
                    dpi=300)
            _log(f"Fit plot saved at: {figname}")

        plt.close()
        return fig


class PhotometryFitModule(BaseModule):
    """Base class for photometry fitting modules in BESTA."""
    
    def prepare_observed_photometry(self, options: SectionOptions):
        """Prepare the Photometric Data.

        Parameters
        ----------
        options : :class:`DataBlock`
        """
        _log("\n-> Configuring photometric data")
        photometry_file = os.path.expandvars(options["inputPhotometry"])

        # Read the data
        input_data = ascii.read(photometry_file)
        filter_names = input_data[input_data.colnames[0]].value.astype(str)
        flux = input_data[input_data.colnames[1]].value.astype(float)
        flux_err = input_data[input_data.colnames[2]].value.astype(float)
        # Handle upper and lower limits
        if len(input_data.colnames) > 3:
            measure_limits = input_data[input_data.colnames[3]].value.astype(str)
            is_upper = measure_limits == "upper"
            is_lower = measure_limits == "lower"
        else:
            is_upper = None
            is_lower = None

        # Unit conversion

        if options.has_value("fluxUnits"):
            _log("Converting input flux units to uJy")
            flux_units = u.Unit(options["fluxUnits"])
            flux = (flux << flux_units).to("uJy").value
            flux_err = (flux_err << flux_units).to("uJy").value
            flux_units = u.Unit("uJy")
        else:
            _log("Assuming input flux units are in uJy")
            flux_units = u.Unit("uJy")

        self.config["photometry_flux"] = flux
        self.config["photometry_flux_var"] = flux_err**2
        self.config["photometry_flux_unit"] = flux_units
        self.config["photometry_upper_limit"] = is_upper
        self.config["photometry_lower_limit"] = is_lower

        # Load the photometric filters
        photometric_filters = []
        for filter_name in filter_names:
            _log(f"Loading photometric filter: {filter_name}")
            if os.path.exists(os.path.expandvars(filter_name)):
                filt = Filter.from_text_file(os.path.expandvars(filter_name))
            else:
                filt = Filter.from_svo(filter_name)
            photometric_filters.append(filt)
        # TODO: For now save both for backwards compatibility
        # TODO: Superseed by FilterList
        self.config["filters"] = photometric_filters
        self.config["filter_list"] = FilterList(photometric_filters)

        redshift = options.get_double("redshift", default=0.0)
        self.config["redshift"] = redshift
        _log("Source redshift: ", redshift)
        _log("-> Configuration done.")

    def prepare_galaxy(self, options):
        """Build and configure a :class:`pst.galaxy.GalaxySED` model for photometry.

        The method initializes stellar, attenuation, and optional dust emission
        components, prepares the target wavelength grid, and stores the galaxy
        model and parameter index in ``self.config``.
        """
        
        # Stellar emissions
        ssp_model = self.config.get("ssp_model")
        if ssp_model is None:
            self.prepare_ssp_model(options)
            ssp_model = self.config["ssp_model"]

        sfh_model = self.config.get("sfh_model")
        if sfh_model is None:
            self.prepare_sfh_model(options)
            sfh_model = self.config["sfh_model"].model

        filters = self.config.get("filter_list")
        if filters is None:
            self.prepare_observed_photometry(options)

        ## Create stellar emission model
        _log("Setting up stellar emission component")
        stars = sed.StellarComponent(ssp=ssp_model, sfh=sfh_model)

        # Dust extinction and emission
        dust_attenuation = None
        dust_emission = None
        if options.get_bool("DustAttenuation", False):
            _log("Setting up dust attenuation")
            att_name = options.get_string(
                "DustAttenuationModel", "DustScreenAttenuation")
            ext_law = options.get_string("ExtinctionLaw", "ccm89")
            dust_curve = dust.ExtinctionLibCurve(law=ext_law)
            _log("DurstAttenuationModel: ", att_name)
            # TODO: implement interface for other models
            if att_name == "DustScreenAttenuation":
                dust_attenuation = dust.DustScreenAttenuation(curve=dust_curve)

            if options.get_bool("DustEmission", False):
                _log("Setting up dust emission model")
                dust_sed = dust.Casey2012DustComponent()
                if options.get_bool("DustCalorimetric", True):
                    _log("Setting energy balance approximation")
                    dust_emission = dust.CalorimetricDustComponent(
                        attenuation=dust_attenuation,
                        dust_sed_component=dust_sed
                    )

        # Setup target wavelength range
        min_wl, max_wl = filters.wavelength_range()
        if dust_emission is not None:
            min_wl = np.min((
                min_wl.to_value(u.AA),
                stars.ssp.wavelength.min().to_value("AA"))) << u.AA            
            max_wl = np.max((
                max_wl.to_value(u.AA),
                getattr(dust_emission.dust_sed_component, "ir_range", [None, 1 << u.AA])[1].to_value(u.AA)
            )) << u.AA

        z_obs = self.config.get("redshift", 0.0)

        if options.get_bool("logwave", False):
            target_wl = np.geomspace(min_wl.to_value("AA") / (1 + z_obs),
                                     max_wl.to_value("AA"), 3000) << u.AA
        else:
            target_wl = np.arange(min_wl.to_value("AA") / (1 + z_obs),
                              max_wl.to_value("AA"),
                              10) << u.AA
        _log("Target wavelength range: ",
              f"{target_wl[0]:.1f} -- {target_wl[-1]:.1f} ({target_wl.size} pix)")

        _log("Setting up galaxy model")
        galaxy = GalaxySED(stellar_model=stars,
                           dust_attenuation_model=dust_attenuation,
                           dust_model=dust_emission,
                           target_wavelength=target_wl,
                           redshift=z_obs,
                           cosmology=cosmology,
                           filters=filters)

        params = galaxy.build_param_index(include_fixed=False, prefix="")
        sections = [s.rsplit(".", 1) for s in params]
        self.config["galaxy-params"] = params
        self.config["galaxy-sections"] = sections
        self.config["galaxy"] = galaxy

    def plot_solution(self, solution: DataBlock, figname=None):
        """Plot the fit."""
        flux_model, full_spec = self.make_observable(solution, parse=True, include_spec=True)
        if isinstance(flux_model, tuple):
            flux_model = flux_model[0]

        # Include input weights
        fig, axs = plt.subplots(ncols=2, nrows=3, sharex="col", sharey="row",
                                constrained_layout=True,
                                width_ratios=[4, 1],
                                height_ratios=[2, 1, 1],
                                figsize=(16, 9))
        flux_model = self.make_observable(solution, parse=True)
        # Grab the solution values (visualuzation purpose only)
        param_keys = [k[1] for k in solution.keys("parameters") if k[0] == "parameters"]
        param_val = [solution["parameters", k] for k in param_keys]

        fig, axs = plt.subplots(ncols=2, nrows=2, sharex="col", sharey="row",
                                constrained_layout=True,
                                squeeze=False,
                                width_ratios=[4, 1],
                                height_ratios=[4, 1],
                                figsize=(8, 5))
        plt.suptitle(f"Module: {self.name}")

        # Display the information
        ax = axs[0, 1]
        sol_keys = solution.keys()
        sol_sections = {}
        sol_sections["Settings"] = {
            "Redshift": self.config["redshift"],
            "No. bands": self.config["photometry_flux"].size,
            "use_transforms": self.config.get("use_transforms", False)}

        for (sec, name) in sol_keys:
            if sec not in sol_sections:
                sol_sections[sec] = {name: solution[sec, name]}
            else:
                sol_sections[sec].update({name: solution[sec, name]})

        if sol_sections["Settings"]["use_transforms"]:
        # Pixel masking information
        info = {"Total bands": self.config["photometry_flux"].size,
                     }
        model_params = dict(zip(param_keys, param_val))
        model_params["use_transforms"] = self.config.get(
            "use_transforms", False)
        if model_params["use_transforms"]:
            sfh_params_latent = self.config["sfh_model"].get_sfh_parameters_array(solution)
            sfh_params_phys = self.config["sfh_model"].to_physical(sfh_params_latent)

            for key, value in zip(self.config["sfh_model"].sfh_bin_keys,
                                  sfh_params_phys):
                sol_sections[self.config["sfh_model"].sect_name][key] = value

        sections = [(k, v) for k, v in sol_sections.items()]

        _ = draw_dict_in_axes(ax, sections, section_spacing=1,
                          title_style="underline")
        ax.axis("off")
        # Plot input photometry and model
        eff_wl = self.config["filter_list"].effective_wavelength
        ax = axs[0, 0]
        snr = np.nanpercentile(
            self.config["photometry_flux"] / np.sqrt(self.config["photometry_flux_var"]),
            (16, 50, 84)
        )
        ax.annotate(f"SNR (16, 50, 84 percentiles): "
                    f"{snr[0]:.1f}, {snr[1]:.1f}, {snr[2]:.1f}",
                    xy=(0.02, 0.98), xycoords="axes fraction", va="top",
                    fontsize=8)
        
        uplim = self.config["photometry_upper_limit"]
        uplim = uplim if uplim is not None else False
        lolim = self.config["photometry_lower_limit"]
        lolim = lolim if lolim is not None else False

        ax.errorbar(eff_wl.to_value("AA"), self.config["photometry_flux"],
                    yerr=np.sqrt(self.config["photometry_flux_var"]),
                    lolims=lolim, uplims=uplim,
                    capsize=2, label="Observed",
                    fmt="s", mec="k", mfc="none", ecolor="k"
                    )

        # Plot model
        ax.scatter(eff_wl.to_value("AA"), flux_model, edgecolors="r",
                   fc="none", lw=2, label="Model")
        ax.plot(self.config["galaxy"].target_wavelength.to_value("AA"),
                full_spec, color="k", alpha=0.4)
        # Plot residuals
        ax.set_ylabel(f"Flux density ({self.config['photometry_flux_unit']})")
        ax.legend(bbox_to_anchor=(0.5, 1.01), loc="lower center",
                  ncols=5, fontsize=8)

        ax.set_ylim(min(flux_model.min(), self.config["photometry_flux"].min()) * 0.8,
                    max(flux_model.max(), self.config["photometry_flux"].max()) * 1.2)

        # If wavelength range is too large, use log scale
        wlmin, wlmax = self.config["galaxy"].target_wavelength[[0, -1]]
        if wlmax / wlmin > 10:
            ax.set_xscale("log")
            ax.set_yscale("log")

        twax = ax.twinx()
        for f in self.config["filter_list"].filters:
            twax.plot(f.filter_wavelength, f.filter_resp / f.filter_resp.max(), label=f.name)
        twax.legend(fontsize=6, loc="lower right")
        twax.set_ylabel("Filter response")

        min_wl, max_wl = self.config["filter_list"].wavelength_range()
        ax.set_xlim(min_wl.to_value(u.AA) * 0.8, max_wl.to_value(u.AA) * 1.2)
        # Flux density per wavelength unit
        ax = axs[1, 0]
        flam = (full_spec * u.Unit("uJy")).to("1e-16 erg / (s cm**2 AA)", u.spectral_density(self.config["galaxy"].target_wavelength))
        ax.plot(
            self.config["galaxy"].target_wavelength.to_value("AA"),
            flam,
            color="k", alpha=0.4)
        ax.set_ylabel("Flux density (1e-16 erg / (s cm**2 AA))")
        # chi2 as function of wavelength
        chi2 = (flux_model - self.config["photometry_flux"]) ** 2 / self.config["photometry_flux_var"]
        ax = axs[2, 0]
        ax.scatter(eff_wl.to_value("AA"), chi2, c="k")
        ax.grid(visible=True)
        ax.set_ylabel(r"$\chi^2$")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_ylim(0, np.nanmax(chi2) * 2)
        ax.set_xlabel("Wavelength (AA)")

        ax = axs[1, 1]
        ax.axis("off")
        ax = axs[2, 1]
        ax.axis("off")

        if figname is not None:
            fig.savefig(figname, bbox_inches="tight",
                    dpi=300)
            _log(f"Fit plot saved at: {figname}")
        plt.close()
        return fig

class EquivalentWidthFitModule(BaseModule):
    """Base class for equivalent width fit modules in BESTA."""
    pass

    def plot_solution(self, solution: DataBlock, figname=None):
        pass

class GridFitMixin:
    """Mixin class for grid-based fitting modules in BESTA."""

    def _load_callable_from_file(self,
        file_path: str | pathlib.Path,
        func_name: str,
        *,
        module_name: Optional[str] = None,
    ) -> Callable:
        """
        Load a callable named `func_name` from a Python source file at `file_path`.

        Parameters
        ----------
        file_path
            Path to the .py file (does not need to be importable / on PYTHONPATH).
        func_name
            Name of the function (or other callable) defined in that file.
        module_name
            Optional module name to assign during loading. If None, a unique
            name is generated from the filename.

        Returns
        -------
        func
            The loaded callable object.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ImportError
            If the module cannot be loaded.
        AttributeError
            If func_name is not found in the module.
        TypeError
            If the loaded attribute is not callable.
        """
        file_path = pathlib.Path(file_path).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(str(file_path))
        if file_path.suffix != ".py":
            raise ImportError(f"Expected a .py file, got: {file_path}")

        # Give the module a deterministic-ish name to help debugging and caching
        if module_name is None:
            module_name = f"_user_boundary_{file_path.stem}"

        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for: {file_path}")

        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
        except Exception as e:
            raise ImportError(f"Error importing {file_path}: {e}") from e

        obj = getattr(module, func_name)  # may raise AttributeError
        if not callable(obj):
            raise TypeError(f"{func_name!r} in {file_path} is not callable (got {type(obj)})")

        return obj

    def prepare_grid_model(self, options):
        """Prepare the model grid.

        Parameters
        ----------
        options : :class:`DataBlock`
            Input options to initialise the model.
        """
        print("\n-> Configuring model grid")
        if not options.has_value("modelGridFile"):
            raise ValueError("No input model grid file provided.")
        grid_file = os.path.expandvars(options["modelGridFile"])
        print("Loading model grid from file: ", grid_file)
        if not os.path.isfile(grid_file):
            raise FileNotFoundError(f"Input model grid file {grid_file} not found.")
        print("Reading model grid... fluxes must be in microJansky / Msun")
        model_grid = ModelGrid.load_auto(grid_file)

        if options.has_value("boundaryFunctionFile"):
            boundary_file = os.path.expandvars(
                options["boundaryFunctionFile"])
            print("Loading boundary function from file: ", boundary_file)
            # Split the path to the file and the function name given by []
            mthd_s = boundary_file.find("[")
            mthd_e = boundary_file.find("]")
            boundary_func_name = boundary_file[mthd_s + 1:mthd_e]
            boundary_file = boundary_file[:mthd_s]
            boundary_func = self._load_callable_from_file(
                boundary_file, boundary_func_name)
            model_grid.check_boundaries = boundary_func
            print("Applied boundary function to model grid.")

        self.config["model_grid"] = model_grid
        
        if options.has_value("knn"):
            self.config["knn"] = options["knn"]
        else:
            self.config["knn"] = int(4 * model_grid.n_targets)
        print("-> Configuration done.")


class EmulatorMixin:
    """Mixin class for modules using ML emulators in BESTA."""

    def prepare_emulator(self, options):
        """Prepare the ML emulator.

        Parameters
        ----------
        options : :class:`DataBlock`
            Input options to initialise the model.
        
        Notes
        -----
        The ML emulator is expected to be stored in a joblib (.joblib) file.
        """
        try:
            import joblib
        except ImportError:
            raise ImportError("joblib is required to load ML emulators."
                              "Please install joblib and try again.")
        print("\n-> Configuring ML emulator")
        if not options.has_value("emulatorFile"):
            raise ValueError("No input emulator file provided.")
        emulator_file = os.path.expandvars(options["emulatorFile"])
        print("Loading ML emulator from file: ", emulator_file)
        if not os.path.isfile(emulator_file):
            raise FileNotFoundError(f"Input emulator file {emulator_file} not found.")
        print("Reading ML emulator...")
        ml_emulator = joblib.load(emulator_file)
        self.config["ml_emulator"] = ml_emulator
        print("-> Configuration done.")
        self.config["emulator"] = ml_emulator
        print("-> Configuration done.")
