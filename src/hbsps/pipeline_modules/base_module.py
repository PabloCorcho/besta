"""
Base pipeline module. This module contains the base class for creating new
pipeline modules in BESTA.
"""
from abc import abstractmethod
import os
import sys

import numpy as np
from sklearn.decomposition import NMF
from astropy import units as u

from cosmosis import ClassModule
from cosmosis import DataBlock
from cosmosis.datablock import SectionOptions, option_section

from pst.observables import Filter
from pst import SSP, dust

from hbsps import spectrum
from hbsps import sfh
from hbsps.config import cosmology


class BaseModule(ClassModule):
    """BESTA Pipeline module base class."""

    @abstractmethod
    def make_observable(*args, **kwargs):
        """Create an observable from an input set of model parameters."""

    @abstractmethod
    def execute(self, block, config):
        """Execute the pipeline."""
        return super().execute(block, config)

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
        options : :class:`DataBlock`
        """
        if isinstance(options, dict):
            options = DataBlock.from_dict(options)
            if options.has_section(option_section):
                options._delete_section(option_section)
            for section, name in options.keys(self.name):
                options[option_section, name] = options[section, name]
            options = SectionOptions(options)
        return options

    def prepare_observed_spectra(
        self, options: DataBlock, normalize=False, luminosity=False
    ):
        """Prepare the input spectra data.

        Parameters
        ----------
        options : :class:`DataBlock`
        normalize : bool, optional
            If ``True``, normalizes the spectra using the given wavelength range.
        luminosity : bool, optional
            If ``True``, converts the input flux to luminosities.
        """
        print("\n-> Configuring input observed spectra")
        filename = options["inputSpectrum"]
        # Read wavelength and spectra
        print("Loading observed spectra from input file: ", filename)
        wavelength, flux, error = np.loadtxt(filename, unpack=True)
        print("Wavelength coverage: ", wavelength[[0, -1]])
        print("Size: ", wavelength.size)
        # Wavelegth range to include in the fit
        if options.has_value("wlRange"):
            wl_range = options["wlRange"]
        else:
            wl_range = wavelength[[0, -1]]
        # Wavelegth range to renormalize the spectra
        if options.has_value("wlNormRange"):
            wl_norm_range = options["wlNormRange"]
        else:
            wl_norm_range = wavelength[[0, -1]]
        # Input redshift (initial guess)
        if options.has_value("redshift"):
            redshift = options["redshift"]
        else:
            print("No input redshift value provided (defaulting to 0)")
            redshift = 0.0
        # Load mask
        if options.has_value("mask"):
            weights = np.array(np.loadtxt(options["mask"]), dtype=float)
        else:
            weights = np.ones_like(flux)
        # Apply redshift
        print(f"Setting to restframe with respect to input redshift: {redshift}")
        wavelength /= 1.0 + redshift
        print("Constraining fit to wavelength range: ", wl_range)
        good_idx = np.where((wavelength >= wl_range[0]) & (wavelength <= wl_range[1]))[0]
        wavelength = wavelength[good_idx]
        flux = flux[good_idx]
        cov = error[good_idx] ** 2
        weights = weights[good_idx]
        print("Number of selected pixels within wavelength range: ", good_idx.size)
        if options.has_value("velscale"):
            velscale = options["velscale"]
        else:
            # Set velscale to None
            velscale = None
        print("Log-binning spectra to velocity scale: ", velscale, " (km/s)")
        # Update the value of velscale
        flux, ln_wave, velscale = spectrum.log_rebin(
            wavelength, flux, velscale=velscale
        )
        cov, _, _ = spectrum.log_rebin(wavelength, cov, velscale=velscale)
        weights, _, _ = spectrum.log_rebin(wavelength, weights, velscale=velscale)
        wavelength = np.exp(ln_wave)
        print("Number of pixels after interpolation: ", wavelength.size)
        # Normalize spectra
        if normalize:
            print("Spectra normalized using wavelength range: ", wl_norm_range)
            norm_idx = np.where(
                (wavelength >= wl_norm_range[0]) & (wavelength <= wl_norm_range[1])
            )[0]
            norm_flux = np.nanmedian(flux[norm_idx])
            flux /= norm_flux
            cov /= norm_flux**2
        else:
            norm_flux = 1.0

        if luminosity:
            dl_sq = cosmology.luminosity_distance(redshift).to("cm").value ** 2
            #  Input spectra is expected to be a specific flux density per
            # wavelength unit.
            dl_sq = 4 * np.pi * dl_sq**2 * (1 + redshift)
            dl_sq = np.max((dl_sq, 1.0))
            flux *= dl_sq
            cov *= dl_sq

        self.config["flux"] = flux
        self.config["cov"] = cov
        self.config["norm_flux"] = norm_flux
        self.config["wavelength"] = wavelength << u.angstrom
        self.config["ln_wave"] = ln_wave
        self.config["weights"] = weights
        print("-> Configuration done.")

    def prepare_observed_photometry(self, options):
        """Prepare the Photometric Data.

        Parameters
        ----------
        options : :class:`DataBlock`
        """
        print("\n-> Configuring photometric data")
        photometry_file = options["inputPhotometry"]

        # Read the data
        filter_names = np.loadtxt(photometry_file, usecols=0, dtype=str)
        # Assuming flux units == nanomaggies
        flux, flux_err = np.loadtxt(
            photometry_file, usecols=(1, 2), unpack=True, dtype=float
        )
        self.config["photometry_flux"] = flux
        self.config["photometry_flux_var"] = flux_err**2
        # TODO: include redshift and flux conversion to luminosities
        # Load the photometric filters
        photometric_filters = []
        for filter_name in filter_names:
            print(f"Loading photometric filter: {filter_name}")
            if os.path.exists(filter_name):
                filt = Filter.from_text_file(filter_name)
            else:
                filt = Filter.from_svo(filter_name)
            photometric_filters.append(filt)
        self.config["filters"] = photometric_filters
        print("-> Configuration done.")

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
        print("\n-> Configuring SSP model")

        ssp_name = options["SSPModel"]
        ssp_dir = options["SSPDir"]

        if "none" in ssp_dir.lower():
            ssp_dir = None

        # Additional arguments to be passed to the SSP model
        if options.has_value("SSPModelArgs"):
            ssp_args = options.get_string("SSPModelArgs")
            ssp_args = ssp_args.split(",")
            print("SSP Model extra arguments: ", ssp_args)
        else:
            ssp_args = []

        ssp = getattr(SSP, ssp_name)(*ssp_args, path=ssp_dir)

        # Parameters to format the templates to the input spectra
        velscale = options["velscale"]

        if options.has_value("wlNormRange"):
            wl_norm_range = options["wlNormRange"]
        else:
            wl_norm_range = None

        if options.has_value("SSP-NMF-N"):
            n_nmf = options.get_int("SSP-NMF-N")
        else:
            n_nmf = None

        if ssp_dir == "None":
            ssp_dir = None
        else:
            print(f"Loading SSP model from input directory: {ssp_dir}")

        # Rebin the spectra
        dlnlam = velscale / spectrum.constants.c.to("km/s").value
        extra_offset_pixel = int(velocity_buffer / velscale)
        print("Log-binning SSP spectra to velocity scale: ", velscale, " km/s",
              f"\nKeeping {extra_offset_pixel} extra pixels at both edges")

        if "ln_wave" in self.config:
            ln_wl_edges = self.config["ln_wave"][[0, -1]]
            # Add extra pixels at the edges to prevent corruption during convolution
        else:
            ln_wl_edges = np.log(ssp.wavelength[[0, -1]].to_value("angstrom"))
            extra_offset_pixel = 0

        lnlam_bin_edges = np.arange(
            ln_wl_edges[0] - 0.5 * dlnlam  - dlnlam * extra_offset_pixel,
            ln_wl_edges[-1] + dlnlam * (1 + extra_offset_pixel),
            dlnlam)

        # Resample the SED
        ssp.interpolate_sed(np.exp(lnlam_bin_edges))
        print("SSP Model SED dimensions (met, age, lambda): ", ssp.L_lambda.shape)

        if normalize and wl_norm_range is not None:
            print("Normalizing SSP model SED within range ", wl_norm_range)
            mlr = ssp.get_specific_mass_lum_ratio(wl_norm_range)
            ssp.L_lambda = (
                ssp.L_lambda.value * mlr.value[:, :, np.newaxis]
            ) * ssp.L_lambda.unit

        # Reshape the SSP model from (metal, age, wave) -> (metal * age, wave)
        ssp_sed = ssp.L_lambda.value.reshape(
            (ssp.L_lambda.shape[0] * ssp.L_lambda.shape[1], ssp.L_lambda.shape[2])
        )
        # Apply Non-negative Matrix Factorisation for reducing dimensionality
        if n_nmf is not None:
            print(
                "Reducing SSP model dimensionality with Non-negative Matrix Factorisation",
                "\nNo. of components: ",
                n_nmf,
            )
            # TODO: hard-coded parameters
            pca = NMF(n_components=n_nmf, alpha_H=1.0, max_iter=n_nmf * 1000)
            pca.fit(ssp_sed)
            ssp_sed = pca.components_
        # ------------------------------------------------------------------------ #
        self.config["ssp_model"] = ssp
        self.config["ssp_sed"] = ssp_sed
        self.config["ssp_wl"] = ssp.wavelength.to_value("Angstrom")
        # Grid parameters
        self.config["velscale"] = velscale
        self.config["extra_pixels"] = extra_offset_pixel
        print("-> Configuration done.")
        return

    def prepare_extinction_law(self, options):
        """Prepare an dust extinction model.

        options : :class:`DataBlock`
            Input options to initialise the model.
        """
        print("\n -> Configuring Dust extinction model")
        if not options.has_value("ExtinctionLaw"):
            self.config["extinction_law"] = None
            return
        ext_law = options.get_string("ExtinctionLaw")
        print("Extinction law: ", ext_law)
        self.config["extinction_law"] = dust.DustScreen(ext_law)
        print("-> Configuration is done.")

    def prepare_sfh_model(self, options):
        """Prepare the SFH model.

        Parameters
        ----------
        options : :class:`DataBlock`
            Input options to initialise the model.
        """
        print("\n-> Configuring SFH model")
        sfh_model_name = options["SFHModel"]
        sfh_args = []
        key = "SFHArgs1"
        while options.has_value(key):
            value = options[key]
            if isinstance(value, str):
                if "," in value:
                    value = np.array(value.split(","), dtype=float)
            sfh_args.append(value)
            key = key.replace(key[-1], str(int(key[-1]) + 1))
        print("SFH model name: ", sfh_model_name)
        sfh_model = getattr(sfh, sfh_model_name)
        sfh_model = sfh_model(*sfh_args, **self.config)
        self.config["sfh_model"] = sfh_model
        print("-> Configuration done")

    def log_like(self, data, model, cov):
        """Compute the likelihood between an input data set and a model.

        Parameters
        ----------
        data : np.ndarray
            Input data array
        model : np.ndarray
            Input model
        cov : np.ndarray
            Covariance matrix.

        Returns
        -------
        loglike : np.ndarray
            The log-likelihood associated to the model given the data.
        """
        chi = (model - data) / cov
        loglike = -0.5 * np.sum(chi * chi)
        return loglike
