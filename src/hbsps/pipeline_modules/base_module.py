from abc import abstractmethod
import os
import numpy as np
from sklearn.decomposition import NMF
from astropy import units as u

from cosmosis import ClassModule
from cosmosis import DataBlock
from cosmosis.datablock import SectionOptions, option_section

from pst.observables import Filter
from pst import SSP, dust

from hbsps import specBasics
from hbsps import sfh
from hbsps.utils import cosmology


class BaseModule(ClassModule):

    @abstractmethod
    def make_observable():
        pass

    @abstractmethod
    def execute(self, block, config):
        return super().execute(block, config)

    def parse_options(self, options):
        if isinstance(options, dict):
            options = DataBlock.from_dict(options)
            if options.has_section(option_section):
                options._delete_section(option_section)
            for (section, name) in options.keys(self.name):
                options[option_section, name] = options[section, name]
            options = SectionOptions(options)
        return options

    def prepare_observed_spectra(self, options, normalize=True, luminosity=False):
        print("\n-> Configuring input observed spectra")
        fileName = options["inputSpectrum"]
        # Wavelegth range to include in the fit
        wl_range = options["wlRange"]
        # Wavelegth range to renormalize the spectra
        wl_norm_range = options["wlNormRange"]
        # Input redshift (initial guess)
        redshift = options["redshift"]
        velscale = options["velscale"]

        # Read wavelength and spectra
        print("Loading observed spectra from input file: ", fileName)
        wavelength, flux, error = np.loadtxt(fileName, unpack=True)
        print("Wavelength coverage: ", wavelength[[0, -1]])
        print("Size: ", wavelength.size)
        # Load mask
        if options.has_value("mask"):
            weights = np.array(np.loadtxt(options["mask"]), dtype=float)
        else:
            weights = np.ones_like(flux)
        # Apply redshift 
        print(f"Setting to restframe with respect to input redshift: {redshift}")
        wavelength /= 1 + redshift
        print("Constraining fit to wavelength range: ", wl_range)
        goodIdx = np.where((wavelength >= wl_range[0]) & (wavelength <= wl_range[1]))[0]
        wavelength = wavelength[goodIdx]            
        flux = flux[goodIdx]
        cov = error[goodIdx] ** 2
        weights = weights[goodIdx]
        print("Number of selected pixels: ", goodIdx.size)
        print("Log-binning spectra to velocity scale: ", velscale, " (km/s)")
        flux, ln_wave, _ = specBasics.log_rebin(wavelength, flux, velscale=velscale)
        cov, _, _ = specBasics.log_rebin(wavelength, cov, velscale=velscale)
        weights, _, _ = specBasics.log_rebin(wavelength, weights, velscale=velscale)
        wavelength = np.exp(ln_wave)
        # Normalize spectra
        print("Spectra normalized using wavelength range: ", wl_norm_range)
        normIdx = np.where(
            (wavelength >= wl_norm_range[0]) & (wavelength <= wl_norm_range[1])
        )[0]
        norm_flux = np.nanmedian(flux[normIdx])

        if normalize:
            flux /= norm_flux
            cov /= norm_flux**2
        if luminosity:
            dl_sq = cosmology.luminosity_distance(redshift).to('cm').value**2
            dl_sq = 4 * np.pi * dl_sq**2
            dl_sq = np.max((dl_sq, 1.0))
            flux *= dl_sq
            cov *= dl_sq

        self.config["flux"] = flux
        self.config["cov"] = cov
        self.config["norm_flux"] = norm_flux
        self.config["wavelength"] = wavelength * u.angstrom
        self.config["ln_wave"] = ln_wave
        self.config["weights"] = weights
        print("-> Configuration done.")

    def prepare_observed_photometry(self, options):
        """Prepare the Photometric Data."""
        print("\n-> Configuring photometric data")
        photometry_file = options["inputPhotometry"]

        # Read the data
        filter_names = np.loadtxt(photometry_file, usecols=0, dtype=str)
        # Assuming flux units == nanomaggies
        flux, flux_err = np.loadtxt(
            photometry_file, usecols=(1, 2), unpack=True, dtype=float)
        self.config['photometry_flux'] = flux
        self.config['photometry_flux_var'] = flux_err**2

        # Load the photometric filters
        photometric_filters = []
        for filter_name in filter_names:
            print(f"Loading photometric filter: {filter_name}")
            if os.path.exists(filter_name):
                f = Filter.from_text_file(filter_name)
            else:
                f = Filter.from_svo(filter_name)
            photometric_filters.append(f)
        self.config['filters'] = photometric_filters
        print("-> Configuration done.")

    def prepare_ssp_model(self, options, normalize=False):
        """Prepare the SSP data."""
        print("\n-> Configuring SSP model")
        velscale = options["velscale"]
        oversampling = options["oversampling"]
        ssp_name = options["SSPModel"]
        ssp_dir = options["SSPDir"]

        if options.has_value("SSPModelArgs"):
            ssp_args = options.get_string("SSPModelArgs")
            ssp_args = ssp_args.split(",")
            print("SSP Model extra arguments: ", ssp_args)
        else:
            ssp_args = []

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
        
        ssp = getattr(SSP, ssp_name)(*ssp_args, path=ssp_dir)
        # Rebin the spectra
        print(
            "Log-binning SSP spectra to velocity scale: ", velscale / oversampling, " km/s"
        )
        dlnlam = velscale / specBasics.constants.c.to("km/s").value
        dlnlam /= oversampling

        if "ln_wave" in self.config:
            ln_wl_edges = self.config["ln_wave"][[0, -1]]
            extra_offset_pixel = int(300 / velscale)
        else:
            ln_wl_edges = np.log(ssp.wavelength[[0, -1]].to_value("angstrom"))
            extra_offset_pixel = 0

        lnlam_bin_edges = np.arange(
            ln_wl_edges[0]
            - dlnlam * extra_offset_pixel * oversampling
            - 0.5 * dlnlam,
            ln_wl_edges[-1]
            + dlnlam * (1 + extra_offset_pixel) * oversampling
            + 0.5 * dlnlam,
            dlnlam,
        )
        ssp.interpolate_sed(np.exp(lnlam_bin_edges))
        print("SSP Model SED shape (met, age, lambda): ", ssp.L_lambda.shape)

        if normalize:
            print("Normalizing SSP model SED within range ", wl_norm_range)
            mlr = ssp.get_specific_mass_lum_ratio(wl_norm_range)
            ssp.L_lambda = (ssp.L_lambda.value * mlr.value[:, :, np.newaxis]
            ) * ssp.L_lambda.unit

        ssp_sed = ssp.L_lambda.value.reshape(
            (ssp.L_lambda.shape[0] * ssp.L_lambda.shape[1], ssp.L_lambda.shape[2]))
        
        if n_nmf is not None:
            print("Reducing SSP model dimensionality with Non-negative Matrix Factorisation",
                  "\nNo. of components: ", n_nmf)
            pca = NMF(n_components=n_nmf, alpha_H=1.0, max_iter=n_nmf * 1000)
            pca.fit(ssp_sed)
            ssp_sed = pca.components_
        # ------------------------------------------------------------------------ #
        self.config["ssp_model"] = ssp
        self.config["ssp_sed"] = ssp_sed
        self.config["ssp_wl"] = ssp.wavelength.to_value("Angstrom")
        # Grid parameters
        self.config["velscale"] = velscale
        self.config["oversampling"] = oversampling
        self.config["extra_pixels"] = extra_offset_pixel
        print("-> Configuration done.")
        return

    def prepare_extinction_law(self, options):
        print("\n -> Configuring Dust extinction model")
        if not options.has_value("ExtinctionLaw"):
            self.config["extinction_law"] = None
            return
        ext_law = options.get_string("ExtinctionLaw")
        wl_norm_range = options["wlNormRange"]
        print("Extinction law: ", ext_law)
        self.config["extinction_law"] = dust.DustScreen(ext_law)
        print("-> Configuration is done.")

    def prepare_sfh_model(self, options):
        """Prepare the SFH model."""
        print("\n-> Configuring SFH model")
        sfh_model_name = options["SFHModel"]
        sfh_args = []
        key = "SFHArgs1"
        while options.has_value(key):
            value = options[key]
            if "," in value:
                value = np.array(value.split(","), dtype=float)
            sfh_args.append(value)
            key = key.replace(key[-1], str(int(key[-1]) + 1))
        print("SFH model name: ", sfh_model_name)
        sfh_model = getattr(sfh, sfh_model_name)
        sfh_model = sfh_model(*sfh_args, **self.config)
        self.config["sfh_model"] = sfh_model
        print("-> Configuration done")

    def X2min(self, spectrum, recSp, cov):
        # Determine residual, divide first residual vector by 
        # diagonal elements covariance matrix.
        residual1 = recSp - spectrum
        residual2 = np.copy(residual1)
        residual1 /= cov
            
        # Determine likelihood term (i.e. X2-value)
        chiSq = -0.5 * np.dot(residual1, residual2)
        
        return chiSq