"""
Base pipeline module. This module contains the base class for creating new
pipeline modules in BESTA.
"""
from abc import abstractmethod
import os
import pickle
import sys

from matplotlib import pyplot as plt
from besta.visualization import draw_dict_in_axes

import numpy as np
from sklearn.decomposition import NMF
from astropy import units as u

from cosmosis import ClassModule
from cosmosis import DataBlock
from cosmosis.datablock import SectionOptions, option_section

from pst.utils import flux_conserving_interpolation
from pst.observables import Filter
from pst import SSP, dust

from besta import spectrum
from besta import kinematics
from besta import sfh
from besta import io
from besta.config import cosmology, memory


class BaseModule(ClassModule):
    """BESTA Pipeline module base class."""

    def __init__(self, options):
        """Set-up the COSMOSIS sampler.
        Args:
            options: options from startup file (i.e. .ini file)
        """
        options = self.parse_options(options)
        self.config = {}
        # Likelihood name
        if options.has_value("like_name"):
            self.like_name = options["like_name"]
            if "_like" not in self.like_name:
                self.like_name += "_like"
        else:
            self.like_name = self.name + "_like"
            print("Setting module likelihood name to default: ", self.like_name)

    @abstractmethod
    def make_observable(self, *args, **kwargs):
        """Create an observable from an input set of model parameters."""

    @abstractmethod
    def execute(self, block: DataBlock, config: dict):
        """Execute the pipeline."""
        return super().execute(block, config)

    @abstractmethod
    def plot_fit(self, *args, **kwargs):
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
        options : :class:`DataBlock`
        """
        if isinstance(options, dict):
            options = DataBlock.from_dict(options)
            if options.has_section(option_section):
                options._delete_section(option_section)
            keys = options.keys(self.name)
            if not keys:
                raise ValueError(f"No options found for module {self.name}")
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
        print("\n-> Configuring SSP model")

        if options.has_value("SSPModelFromPickle"):
            print("\n-> Loading preconfigured SSP model from pickle")
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
            print("-> Configuration done.")
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

        # Rebin the spectra
        dlnlam = velscale / spectrum.constants.c.to("km/s").value
        extra_offset_pixel = int(velocity_buffer / velscale)
        print(
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
        print("SSP Model SED dimensions (met, age, lambda): ", ssp.L_lambda.shape)

        # Convolve with instrumental LSF
        if "lsf" in self.config:
            print("Convolving SSP model with instrumental LSF")
            inst_lsf = np.interp(ssp.wavelength, self.config["wavelength"],
                                 self.config["lsf"])

            if options.has_value("SSPLSF"):
                print("Including SSP resolution")
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
            print("Starting convolution of SSP models with wavelength-dependent",
                  f"LSF [min sigma={lsf_sigma_pixels.min():.2},"
                  f" max sigma={lsf_sigma_pixels.max():.2} pix]")
            try:
                io.check_array_memory(
                (ssp.L_lambda.shape[0], ssp.L_lambda.shape[1],
                 ssp.L_lambda.shape[2], ssp.L_lambda.shape[2]),
                dtype=ssp.L_lambda[0, 0, 0].dtype, unit='GB',
                safety_margin=memory["ram_safety_margin"])
                print("Convolving full SSP model at once")
                ssp.L_lambda = kinematics.convolve_variable_gaussian_kernel(
                    ssp.L_lambda, lsf_sigma_pixels)
            except MemoryError:
                # Do a loop along metallicity axis to prevent memory overflows
                print("Insufficient RAM memory for full SSP SED convolution")
                print("Looping along metallicity axis")
                io.check_array_memory(
                (ssp.L_lambda.shape[1], ssp.L_lambda.shape[2],
                 ssp.L_lambda.shape[2]),
                dtype=ssp.L_lambda[0, 0, 0].dtype, unit='GB',
                safety_margin=memory["ram_safety_margin"])

                for ith in range(ssp.L_lambda.shape[0]):
                    ssp.L_lambda[ith] = kinematics.convolve_variable_gaussian_kernel(
                    ssp.L_lambda[ith], lsf_sigma_pixels)

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

        self.config["ssp_model"] = ssp
        self.config["ssp_sed"] = ssp_sed
        self.config["ssp_wl"] = ssp.wavelength.to_value("Angstrom")
        # Grid parameters
        self.config["velscale"] = velscale
        self.config["extra_pixels"] = extra_offset_pixel
        if options.has_value("SaveSSPModel"):
            print("Saving photometry grid to ", options["SaveSSPModel"])
            with open(os.path.expandvars(options["SaveSSPModel"]), 'wb') as file:
                pickle.dump(ssp, file, pickle.HIGHEST_PROTOCOL)
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
        # TODO: add more extinction laws
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
        i = 1
        while options.has_value(f"SFHArgs{i}"):
            key = f"SFHArgs{i}"
            i += 1
            value = options[key]
            if isinstance(value, str):
                if "," in value:
                    value = np.array(value.split(","), dtype=float)
            sfh_args.append(value)
        print("SFH model name: ", sfh_model_name)
        sfh_model = getattr(sfh, sfh_model_name)
        sfh_model = sfh_model(*sfh_args, **self.config)
        self.config["sfh_model"] = sfh_model
        print("-> Configuration done")

    def log_like(self, data, model, cov, weights=None):
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
        chi2 = (model - data)**2 / cov
        if weights is not None:
            loglike = -0.5 * np.sum(chi2 * weights) / np.sum(weights)
        else:
            loglike = -0.5 * np.sum(chi2)

        return loglike


class SpectraFitModule(BaseModule):
    """Base class for spectral fitting modules in BESTA."""

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
        filename = os.path.expandvars(options["inputSpectrum"])
        # Read wavelength and spectra
        print("Loading observed spectra from input file: ", filename)
        wavelength, flux, error = np.loadtxt(filename, unpack=True)
        print("Wavelength coverage: ", wavelength[[0, -1]])
        print("Size: ", wavelength.size)

        # Convert units if needed
        if options.has_value("wlUnits"):
            print("Converting wavelength units to Angstrom")
            wl_units = u.Unit(options["wlUnits"])
            wavelength = (wavelength << wl_units).to("Angstrom").value
        else:
            print("Assuming input wavelength units are in Angstrom")
            wl_units = u.angstrom

        if options.has_value("fluxUnits"):
            print("Converting flux units to 1e-16 erg/s/cm^2/Angstrom")
            flux_units = u.Unit(options["fluxUnits"])
            flux = (flux << flux_units).to(
                "1e-16 erg / (s cm2 Angstrom)").value
            error = (error << flux_units).to(
                "1e-16 erg / (s cm2 Angstrom)").value
        else:
            print("Assuming input flux units are in 1e-16 erg/s/cm^2/Angstrom")
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
            print("No input redshift value provided (defaulting to 0)")
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
        # Apply redshift
        print(f"Setting wavelength array to restframe (redshift: {redshift})")
        wavelength /= 1.0 + redshift
        print("Constraining fit to wavelength range: ", wl_range)
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

        print("Number of selected pixels within wavelength range: ", good_idx.size)
        if options.has_value("velscale"):
            velscale = options["velscale"]
        else:
            # Set velscale to None
            velscale = None
        print("Log-binning spectra to velocity scale: ", velscale, " (km/s)")
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
            flux_units = u.dimensionless_unscaled
        else:
            norm_flux = 1.0

        if luminosity:
            if redshift > 0:
                print(f"Converting input flux to luminosity at redshift {redshift}")
                dl_sq = cosmology.luminosity_distance(redshift).to("cm").value ** 2
                dl_sq = 4 * np.pi * dl_sq * (1 + redshift)
            else:
                print("Converting input flux to luminosity at 10 pc")
                dl_sq = (10 * u.pc).to("cm").value ** 2 * 4 * np.pi
            #  Input spectra is expected to be a specific flux density per
            # wavelength unit.
            flux *= dl_sq
            cov *= dl_sq * dl_sq

        self.config["flux"] = flux
        self.config["cov"] = cov
        self.config["redshift"] = redshift
        self.config["wlUnits"] = wl_units
        self.config["fluxUnits"] = flux_units
        self.config["norm_flux"] = norm_flux
        self.config["wavelength"] = wavelength << u.angstrom
        self.config["ln_wave"] = ln_wave
        self.config["weights"] = weights
        if not (instrumental_lsf == 0).all():
            self.config["lsf"] = instrumental_lsf

        print("-> Configuration done.")

    def prepare_legendre_polynomials(self, options):
        """Prepare the set of Legendre polynomials used during the fit.

        Parameters
        ----------
        options : :class:`DataBlock`
            Input options to initialise the model.
        """
        print("\n-> Configuring multiplicative polynomial")
        if options.has_value("legendre_deg"):
            kwargs = {}
            if options.has_value("legendre_bounds"):
                kwargs["bounds"] = options["legendre_bounds"]
            if options.has_value("legendre_scale"):
                kwargs["scale"] = options["legendre_scale"]
            if options.has_value("legendre_clip_first_zero"):
                kwargs["clip_first_zero"] = options["legendre_clip_first_zero"]
            print(f"Using Legendre polynomials up to degree {options['legendre_deg']}",
                  "\nAdditional arguments: ", kwargs)
            self.config["legendre_pol"] = spectrum.get_legendre_polynomial_array(
                self.config["wavelength"], options["legendre_deg"], **kwargs)
        else:
            print(f"Not using multiplicative Legendre polynomials")
        print("-> Configuration done")

    def plot_fit(self, solution: DataBlock, figname=None):
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
        param_keys = [k[1] for k in solution.keys("parameters") if k[0] == "parameters"]
        param_val = [solution["parameters", k] for k in param_keys]

        fig, axs = plt.subplots(ncols=2, nrows=2, sharex="col", sharey="row",
                                constrained_layout=True,
                                width_ratios=[4, 1],
                                height_ratios=[2, 1],
                                figsize=(np.round(flux_model.size / 300, 0), 6))
        plt.suptitle(f"Module: {self.name}")

        # Display the solution
        ax = axs[0, 1]
        ax.set_title("Model parameters")
        text = draw_dict_in_axes(ax, dict(zip(param_keys, param_val)))
        ax.axis("off")
        # Plot input spectra and best-fit model
        ax = axs[0, 0]
        ax.fill_between(
            self.config["wavelength"].value,
            self.config["flux"] - self.config["cov"] ** 0.5,
            self.config["flux"] + self.config["cov"] ** 0.5,
            color="k",
            alpha=0.5,
        )
        ax.plot(
            self.config["wavelength"], self.config["flux"], c="k", label="Observed",
            lw=0.7)
        # Show masked pixels
        nan_mask = np.ones_like(self.config["flux"])
        nan_mask[weights <= 0] = np.nan
        ax.plot(
            self.config["wavelength"],
            self.config["flux"] * nan_mask,
            c="r",
            lw=0.7,
            label="Masked",
        )
        # Plot model
        ax.plot(self.config["wavelength"], flux_model, c="b", label="Model",
                lw=0.7)
        # Plot residuals
        residuals = flux_model - self.config["flux"]
        ax.plot(
            self.config["wavelength"],
            residuals,
            c="orange",
            label="Residuals",
            lw=0.7
        )
        ax.axhline(0, ls="--", color="k", alpha=0.2)
        ax.set_ylabel("Flux")
        ax.legend(bbox_to_anchor=(0.5, 1.01), loc="lower center",
                  ncols=4, fontsize=8)

        p5, p95 = np.nanpercentile(self.config["flux"], [5, 95])
        p_residuals = np.nanpercentile(residuals, 5) * 0.95
        ax.set_ylim(np.min([p_residuals, p5 * 0.8]), p95 * 1.2)

        # Plot chi2
        good_pixels = weights > 0
        chi2 = (flux_model - self.config["flux"]) ** 2 / self.config["cov"]
        mean_chi2 = np.nanmean(chi2[good_pixels])
        median_chi2 = np.nanmedian(chi2[good_pixels])
        nmad_chi2 = 1.4826 * np.nanmedian(
            np.abs(chi2[good_pixels] - median_chi2))
        loglike = self.log_like(self.config["flux"][good_pixels],
                                flux_model[good_pixels],
                                self.config["cov"][good_pixels],
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
            print(f"Fit plot saved at: {figname}")

        plt.close()
        return fig


class PhotometryFitModule(BaseModule):
    """Base class for photometry fitting modules in BESTA."""
    
    def prepare_observed_photometry(self, options):
        """Prepare the Photometric Data.

        Parameters
        ----------
        options : :class:`DataBlock`
        """
        print("\n-> Configuring photometric data")
        photometry_file = os.path.expandvars(options["inputPhotometry"])

        # Read the data
        filter_names = np.loadtxt(photometry_file, usecols=0, dtype=str)
        flux, flux_err = np.loadtxt(
            photometry_file, usecols=(1, 2), unpack=True, dtype=float
        )

        nanomaggie = u.def_unit('nanomaggie', 3631e-9 * u.Jy)
        if options.has_value("fluxUnits"):
            print("Converting flux units to nanomaggies")
            flux_units = u.Unit(options["fluxUnits"])
            flux = (flux << flux_units).to(nanomaggie).value
            flux_err = (flux_err << flux_units).to(nanomaggie).value
        else:
            print("Assuming input flux units are in nanomaggies")
            flux_units = nanomaggie

        self.config["photometry_flux"] = flux
        self.config["photometry_flux_var"] = flux_err**2
        self.config["photometry_flux_units"] = flux_units
        # TODO: include redshift and flux conversion to luminosities
        # Load the photometric filters
        photometric_filters = []
        for filter_name in filter_names:
            print(f"Loading photometric filter: {filter_name}")
            if os.path.exists(os.path.expandvars(filter_name)):
                filt = Filter.from_text_file(os.path.expandvars(filter_name))
            else:
                filt = Filter.from_svo(filter_name)
            photometric_filters.append(filt)
        self.config["filters"] = photometric_filters

        if options.has_value("flux_to_lum"):
            if options["flux_to_lum"] == True:
                print("Converting input fluxes to absolute flux at 10 pc using"
                      f"input redshift {options['redshift']}")
                distance = cosmology.luminosity_distance(
                    options["redshift"]).to_value("10 pc")
                self.config["photometry_flux"] *= distance**2
                self.config["photometry_flux_var"] *= distance**4

        print("-> Configuration done.")

    def plot_fit(self, solution: DataBlock, figname=None):
        pass

class EquivalentWidthFitModule(BaseModule):
    """Base class for equivalent width fit modules in BESTA."""
    pass

    def plot_fit(self, solution: DataBlock, figname=None):
        pass