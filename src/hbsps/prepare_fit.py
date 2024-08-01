import os
import numpy as np
from astropy.io import fits
from astropy import units as u
from sklearn.decomposition import NMF

from cosmosis.datablock import option_section, names as section_names, DataBlock
from cosmosis.runtime import Inifile
from cosmosis.utils import underline

import hbsps.specBasics as specBasics
from hbsps import output
from hbsps.utils import cosmology
from hbsps import dust_extinction, sfh
from hbsps import kinematics


try:
    from pst import SSP, observables
except:
    raise ImportError("PST module was not found")


def prepare_observed_spectra(cosmosis_options, config, module=None,
                             normalize=True, luminosity=False):
    print(underline("\nConfiguring input observational data"))
    if type(cosmosis_options) is DataBlock:
        fileName = cosmosis_options[option_section, "inputSpectrum"]
        # Wavelegth range to include in the fit
        wl_range = cosmosis_options[option_section, "wlRange"]
        # Wavelegth range to renormalize the spectra
        wl_norm_range = cosmosis_options[option_section, "wlNormRange"]
        # Input redshift (initial guess)
        redshift = cosmosis_options[option_section, "redshift"]
        velscale = cosmosis_options[option_section, "velscale"]
    elif type(cosmosis_options) is Inifile:
        fileName = cosmosis_options[module, "inputSpectrum"]
        # Wavelegth range to include in the fit
        wl_range = cosmosis_options[module, "wlRange"]
        wl_range = wl_range.strip("[]")
        wl_range = wl_range.split(" ")
        # Remove rogue spaces
        while "" in wl_range:
            wl_range.remove("")
        wl_range = np.array(wl_range, dtype=float)
        # Wavelegth range to renormalize the spectra
        wl_norm_range = cosmosis_options[module, "wlNormRange"]
        wl_norm_range = wl_norm_range.strip("[]")
        wl_norm_range = np.array(wl_norm_range.split(" "), dtype=float)
        # Input redshift (initial guess)
        redshift = cosmosis_options[module, "redshift"]
        redshift = float(redshift.strip("[]"))
        velscale = cosmosis_options[module, "velscale"]
        velscale = float(velscale.strip("[]"))

    # Read wavelength and spectra
    wavelength, flux, error = np.loadtxt(fileName, unpack=True)
    print(f"Input redshift: {redshift}")
    wavelength /= 1 + redshift
    print("Constraining fit to wavelength range: ", wl_range)
    goodIdx = np.where((wavelength >= wl_range[0]) & (wavelength <= wl_range[1]))[0]
    wavelength = wavelength[goodIdx]
    flux = flux[goodIdx]
    cov = error[goodIdx] ** 2
    print("Log-binning spectra to velocity scale: ", velscale)
    flux, ln_wave, _ = specBasics.log_rebin(wavelength, flux, velscale=velscale)
    cov, _, _ = specBasics.log_rebin(wavelength, cov, velscale=velscale)
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

    config["flux"] = flux
    config["cov"] = cov
    config["norm_flux"] = norm_flux
    config["wavelength"] = wavelength
    config["ln_wave"] = ln_wave
    print(underline("\nConfiguration done"))
    return flux, cov, wavelength, ln_wave


def prepare_observed_photometry(cosmosis_options=None, config={}, module=None):
    """Prepare the Photometric Data."""
    print(underline("\nConfiguring photometric data"))
    if type(cosmosis_options) is DataBlock:
        photometry_file = cosmosis_options[option_section, "inputPhotometry"]
    elif type(cosmosis_options) is Inifile:
        photometry_file = cosmosis_options[module, "inputPhotometry"]

    # Read the data
    filter_names = np.loadtxt(photometry_file, usecols=0, dtype=str)
    # Assuming flux units == nanomaggies
    flux, flux_err = np.loadtxt(
        photometry_file, usecols=(1, 2), unpack=True, dtype=float)
    config['photometry_flux'] = flux
    config['photometry_flux_var'] = flux_err**2

    # Load the photometric filters
    photometric_filters = []
    for filter_name in filter_names:
        print(f"Loading photometric filter: {filter_name}")
        if os.path.exists(filter_name):
            f = observables.Filter(filter_path=filter_name)
        else:
            f = observables.Filter(filter_name=filter_name)
        photometric_filters.append(f)
    config['filters'] = photometric_filters
    return config


def prepare_ssp_from_fits(filename, config={}):
    """Load the SSP from a FITS file."""
    print(f"Loading SSP model from FITS file: {filename}")
    with fits.open(filename) as hdul:
        config["velscale"] = hdul[0].header["velscale"]
        config["oversampling"] = hdul[0].header["oversampling"]
        config["extra_pixels"] = hdul[0].header["extra_pixels"]

        config["ssp_wl"] = hdul["WAVE"].data
        config["ssp_sed"] = hdul["SED"].data
        config["ssp_metals_edges"] = hdul["METALS_EDGES"].data
        config["ssp_ages_edges"] = hdul["AGES_EDGES"].data
    return config


def prepare_ssp_model(cosmosis_options=None, config={}, module=None,
                      normalize=False):
    """Prepare the SSP data."""
    print(underline("\nConfiguring SSP parameters"))
    # Wavelegth range to renormalize the spectra
    if type(cosmosis_options) is DataBlock:
        velscale = cosmosis_options[option_section, "velscale"]
        oversampling = cosmosis_options[option_section, "oversampling"]
        ssp_name = cosmosis_options[option_section, "SSPModel"]
        ssp_dir = cosmosis_options[option_section, "SSPDir"]

        if cosmosis_options.has_value(option_section, "SSPModelArgs"):
            ssp_args = cosmosis_options.get_string(option_section, "SSPModelArgs")
            ssp_args = ssp_args.split(",")
        else:
            ssp_args = []

        if cosmosis_options.has_value(option_section, "wlNormRange"):
            wl_norm_range = cosmosis_options[option_section, "wlNormRange"]
        else:
            wl_norm_range = None
        do_nmf = cosmosis_options.has_value(option_section, "SSP-NMF")
        if do_nmf:
            do_nmf = cosmosis_options.get_bool(option_section, "SSP-NMF")
            n_nmf = cosmosis_options.get_int(option_section, "SSP-NMF-N")

    elif type(cosmosis_options) is Inifile:
        velscale = cosmosis_options[module, "velscale"]
        velscale = float(velscale.strip("[]"))
        oversampling = cosmosis_options[module, "oversampling"]
        oversampling = int(oversampling.strip("[]"))
        ssp_name = cosmosis_options[module, "SSPModel"]
        ssp_dir = cosmosis_options[module, "SSPDir"]
        ssp_args = cosmosis_options.get(module, "SSPModelArgs", fallback=[])
        if len(ssp_args) > 0:
            ssp_args = ssp_args.split(",")
 
        if cosmosis_options.get(module, "wlNormRange", fallback='none') != 'none':
            wl_norm_range = cosmosis_options[module, "wlNormRange"]
            wl_norm_range = wl_norm_range.strip("[]")
            wl_norm_range = np.array(wl_norm_range.split(" "), dtype=float)
        else:
            wl_norm_range = None

        do_nmf = cosmosis_options.getboolean(module, "SSP-NMF", fallback=False)
        if do_nmf:
            n_nmf = cosmosis_options[module, "SSP-NMF-N"]
            n_nmf = int(n_nmf.strip("[]"))

    ssp = getattr(SSP, ssp_name)

    if ssp_dir == "None":
        ssp_dir = None
    else:
        print(f"SSP model directory: {ssp_dir}")
    print("SSP Model extra arguments: ", ssp_args)
    ssp = ssp(*ssp_args, path=ssp_dir)
    # Rebin the spectra
    print(
        "Log-binning SSP spectra to velocity scale: ", velscale / oversampling, " km/s"
    )
    dlnlam = velscale / specBasics.constants.c.to("km/s").value
    dlnlam /= oversampling

    if "ln_wave" in config:
        ln_wl_edges = config["ln_wave"][[0, -1]]
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
    print("SED shape: ", ssp.L_lambda.shape)

    if normalize:
        print("Normalizing SSPs")
        mlr = ssp.get_specific_mass_lum_ratio(wl_norm_range)
        ssp.L_lambda = (ssp.L_lambda.value * mlr.value[:, :, np.newaxis]
        ) * ssp.L_lambda.unit

    ssp_sed = ssp.L_lambda.value.reshape(
        (ssp.L_lambda.shape[0] * ssp.L_lambda.shape[1], ssp.L_lambda.shape[2]))
    
    if do_nmf:
        print("Reducing SSP dimensionality with Non-negative Matrix Factorisation")
        pca = NMF(n_components=n_nmf, alpha_H=1.0, max_iter=n_nmf * 1000)
        pca.fit(ssp_sed)
        ssp_sed = pca.components_
    # ------------------------------------------------------------------------ #
    config["ssp_model"] = ssp
    config["ssp_sed"] = ssp_sed
    config["ssp_wl"] = ssp.wavelength.to_value("Angstrom")
    # Grid parameters
    config["velscale"] = velscale
    config["oversampling"] = oversampling
    config["extra_pixels"] = extra_offset_pixel
    print(underline("\nConfiguration done"))
    return


def prepare_ssp_model_preprocessing(cosmosis_options, config, module=None):
    print("Preprocessing SSP model")
    if type(cosmosis_options) is DataBlock:
        if cosmosis_options.has_value(option_section, "los_vel"):
            if cosmosis_options.has_value(option_section, "los_sigma"):
                print(f"Convolving SSP models with Gaussian LOSF")
                ssp, mask = kinematics.convolve_ssp_model(
                    config,
                    cosmosis_options[option_section, "los_sigma"],
                    cosmosis_options[option_section, "los_vel"])
                config['ssp_model'] = ssp
                config['mask'] = mask
                print("Valid pixels: ", np.count_nonzero(mask), mask.size)
        else:
            print("No kinematic information was provided")

        if cosmosis_options.has_value(option_section, "ExtinctionLaw"):
            av = cosmosis_options[option_section, "av"]
            print(f"Reddening SSP models using Av={av}")
            dust_extinction.redden_ssp_model(config, av)
    elif type(cosmosis_options) is Inifile:
        los_vel = cosmosis_options.getfloat(module, "los_vel", fallback=np.nan)
        los_sigma = cosmosis_options.getfloat(module, "los_sigma", fallback=np.nan)
        print(module, los_sigma, los_vel)
        if (los_vel is not np.nan) and (los_sigma is not np.nan):
            print(f"Convolving SSP models with Gaussian LOSF")
            ssp, mask = kinematics.convolve_ssp_model(
                config, los_sigma, los_vel)
            config['ssp_model'] = ssp
            config['mask'] = mask
            print("Valid pixels: ", np.count_nonzero(mask), mask.size)
        else:
            print("No kinematic information was provided")
        ext_law = cosmosis_options.get(module, "ExtinctionLaw", fallback=None)
        if ext_law:
            av = cosmosis_options.getfloat(module, "av", fallback=0.0)
            print(f"Reddening SSP models using Av={av}")
            dust_extinction.redden_ssp_model(config, av)


def prepare_extinction_law(cosmosis_options, config, module=None):
    print(underline("\nConfiguring dust extinction parameters"))
    if type(cosmosis_options) is DataBlock:
        if not cosmosis_options.has_value(option_section, "ExtinctionLaw"):
            config["extinction_law"] = None
            config["norm_extinction"] = None
            return
        ext_law = cosmosis_options.get_string(option_section, "ExtinctionLaw")
        wl_norm_range = cosmosis_options[option_section, "wlNormRange"]
    elif type(cosmosis_options) is Inifile:
        if not cosmosis_options.get(module, "ExtinctionLaw", fallback=False):
            config["extinction_law"] = None
            config["norm_extinction"] = None
            return
        ext_law = cosmosis_options[module, "ExtinctionLaw"]
        wl_norm_range = cosmosis_options[module, "wlNormRange"]
        wl_norm_range = wl_norm_range.strip("[]")
        wl_norm_range = np.array(wl_norm_range.split(" "), dtype=float)

    extinction_law = dust_extinction.DustScreen(ext_law_name=ext_law,
                                                wave_norm_range=wl_norm_range)
    config["extinction_law"] = extinction_law
    print(underline("\nConfiguration done"))
    return


def prepare_sfh_model(cosmosis_options=None, config={}, module=None):
    """Prepare the SFH model."""
    print(underline("\nConfiguring SFH model"))
    if type(cosmosis_options) is DataBlock:
        sfh_model_name = cosmosis_options[option_section, "SFHModel"]
        sfh_args = []
        key = "SFHArgs1"
        while cosmosis_options.has_value(option_section, key):
            sfh_args.append(cosmosis_options[option_section, key])
            key = key.replace(key[-1], str(int(key[-1]) + 1))
    elif type(cosmosis_options) is Inifile:
        sfh_model_name = cosmosis_options[module, "SFHModel"]
        sfh_args = []
        key = "SFHArgs1"
        while cosmosis_options.get(module, key, fallback=False):
            sfh_args.append(cosmosis_options[module, key])
            key = key.replace(key[-1], str(int(key[-1]) + 1))

    sfh_model = getattr(sfh, sfh_model_name)
    sfh_model = sfh_model(*sfh_args, **config)
    sfh_model.make_ini(cosmosis_options["pipeline", "values"])
    config["sfh_model"] = sfh_model
    print(underline("\nConfiguration done"))

    return



