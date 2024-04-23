import os
import numpy as np

from sklearn.decomposition import NMF

from cosmosis.datablock import option_section, names as section_names, DataBlock
from cosmosis.runtime import Inifile
from cosmosis.utils import underline

import hbsps.specBasics as specBasics
from hbsps import output

def prepare_observed_spectra(cosmosis_options, config, module=None):
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
        wl_range = np.array(wl_range.split(" "), dtype=float)
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
    wavelength /= (1 + redshift)
    print("Constraining fit to wavelength range: ", wl_range)
    goodIdx = np.where(
		(wavelength >= wl_range[0]) & (wavelength <= wl_range[1])
					)[0]
    wavelength = wavelength[goodIdx]
    flux = flux[goodIdx]
    cov = error[goodIdx]**2
    print("Log-binning spectra to velocity scale: ", velscale)
    flux, ln_wave, _ = specBasics.log_rebin(wavelength, flux, velscale=velscale)
    cov, _, _ = specBasics.log_rebin(wavelength, cov, velscale=velscale)
    wavelength = np.exp(ln_wave)
    # Normalize spectra
    print("Spectra normalized using wavelength range: ", wl_norm_range)
    normIdx = np.where((wavelength >= wl_norm_range[0]) & (
        wavelength <= wl_norm_range[1]))[0]
    norm_flux = np.nanmedian(flux[normIdx])
    flux /= norm_flux
    cov /= norm_flux**2
    config['flux'] = flux
    config['cov'] = cov
    config['wavelength'] = wavelength
    config['ln_wave'] = ln_wave
    return flux, cov, wavelength, ln_wave

def prepare_ssp_data(cosmosis_options, config, module=None):
    print(underline("\nConfiguring SSP parameters"))
    # Wavelegth range to renormalize the spectra
    if type(cosmosis_options) is DataBlock:
        wl_norm_range = cosmosis_options[option_section, "wlNormRange"]
        velscale = cosmosis_options[option_section, "velscale"]
        oversampling = cosmosis_options[option_section, "oversampling"]
        ssp_name = cosmosis_options[option_section, "SSPModel"]
        ssp_dir = cosmosis_options[option_section, "SSPDir"]
        if cosmosis_options.has_value(option_section, "SSPModelArgs"):
            ssp_args = cosmosis_options.get_string(option_section, "SSPModelArgs")
            ssp_args = ssp_args.split(",")
        else:
            ssp_args = []
        if cosmosis_options.has_value(option_section, "SSPSave"):
            ssp_save = cosmosis_options.get_bool(option_section, "SSPSave")
        else:
            ssp_save = False
        do_nmf = cosmosis_options.has_value(option_section, "SSP-NMF")
        if do_nmf:
            do_nmf = cosmosis_options.get_bool(option_section, "SSP-NMF")
            n_nmf = cosmosis_options.get_int(option_section, "SSP-NMF-N")

    elif type(cosmosis_options) is Inifile:
        wl_norm_range = cosmosis_options[module, "wlNormRange"]
        wl_norm_range = wl_norm_range.strip("[]")
        wl_norm_range = np.array(wl_norm_range.split(" "), dtype=float)
        velscale = cosmosis_options[module, "velscale"]
        velscale = float(velscale.strip("[]"))
        oversampling = cosmosis_options[module, "oversampling"]
        oversampling = int(oversampling.strip("[]"))
        ssp_name = cosmosis_options[module, "SSPModel"]
        ssp_dir = cosmosis_options[module, "SSPDir"]
        ssp_args = cosmosis_options.get(module, "SSPModelArgs", fallback=[])
        if len(ssp_args) > 0:
            ssp_args = ssp_args.split(",")
        # ssp_save = cosmosis_options.get(module, "SSPSave", fallback=False)
        ssp_save = False
        do_nmf = cosmosis_options.getboolean(module, "SSP-NMF")
        if do_nmf:
            n_nmf = cosmosis_options[module, "SSP-NMF-N"]
            n_nmf = int(n_nmf.strip("[]"))
    try:
        from pst import SSP
    except:
        raise ImportError("PST module was not found")
    
    ssp = getattr(SSP, ssp_name)

    if ssp_dir == 'None':
        ssp_dir = None
    else:
        print(f"SSP model directory: {ssp_dir}")
    print("SSP Model extra arguments: ", ssp_args)
    ssp = ssp(*ssp_args, path=ssp_dir)
    # Renormalize SSP Light-to-mass ratio
    ssp_lmr = 1 / ssp.get_mass_lum_ratio(wl_norm_range)
    ssp.L_lambda /= ssp_lmr[:, :, np.newaxis]

    # Rebin the spectra
    print("Log-binning SSP spectra to velocity scale: ", velscale / oversampling, " km/s")
    dlnlam = velscale / specBasics.constants.c.to('km/s').value
    extra_offset_pixel = int(300 / velscale)
    dlnlam /= oversampling
    lnlam_bin_edges = np.arange(
        config['ln_wave'][0] - dlnlam * extra_offset_pixel * oversampling - 0.5 * dlnlam,
        config['ln_wave'][-1] + dlnlam * (1 + extra_offset_pixel) * oversampling + 0.5 * dlnlam,
        dlnlam)
    ssp.interpolate_sed(np.exp(lnlam_bin_edges))

    # Reshape the SSPs SED to (n_models, wavelength)
    ssp_sed = ssp.L_lambda.reshape(
        (ssp.L_lambda.shape[0] * ssp.L_lambda.shape[1], ssp.L_lambda.shape[2]))
    ssp_wl = ssp.wavelength
    # ------------------------------ Decomposition --------------------------- #
    if do_nmf:
        print("Reducing dimensionality with Non-negative Matrix Factorisation")
        pca = NMF(n_components=n_nmf, alpha_H=1.0, max_iter=1000)
        pca.fit(ssp_sed)
        ssp_sed = pca.components_
    # ------------------------------------------------------------------------ #
    print("Final SSP model shape: ", ssp_sed.shape)
    if ssp_save is not None:
        output.save_ssp(
            filename=os.path.join(
                os.path.dirname(cosmosis_options['output', 'filename']),
                "SSP_model.dat"),
            ssp_wave=ssp_wl, ssp_sed=ssp_sed)

    config['ssp_sed'] = ssp_sed
    config['ssp_wl'] = ssp_wl
    # Grid parameters
    config['velscale'] = velscale
    config['oversampling'] = oversampling
    config['extra_pixels'] = extra_offset_pixel
    return 

def prepare_extinction_law(cosmosis_options, config, module=None):
    print(underline("\nConfiguring dust extinction parameters"))
    if type(cosmosis_options) is DataBlock:
        if not cosmosis_options.has_value(option_section, "ExtinctionLaw"):
            config['extinction_law'] = None
            config['norm_extinction'] = None
            return
        ext_law = cosmosis_options.get_string(option_section, "ExtinctionLaw")
        wl_norm_range = cosmosis_options[option_section, "wlNormRange"]
    elif type(cosmosis_options) is Inifile:
        if not cosmosis_options.get(module, "ExtinctionLaw", fallback=False):
            config['extinction_law'] = None
            config['norm_extinction'] = None
            return
        ext_law = cosmosis_options[module, "ExtinctionLaw"]
        wl_norm_range = cosmosis_options[module, "wlNormRange"]
        wl_norm_range = wl_norm_range.strip("[]")
        wl_norm_range = np.array(wl_norm_range.split(" "), dtype=float)

    try:
        import extinction
    except:
        raise ImportError("extinction library was not found")
    extinction_law = getattr(extinction, ext_law)
    config['extinction_law'] = extinction_law
    # Wavelegth range to renormalize the spectra
    normIdx = np.where((config['wavelength'] >= wl_norm_range[0]) & (
        config['wavelength'] <= wl_norm_range[1]))[0]
    
    norm_extinction = np.median(
        extinction_law(config['wavelength'][normIdx], 1.0, 3.1))
    config['norm_extinction'] = norm_extinction
    return
