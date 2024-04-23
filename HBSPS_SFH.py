import os
import sys
import numpy as np
from scipy.signal import convolve

from cosmosis.datablock import option_section, names as section_names
import hbsps.specBasics as specBasics
from pst import SSP

import extinction


def X2min(spectrum, recSp, cov):
	# Determine residual, divide first residual vector by 
	# diagonal elements covariance matrix.
	residual1 = recSp - spectrum
	residual2 = np.copy(residual1)
	residual1 /= cov
		
	# Determine likelihood term (i.e. X2-value)
	chiSq = -0.5 * np.dot(residual1, residual2)
	
	return chiSq

def make_values_file(values_file, n_ssp, los_v_lims=[-300, 300],
					 sigma_lims=[10, 300], inc_extinction=True,
					 av_lims=[0.0, 3.0]):
	print(f"Creating values file: {values_file}")
	with open(values_file, "w") as f:
		f.write("[parameters]\n")
		for i in range(1, n_ssp):
			f.write(f"ssp{i} = -4 -0.3 0\n")
		f.write(
			f"los_vel = {los_v_lims[0]} {(los_v_lims[0] + los_v_lims[1]) / 2} {los_v_lims[1]}\n")
		f.write(
			f"sigma = {sigma_lims[0]} {(sigma_lims[0] + sigma_lims[1]) / 2} {sigma_lims[1]}\n")
		if inc_extinction:
			f.write(
			f"av = {av_lims[0]} {(av_lims[0] + av_lims[1]) / 2} {av_lims[1]}\n")


def setup(options):
	"""Set-up the COSMOSIS sampler.
	
	Args:
		options: options from startup file (i.e. .ini file)
	Returns:
		config: parameters or objects that are passed to 
			the sampler.
			
	"""
	# Read paramaters/options from start-up file
	fileName = options[option_section, "inputSpectrum"]
	# Wavelegth range to include in the fit
	wl_range = options[option_section, "wlRange"]
	# Wavelegth range to renormalize the spectra
	wl_norm_range = options[option_section, "wlNormRange"]
	# SSP parameters
	ssp_name = options[option_section, "SSPModel"]	
	ssp_dir = options[option_section, "SSPDir"]
	if options.has_value("HBSPS_SFH", "SSPModelArgs"):
		ssp_args = options.get_string("HBSPS_SFH", "SSPModelArgs")
		ssp_args = ssp_args.split(",")
	else:
		ssp_args = []
	age_range = options[option_section, "ageRange"]
	met_range = options[option_section, "metRange"]
	# Dust extinction configuration
	if options.has_value("HBSPS_SFH", "ExtinctionLaw"):
		ext_law = options.get_string("HBSPS_SFH", "ExtinctionLaw")
		extinction_law = getattr(extinction, ext_law)
		inc_extinction = True
	else:
		extinction_law = None
		inc_extinction = False

	# Input redshift (initial guess)
	redshift = options[option_section, "redshift"]
	# Sampling configuration
	velscale = options[option_section, "velscale"]
	oversampling = options[option_section, "oversampling"]
	# Pipeline values file
	values_file = options[option_section, "values"]
	# ------------------------------------------------------------------------ #
	# Read wavelength and spectra
	wavelength, flux, error = np.loadtxt(fileName, unpack=True)
	# Set spectra in rest-frame
	print(f"Input redshift: {redshift}")
	wavelength /= (1 + redshift)

	# Select wavelength range that you want to use
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
	# ------------------------------------------------------------------------ #
    # Initialise SSP model
	ssp = getattr(SSP, ssp_name)
	if ssp_dir == 'None':
		ssp_dir = None
	else:
		print(f"SSP model directory: {ssp_dir}")
	print("SSP Model extra arguments: ", ssp_args)
	ssp = ssp(*ssp_args, path=ssp_dir)
	ssp.regrid(age_range, met_range)

	# Renormalize SSP Light-to-mass ratio
	ssp_lmr = 1 / ssp.get_mass_lum_ratio(wl_norm_range)
	ssp.L_lambda /= ssp_lmr[:, :, np.newaxis]

	# Rebin the spectra
	print("Log-binning SSP spectra to velocity scale: ", velscale / oversampling, " km/s")
	dlnlam = velscale / specBasics.constants.c.to('km/s').value
	extra_offset_pixel = int(300 / velscale)
	dlnlam /= oversampling
	lnlam_bin_edges = np.arange(
		ln_wave[0] - dlnlam * extra_offset_pixel * oversampling - 0.5 * dlnlam,
        ln_wave[-1] + dlnlam * (1 + extra_offset_pixel) * oversampling + 0.5 * dlnlam,
        dlnlam)
	ssp.interpolate_sed(np.exp(lnlam_bin_edges))

    # Reshape the SSPs SED to (n_models, wavelength)
	ssp_sed = ssp.L_lambda.reshape(
		(ssp.L_lambda.shape[0] * ssp.L_lambda.shape[1], ssp.L_lambda.shape[2]))
	ssp_wl = ssp.wavelength
	print("Final SSP model shape: ", ssp_sed.shape)

	if extinction_law is not None:
		norm_extinction = np.median(extinction_law(wavelength[normIdx], 1, 3.1))
	else:
		norm_extinction = None
	# ------------------------------------------------------------------------ #
	make_values_file(values_file=values_file, n_ssp=ssp_sed.shape[0],
				     inc_extinction=inc_extinction)
	config = {}
	# Observational information
	config['flux'] = flux
	config['cov'] = cov
	config['wavelength'] = wavelength
	# SSP model
	config['ssp_sed'] = ssp_sed
	config['ssp_wl'] = ssp_wl
	# Grid parameters
	config['velscale'] = velscale
	config['oversampling'] = oversampling
	config['extra_pixels'] = extra_offset_pixel
	# Dust extinction
	config['extinction_law'] = extinction_law
	config['norm_extinction'] = norm_extinction
	return config
	
def execute(block, config):
	"""Function executed by sampler
	This is the function that is executed many times by the sampler. The
	likelihood resulting from this function is the evidence on the basis
	of which the parameter space is sampled.
	"""
	# Obtain parameters from setup
	flux = config['flux']
	cov = config['cov']
	wavelength = config['wavelength']
	ssp_sed = config['ssp_sed']
	# ssp_wl = config['ssp_wl']
	velscale = config['velscale']
	oversampling = config['oversampling']
	extra_pixels = config['extra_pixels']
	extinction_law = config['extinction_law']
	norm_extinction = config['norm_extinction']

	# Load sampled parameters
	sigma = block["parameters", "sigma"]
	los_vel = block["parameters", "los_vel"]
	lumFrs = np.array([block["parameters", f"ssp{tIdx}"] for tIdx in range(1, ssp_sed.shape[0])],
				   dtype=float)
	lumFrs = 10**lumFrs
	sumLumFrs = lumFrs.sum()
	lumFrs = np.insert(lumFrs, lumFrs.size,
					   np.clip(1 - sumLumFrs, a_min=0, a_max=None))

	lumFrs = np.array(lumFrs, dtype=float)
	flux_model = np.sum(ssp_sed * lumFrs[:, np.newaxis], axis=0)
	# Kinematics
	sigma_pixel = sigma / (velscale / oversampling)
	veloffset_pixel = los_vel / (velscale / oversampling)
	x = np.arange(- 8*sigma_pixel, 8*sigma_pixel) - veloffset_pixel
	losvd_kernel = specBasics.losvd(x, sigma_pixel=sigma_pixel)
	flux_model = convolve(
		flux_model, losvd_kernel, mode='same', method='fft')
	# Rebin model spectra to observed grid
	flux_model = flux_model[
		extra_pixels * oversampling:-(extra_pixels * oversampling +1)
			].reshape((flux.size, oversampling)).mean(axis=1)
	### Mask pixels at the edges with artifacts produced by the convolution
	mask = np.ones_like(flux, dtype=bool)
	mask[:int(5 * sigma_pixel)] = False
	mask[-int(5 * sigma_pixel):] = False

	# Dust extinction
	if extinction_law is not None:
		flux_model = flux_model * 10**(-0.4 * extinction_law(
			wavelength, block["parameters", "av"], 3.1)) / norm_extinction

	# Calculate likelihood-value of the fit
	like = X2min(flux[mask], flux_model[mask], cov[mask])

	# Final posterior for sampling
	block[section_names.likelihoods, "HBSPS_SFH_like"] = like

	return 0


def cleanup(config):		
	return 0
