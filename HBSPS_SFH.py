import os
import sys
import numpy as np
import cosmosis
from cosmosis.datablock import option_section, names as section_names
import specBasics
import SPSbasics

from pst import SSP
from pst.utils import flux_conserving_interpolation

def X2min(spectrum, recSp, cov):
	# Determine residual, divide first residual vector by 
	# diagonal elements covariance matrix.
	residual1 = recSp - spectrum
	residual2 = np.copy(residual1)
	residual1 /= cov
		
	# Determine likelihood term (i.e. X2-value)
	chiSq = -0.5*np.dot(residual1, residual2)
	
	return chiSq

	
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
	ssp_name = options[option_section, "SSPModel"]
	ssp_dir = options[option_section, "SSPDir"]
	age_range = options[option_section, "ageRange"]
	met_range = options[option_section, "metRange"]
	wl_range = options[option_section, "wlRange"]

	wl_norm_range = options[option_section, "wlNormRange"]
	redshift = options[option_section, "redshift"]
	velscale = options[option_section, "velscale"]
	oversampling = options[option_section, "oversampling"]
	# bcov = 10**(options[option_section, "logbcov"])
	polOrder = options[option_section, "polOrder"]

    # INITIALISE OBSERVATIONAL DATA
	# Read wavelength, spectrum and create covariance matrix
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
	nBins = len(wavelength)
	print("Spectra normalized using wavelength range: ", wl_norm_range)
	normIdx = np.where((wavelength >= wl_norm_range[0]) & (wavelength <= wl_norm_range[1]))[0]
	norm_flux = np.mean(flux[normIdx])
	flux = flux[goodIdx] / norm_flux
	error = error[goodIdx] / norm_flux
	cov = error * error
	print("Log-binning spectra to velocity scale: ", velscale)
	flux, ln_wave, _ = specBasics.log_rebin(wavelength, flux, velscale=velscale)
	cov, _, _ = specBasics.log_rebin(wavelength, cov, velscale=velscale)
	wavelength = np.exp(ln_wave)

	# Add additional global covariance parameterized by bcov
	medCov = np.median(cov)
	# cov = cov + bcov*medCov

    # INITIALISE SSP MODEL
	n_ages = len(age_range) - 1
	n_met = len(met_range) - 1
	n_models = n_ages * n_met
	sfh_age_bins = (age_range[:-1] + age_range[1:]) / 2
	sfh_met_bins = (met_range[:-1] + met_range[1:]) / 2
	ssp = getattr(SSP, ssp_name)
	if ssp_dir is 'None':
		ssp_dir = None
	ssp = ssp(path=ssp_dir)
	ssp_sed = np.zeros((n_met, n_ages, ssp.wavelength.size))
    # Create a grid
	spp_age_bins = ssp.log_ages_yr
	lim = 1.5 * spp_age_bins[[0, -1]] - 0.5 * spp_age_bins[[1, -2]]
	ssp_age_bins = np.hstack(
            [lim[0], (spp_age_bins[1:] + spp_age_bins[:-1])/2, lim[1]])

	spp_met_bins = ssp.metallicities
	lim = 1.5 * spp_met_bins[[0, -1]] - 0.5 * spp_met_bins[[1, -2]]
	ssp_met_bins = np.hstack(
            [lim[0], (spp_met_bins[1:] + spp_met_bins[:-1])/2, lim[1]])
	
	ssp_age_idx = np.searchsorted(ssp_age_bins, age_range)
	age_bins = [slice(ssp_age_idx[i], ssp_age_idx[i+1]) for i in range(n_ages)]
	ssp_met_idx = np.searchsorted(ssp_met_bins, met_range)
	met_bins = [slice(ssp_met_idx[i], ssp_met_idx[i+1]) for i in range(n_met)]

	ssp_prefix = []
	for j, m_bin in enumerate(met_bins):
		sed = np.mean(ssp.L_lambda[m_bin], axis=0)
		for i, a_bin in enumerate(age_bins):
			ssp_sed[j, i, :] = np.mean(sed[a_bin], axis=0)
			ssp_prefix.append(f"SSP_logage_{sfh_age_bins[i]:05.2f}_z_{sfh_met_bins[j]:06.4f}")

	ssp.L_lambda = ssp_sed
	
	# Rebin the spectra
	print("Log-binning SSP spectra to velocity scale: ", velscale / oversampling, " km/s")
	dlnlam = velscale / specBasics.constants.c.to('km/s').value
	extra_offset_pixel = 300 / velscale
	dlnlam /= oversampling
	lnlam_bin_edges = np.arange(
		ln_wave[0] - dlnlam * extra_offset_pixel * oversampling - 0.5 * dlnlam,
        ln_wave[-1] + dlnlam * (1 + extra_offset_pixel) * oversampling + 0.5 * dlnlam,
        dlnlam)
	ssp.interpolate_sed(np.exp(lnlam_bin_edges))
	print(ln_wave.size, lnlam_bin_edges.size, ssp.wavelength.size)
	print(ln_wave[:2], np.log(ssp.wavelength[int(extra_offset_pixel * oversampling):int((extra_offset_pixel +1)* oversampling) +1]))
	
    # Reshape the SSPs SED to (n_models, wavelength)
	ssp_sed = ssp.L_lambda.reshape(
		(ssp.L_lambda.shape[0] * ssp.L_lambda.shape[1], ssp.L_lambda.shape[2]))
	ssp_wl = ssp.wavelength
	print("SSP SED shape: ", ssp_sed.shape)
	
	ssp_output_file = os.path.join(os.path.dirname(fileName), "ssp_model_spectra.dat")
	print("Saving model spectra at: ", ssp_output_file)
	np.savetxt(ssp_output_file, np.vstack((ssp_wl, ssp_sed)).T,
			header=f"CSP logarithmically sampled (velscale={velscale / oversampling})" + "\nWavelength, " + ", ".join(ssp_prefix)
			)

	# Basis of Legendre polynomials for multiplicative polynomial
	AL = specBasics.getLegendrePolynomial(wavelength, polOrder, bounds=None)

	# Pass parameters to execute function.
	config = {}
	config['flux'] = flux
	config['wavelength'] = wavelength
	config['velscale'] = velscale
	config['oversampling'] = oversampling
	#config['nBins'] = nBins
	#config['goodIdx'] = goodIdx
	config['error'] = error
	config['cov'] = cov
	config['polOrder'] = polOrder
	config['AL'] = AL
	# config['bcov'] = bcov
	config['ssp_sed'] = ssp_sed
	config['ssp_wl'] = ssp_wl

	return config
	
def execute(block, config):
	"""Function executed by sampler
	This is the function that is executed many times by the sampler. The
	likelihood resulting from this function is the evidence on the basis
	of which the parameter space is sampled.
	"""
	
	# Obtain parameters from setup
	flux = config['flux']
	wavelength = config['wavelength']
	ssp_sed = config['ssp_sed']
	ssp_wl = config['ssp_wl']
	velscale = config['velscale']
	oversampling = config['oversampling']
	#nBins = config['nBins']
	#error = config['error']
	cov = config['cov']
	polOrder = config['polOrder']

	AL = config['AL']
	# Load sampled parameters
	sigma = block["parameters", "sigma"]
	los_vel = block["parameters", "los_vel"]
	lumFrs = np.array([block["parameters", f"ssp{tIdx}"] for tIdx in range(1, ssp_sed.shape[0] + 1)],
				   dtype=float)

	# The total sum of luminosity fractions should always be one. If
	# the sum is larger than one, then penalize model with additional
	# (strict) prior to prevent degeneracies with multiplicative
	# polynomial. Set the fraction of the last remaining SSP to one
	# minus the other luminosity fractions (or zero if this is already
	# higher than one). This saves one parameter in sampling procedure.
	#lumFrs = np.array(lumFrs, dtype=float)
	sumLumFrs = lumFrs.sum()
	# lumFrs /= sumLumFrs
	if sumLumFrs > 1:
		prLumFrs = (sumLumFrs - 1)**2 / (2 * 1e-6)
	else:
		prLumFrs = 0

	lumFrs = np.array(lumFrs, dtype=float)
	flux_model = np.sum(ssp_sed * lumFrs[:, np.newaxis], axis=0)
	# Broaden the spectra
	gauss_kernel_pixel_size = sigma / (velscale / oversampling)
	flux_model = specBasics.smoothSpectrumFast(flux_model, gauss_kernel_pixel_size)
	# Apply redshift offset
	redshift = np.exp(los_vel / specBasics.constants.c.to('km/s').value) -1
	flux_model = flux_conserving_interpolation(
		wavelength, ssp_wl * (1 + redshift), flux_model)
	# Apply polynomial correction to reconstructed spectrum
	residual = flux / flux_model
	norm = np.nanmedian(residual)
	flux_model *= norm

	# ALT = np.copy(AL)
	# ALT = np.transpose(ALT)
	# for i in range(nBins):
	# 	ALT[:,i] = ALT[:,i] / cov[i]
		
	# cL = np.linalg.solve(np.dot(ALT,AL), np.dot(ALT, residual))
	# polL = np.dot(AL,cL)
	# recSpecCSP *= polL
	
	# Calculate likelihood-value of the fit
	like = X2min(flux, flux_model, cov)
	
	# Final posterior for sampling: combination likelihood and prior lumFrs
	block[section_names.likelihoods, "HBSPS_SFH_like"] = like - prLumFrs

	return 0
	
def cleanup(config):
		
	return 0
