import os
import sys
import numpy as np
from scipy.signal import convolve

from cosmosis.datablock import option_section, names as section_names
import hbsps.specBasics as specBasics

from hbsps import prepare_fit, kinematics, dust_extinction


def X2min(spectrum, recSp, cov):
	# Determine residual, divide first residual vector by 
	# diagonal elements covariance matrix.
	residual1 = recSp - spectrum
	residual2 = np.copy(residual1)
	residual1 /= cov
		
	# Determine likelihood term (i.e. X2-value)
	chiSq = -0.5 * np.dot(residual1, residual2)
	
	return chiSq

def setup(options):
	"""Set-up the COSMOSIS sampler.
	
	Args:
		options: options from startup file (i.e. .ini file)
	Returns:
		config: parameters or objects that are passed to 
			the sampler.
			
	"""
	# Pipeline values file
	config = {}
	# ------------------------------------------------------------------------ #
	prepare_fit.prepare_observed_spectra(options, config)
	# ------------------------------------------------------------------------ #
	prepare_fit.prepare_ssp_model(options, config)
	# ------------------------------------------------------------------------ #
	prepare_fit.prepare_extinction_law(options, config)

	if options.has_value(option_section, "los_vel"):
		if options.has_value(option_section, "los_sigma"):
			print(f"Convolving SSP models with Gaussian LOSF")
			sed, mask = kinematics.convolve_ssp(config,
				options[option_section, "los_sigma"],
				options[option_section, "los_vel"])
			config['ssp_sed'] = sed
			config['ssp_wl'] = config['wavelength'].copy()
			config['mask'] = mask
			print("Valid pixels: ", np.count_nonzero(mask), mask.size)
	else:
		print("No kinematic information was provided")

	if options.has_value(option_section, "ExtinctionLaw"):
		av = options[option_section, "av"]
		print(f"Reddening SSP models using Av={av}")
		dust_extinction.redden_ssp(config, av)
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
	ssp_sed = config['ssp_sed']
	mask = config['mask']

	lumFrs = np.array([block["parameters", f"ssp{tIdx}"] for tIdx in range(
		1, ssp_sed.shape[0])], dtype=float)
	lumFrs = 10**lumFrs
	sumLumFrs = lumFrs.sum()
	lumFrs = np.insert(lumFrs, lumFrs.size,
					   np.clip(1 - sumLumFrs, a_min=0, a_max=None))
	flux_model = np.sum(ssp_sed * lumFrs[:, np.newaxis], axis=0)
	# Calculate likelihood-value of the fit
	like = X2min(flux[mask], flux_model[mask], cov[mask])

	# Final posterior for sampling
	block[section_names.likelihoods, "SFH_like"] = like

	return 0


def cleanup(config):		
	return 0
