import numpy as np
import os

from scipy.signal import fftconvolve
from scipy.optimize import nnls

import cosmosis
from cosmosis.datablock import option_section, names as section_names
import hbsps.specBasics as specBasics

from hbsps import prepare_spectra
from hbsps import kinematics
from hbsps.dust_extinction import deredden_spectra

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
	values_file = options[option_section, "values"]
	config = {}
	# ------------------------------------------------------------------------ #
	prepare_spectra.prepare_observed_spectra(options, config)
	# ------------------------------------------------------------------------ #
	prepare_spectra.prepare_ssp_data(options, config)
	# ------------------------------------------------------------------------ #
	prepare_spectra.prepare_extinction_law(options, config)
	# ------------------------------------------------------------------------ #
#	make_values_file(values_file=values_file, inc_extinction=inc_extinction)
	if options.has_value(option_section, "SSPSave"):
		config['ssp_log'] = open(
			os.path.join(
				os.path.dirname(options['output', 'filename']),
				"SSP_weights.dat"),
			"w")
	return config

def execute(block, config):
	"""Function executed by sampler
	This is the function that is executed many times by the sampler. The
	likelihood resulting from this function is the evidence on the basis
	of which the parameter space is sampled.
	"""
	# Obtain parameters from setup
	cov = config['cov']

	# Load sampled parameters
	sigma = block["parameters", "sigma"]
	los_vel = block["parameters", "los_vel"]

	# Kinematics
	sed, mask = kinematics.convolve_ssp(config, sigma, los_vel)

	# Dust extinction
	dered_flux = deredden_spectra(config, block["parameters", "av"], r=3.1)
	
	# Solve the linear system
	solution, rnorm = nnls(sed.T, dered_flux, maxiter=sed.shape[0] * 10)
	if 'ssp_log' in config:
		config['ssp_log'].write(", ".join(np.array(solution, dtype=str)) + "\n")

	flux_model = np.sum(sed * solution[:, np.newaxis], axis=0)
	# Calculate likelihood-value of the fit
	like = X2min(dered_flux[mask], flux_model[mask], cov[mask])

	# Final posterior for sampling
	block[section_names.likelihoods, "KinDust_like"] = like

	return 0

def cleanup(config):
	if "ssp_log" in config:
		print("Saving SSP log")
		config['ssp_log'].close()
	return 0

module = cosmosis.FunctionModule("KinDust", setup, execute)