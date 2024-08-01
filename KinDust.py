import numpy as np
import os

from scipy.signal import fftconvolve
from scipy.optimize import nnls

import cosmosis
from cosmosis.datablock import option_section, names as section_names
import hbsps.specBasics as specBasics

from hbsps import prepare_fit
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
	config = {}
	# ------------------------------------------------------------------------ #
	prepare_fit.prepare_observed_spectra(options, config)
	# ------------------------------------------------------------------------ #
	prepare_fit.prepare_ssp_model(options, config)
	# ------------------------------------------------------------------------ #
	prepare_fit.prepare_extinction_law(options, config)
	# ------------------------------------------------------------------------ #
	# if options.has_value(option_section, "SSPSave"):
	# 	config['ssp_log'] = open(
	# 		os.path.join(
	# 			os.path.dirname(options['output', 'filename']),
	# 			"SSP_weights.dat"),
	# 		"w")
	# 	header = '#' + ", ".join(
	# 		[f"parameters--ssp{i + 1}" for i in range(config['ssp_sed'].shape[0])])
	# 	header += ", parameters--los_vel, parameters--los_sigma, parameters--av, like\n"
	# 	config['ssp_log'].write(header)
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
	sigma = block["parameters", "los_sigma"]
	los_vel = block["parameters", "los_vel"]
	av = block["parameters", "av"]
	# Kinematics
	sed, mask = kinematics.convolve_ssp(config, sigma, los_vel)

	# Dust extinction
	dered_flux = deredden_spectra(config, av)
	
	# Solve the linear system
	solution, rnorm = nnls(sed.T, dered_flux, maxiter=sed.shape[0] * 10)

	flux_model = np.sum(sed * solution[:, np.newaxis], axis=0)
	# Calculate likelihood-value of the fit
	like = X2min(dered_flux[mask], flux_model[mask], cov[mask])
	#TODO: This should be stored as
	# block["parameters", "ssp1"] = solution[0]
	# block["parameters", "ssp2"] = solution[1]

	# Final posterior for sampling
	block[section_names.likelihoods, "KinDust_like"] = like

	# if 'ssp_log' in config:
	# 	config['ssp_log'].write(
	# 		", ".join(np.array(np.log10(solution + 1e-10), dtype=str))
	# 		+ f", {los_vel}, {sigma}, {av}"
	# 		+ f", {like}"
	# 		+ "\n"
	# 		)

	return 0

def cleanup(config):
	# if "ssp_log" in config:
	# 	print("Saving SSP log")
	# 	config['ssp_log'].close()
	return 0

module = cosmosis.FunctionModule("KinDust", setup, execute)