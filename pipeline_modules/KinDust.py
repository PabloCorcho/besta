"""
Module for fitting stellar kinematics using spectroscopic data.
"""
import numpy as np
from scipy.optimize import nnls

import cosmosis
from cosmosis.datablock import option_section, names as section_names
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
	config = {"solution":[],
		      "solution_output": options["output", "filename"] + "_ssp_sol"}
	# ------------------------------------------------------------------------ #
	prepare_fit.prepare_observed_spectra(options, config)
	# ------------------------------------------------------------------------ #
	prepare_fit.prepare_ssp_model(options, config)
	# ------------------------------------------------------------------------ #
	prepare_fit.prepare_extinction_law(options, config)
	# ------------------------------------------------------------------------ #
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

	av = block["parameters", "av"]
	# Kinematics
	sed, mask = kinematics.convolve_ssp(config,
									 los_vel=block["parameters", "los_vel"],
									 los_sigma=block["parameters", "los_sigma"],
									 los_h3=block["parameters", "los_h3"],
									 los_h4=block["parameters", "los_h4"])

	# Dust extinction
	dered_flux = deredden_spectra(config, av)
	
	# Solve the linear system
	solution, _ = nnls(sed.T, dered_flux, maxiter=sed.shape[0] * 10)
	#TODO: This should be stored as
	# block["parameters", "ssp1"] = solution[0]
	# block["parameters", "ssp2"] = solution[1]
	config["solution"].append(solution)

	flux_model = np.sum(sed * solution[:, np.newaxis], axis=0)
	# Calculate likelihood-value of the fit
	like = X2min(dered_flux[mask], flux_model[mask], cov[mask])

	# Final posterior for sampling
	block[section_names.likelihoods, "KinDust_like"] = like
	return 0

def cleanup(config):
	if "solution" in config:
		np.savetxt(config["solution_output"], config["solution"])
	return 0
	

module = cosmosis.FunctionModule("KinDust", setup, execute)