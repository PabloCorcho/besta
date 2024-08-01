import os
import sys
import numpy as np
from scipy.signal import convolve

from cosmosis.datablock import option_section, names as section_names
import hbsps.specBasics as specBasics

from hbsps import prepare_fit, kinematics, dust_extinction
from astropy import units as u
from pst.dust import DustScreen

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
	prepare_fit.prepare_observed_photometry(options, config)
	# ---- Read redshift information ----------------------------------------- #
	config['redshift'] = options[option_section, "redshift"]
	print(f"Input source redshift: {config['redshift']}")
	# ------------------------------------------------------------------------ #
	prepare_fit.prepare_ssp_model(options, config, normalize=False)
	# ------------------------------------------------------------------------ #
	# prepare_fit.prepare_extinction_law(options, config)
	# ------------------------------------------------------------------------ #
	prepare_fit.prepare_sfh_model(options, config)


	if options.has_value(option_section, "los_vel"):
		if options.has_value(option_section, "los_sigma"):
			print(f"Convolving SSP models with Gaussian LOSF")
			ssp, mask = kinematics.convolve_ssp_model(
				config,
				options[option_section, "los_sigma"],
				options[option_section, "los_vel"])
			config['ssp_model'] = ssp
			config['mask'] = mask
			print("Valid pixels: ", np.count_nonzero(mask), mask.size)
	else:
		print("No kinematic information was provided")

	# if options.has_value(option_section, "ExtinctionLaw"):
	# 	av = options[option_section, "av"]
	# 	print(f"Reddening SSP models using Av={av}")
	# 	dust_extinction.redden_ssp_model(config, av)
	print("Producing photometry grid")
	dust_model = DustScreen("ccm89")
	a_v_array = np.linspace(0.1, 3, 30)
	ssps = [dust_model.redden_ssp_model(config['ssp_model'], a_v=av) for av in a_v_array]
	all_photometry = np.zeros(
		(a_v_array.size, len(config['filters']),
                             *config['ssp_model'].L_lambda.shape[:-1])
							 ) * u.Quantity("3631e-9 Jy / Msun")

	for j, ssp in enumerate(ssps):
			photo = ssp.compute_photometry(
				filter_list=config['filters'], z_obs=config['redshift']
				).to("3631e-9 Jy / Msun")
			all_photometry[j] = photo
	config['av_grid'] = a_v_array
	config['photometry_grid'] = all_photometry
	return config
	
def execute(block, config):
	"""Function executed by sampler
	This is the function that is executed many times by the sampler. The
	likelihood resulting from this function is the evidence on the basis
	of which the parameter space is sampled.
	"""
	# Obtain parameters from setup
	sfh_model = config['sfh_model']
	ssp = config['ssp_model']

	# sfh_free_params = {k:block["parameters", k] for k in sfh_model.free_params.keys()}
	# valid = sfh_model.parse_free_params(sfh_free_params)
	valid = sfh_model.parse_datablock(block)
	if not valid:
		print("Invalid")
		block[section_names.likelihoods, "SFH_photometry_like"] = -1e20
		return 0
	
	av = 0.5
	av_idx = np.searchsorted(config['av_grid'], av)
	w_idx = (av - config['av_grid'][av_idx - 1]) / (
		config['av_grid'][av_idx] - config['av_grid'][av_idx - 1])
	photometry = config['photometry_grid'][av_idx] * w_idx + config['photometry_grid'][av_idx - 1] * (1 - w_idx)

	flux_model = sfh_model.model.compute_photometry(
		ssp,
		t_obs=sfh_model.today,
		allow_negative=False,
		photometry=photometry)
	flux_model = flux_model.to_value("3631e-9 Jy")

	normalization = np.mean(config['photometry_flux'] / flux_model)
	print(normalization)
	block['parameters', 'normalization'] = normalization
	flux_model *= normalization
	print(config['photometry_flux'], flux_model)
	like = X2min(
		config['photometry_flux'], flux_model, config['photometry_flux_var'])
	# Final posterior for sampling
	block[section_names.likelihoods, "SFH_photometry_like"] = like
	return 0


def cleanup(config):
	return 0
