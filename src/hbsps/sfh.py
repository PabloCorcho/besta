import numpy as np


def composite_stellar_population(config, weights):
    if type(weights) is dict:
        w_array = np.array([weights[w] for w in weights.keys() if "ssp" in w])
    else:
        w_array = weights
    w_array = 10**w_array
    composite_spectra = np.sum(config["ssp_sed"] * w_array[:, np.newaxis], axis=0)
    return composite_spectra

def reconstruct_sfh(config, weights):
    if type(weights) is dict:
        w_array = np.array([weights[w] for w in weights.keys() if "ssp" in w])
    else:
        w_array = np.asarray(weights)
    w_array = 10**w_array

    ssp_met, ssp_age = config['ssp_metals'], config['ssp_ages']
    ssp_mlr = config['ssp_mlr']
    norm_flux = config['norm_flux']
    w_array = w_array.reshape(ssp_mlr.shape)

    ssp_mass_formed = norm_flux * w_array * ssp_mlr
    mass_history = np.nansum(ssp_mass_formed, axis=0)
    total_mass = np.nansum(mass_history)
    mean_age = np.sum(ssp_mass_formed * ssp_age) / total_mass
    mean_metals = np.sum(ssp_mass_formed * ssp_met) / total_mass
    results = {'ages': ssp_age, 'metals': ssp_met, 'ssp_mass_formed': ssp_mass_formed,
               'mass_formed_history': mass_history, 'total_mass': total_mass,
               'mean_age': mean_age, 'mean_metals': mean_metals}
    return results