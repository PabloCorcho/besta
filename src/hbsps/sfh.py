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
    print("Reconstructing SFH from input solution")
    if type(weights) is dict:
        w_array = np.array([weights[w] for w in weights.keys() if "ssp" in w])
    else:
        w_array = np.asarray(weights)
    w_array = 10**w_array

    ssp_met, ssp_age = config['ssp_metals_edges'], config['ssp_ages_edges']
    # Geometric mean
    ssp_met_bins = (ssp_met[:-1] + ssp_met[1:]) / 2
    ssp_age_bins = (ssp_age[:-1] + ssp_age[1:]) / 2
    ssp_mlr = config['ssp_mlr']
    norm_flux = config['norm_flux']
    w_array = w_array.reshape(ssp_mlr.shape)
    print("Light fraction matrix shape: ", w_array.shape)
    ssp_mass_formed = norm_flux * w_array * ssp_mlr
    mass_history = np.nansum(ssp_mass_formed, axis=0)
    total_mass = np.nansum(mass_history)
    mean_age = 10**(np.sum(ssp_mass_formed * ssp_age_bins[np.newaxis, :]
                      ) / total_mass)
    mean_metals = 10**(np.sum(ssp_mass_formed * ssp_met_bins[:, np.newaxis]
                         ) / total_mass)
    results = {'ages': ssp_age, 'metals': ssp_met,
               'ssp_mass_formed': ssp_mass_formed,
               'mass_formed_history': mass_history, 'total_mass': total_mass,
               'mean_age': mean_age, 'mean_metals': mean_metals}
    return results

def reconstruct_sfh_from_table(config, table):
    print("Reconstructing SFH from input solution")
    weights = 10**np.array([table[k].value for k in table.keys() if "ssp" in k])

    ssp_met, ssp_age = config['ssp_metals_edges'], config['ssp_ages_edges']
    # Geometric mean
    ssp_met_bins = (ssp_met[:-1] + ssp_met[1:]) / 2
    ssp_age_bins = (ssp_age[:-1] + ssp_age[1:]) / 2
    ssp_mlr = config['ssp_mlr']
    norm_flux = config['norm_flux']
    weights = weights.reshape((*ssp_mlr.shape, weights.shape[-1]))

    print("Light fraction matrix shape: ", weights.shape)
    ssp_mass_formed = norm_flux * weights * ssp_mlr[:, :, np.newaxis]
    mass_history = np.nansum(ssp_mass_formed, axis=0)
    total_mass = np.nansum(mass_history, axis=0)
    # mean_age = 10**(np.sum(ssp_mass_formed * ssp_age_bins[np.newaxis, :]
    #                   ) / total_mass)
    # mean_metals = 10**(np.sum(ssp_mass_formed * ssp_met_bins[:, np.newaxis]
    #                      ) / total_mass)
    results = {'ages': ssp_age, 'metals': ssp_met,
               'ssp_mass_formed': ssp_mass_formed,
               'mass_formed_history': mass_history, 'total_mass': total_mass,
                #'mean_age': mean_age, 'mean_metals': mean_metals
               }
    return results