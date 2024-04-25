import numpy as np


def composite_stellar_population(config, weights):
    if type(weights) is dict:
        w_array = np.array([weights[w] for w in weights.keys() if "ssp" in w])
    else:
        w_array = weights
    composite_spectra = np.sum(config["ssp_sed"] * w_array[:, np.newaxis], axis=0)
    return composite_spectra
