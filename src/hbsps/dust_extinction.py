import numpy as np


def redden_ssp(config, av=0, r=3.1):
    extinction_law = config["extinction_law"]

    if extinction_law is not None:
        flux = config["ssp_sed"]
        norm_extinction = config["norm_extinction"]
        ext = 10 ** (-0.4 * extinction_law(config["ssp_wl"], av, r)) / norm_extinction
        # sed = sed * ext[np.newaxis, :]
        config["ssp_sed"] = config["ssp_sed"] * ext[np.newaxis, :]


def redden_spectra(config, av=0, r=3.1):
    extinction_law = config["extinction_law"]

    if extinction_law is not None:
        flux = config["flux"]
        wavelength = config["wavelength"]
        norm_extinction = config["norm_extinction"]
        ext = 10 ** (-0.4 * extinction_law(wavelength, av, r)) / norm_extinction
        # sed = sed * ext[np.newaxis, :]
        red_flux = flux * ext
    else:
        red_flux = flux.copy()
    return red_flux


def deredden_spectra(config, av=0, r=3.1):
    extinction_law = config["extinction_law"]
    if extinction_law is not None:
        flux = config["flux"]
        wavelength = config["wavelength"]
        norm_extinction = config["norm_extinction"]
        ext = 10 ** (-0.4 * extinction_law(wavelength, av, r)) / norm_extinction
        # sed = sed * ext[np.newaxis, :]
        dered_flux = flux / ext
    else:
        dered_flux = flux.copy()
    return dered_flux
