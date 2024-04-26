import numpy as np
import extinction

def redden_ssp(config, av=0):
    extinction_law = config["extinction_law"]
    if extinction_law is not None:
        config["ssp_sed"] = extinction_law(
            config["ssp_wl"], config["ssp_sed"], av)


def redden_spectra(config, av=0):
    extinction_law = config["extinction_law"]

    if extinction_law is not None:
        flux = config["flux"]
        wavelength = config["wavelength"]
        red_flux = extinction_law(wavelength, flux, av)
    else:
        red_flux = flux.copy()
    return red_flux


def deredden_spectra(config, av=0):
    extinction_law = config["extinction_law"]
    if extinction_law is not None:
        flux = config["flux"]
        wavelength = config["wavelength"]
        dered_flux = extinction_law(wavelength, flux, av, deredden=True)
    else:
        dered_flux = flux.copy()
    return dered_flux

class DustScreen(object):
    
    def __init__(self, ext_law_name, wave_norm_range=None, r=3.1):
        self.wave_norm_range = wave_norm_range
        self.norm_wave = np.atleast_1d(np.mean(self.wave_norm_range))
        self.r = r
        self.extinction_law = getattr(extinction, ext_law_name)

    def __call__(self, wave, spectra, av, deredden=False):
        ext = 10**(-0.4 * self.extinction_law(wave, av, self.r))
        ext /= 10**(-0.4 * self.extinction_law(self.norm_wave, av, self.r))

        if spectra.ndim > 1:
            ext = ext[np.newaxis, :]

        if deredden:
            return spectra / ext
        else:
            return spectra * ext

if __name__ == "__main__":
    dust_model = DustScreen("ccm89", wave_norm_range=[5000, 5500])
    print()
    from matplotlib import pyplot as plt
    plt.figure()
    wave = np.linspace(4000, 9000)
    plt.plot(wave, dust_model(wave, np.ones(50), av=2))
    plt.show()
