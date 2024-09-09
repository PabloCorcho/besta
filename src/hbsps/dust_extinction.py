import numpy as np
import extinction

def redden_ssp_model(config, av=0):
    extinction_law = config["extinction_law"]
    if extinction_law is not None:
        config["ssp_model"].L_lambda = extinction_law(
            config["ssp_model"].wavelength, config["ssp_model"].L_lambda, av,
            normalize=False)


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
        if wave_norm_range is None:
            self.norm_wave = None
        else:
            self.norm_wave = np.atleast_1d(np.mean(self.wave_norm_range))
        self.r = r
        self.extinction_law = getattr(extinction, ext_law_name)

    def extinction(self, wave, av):
        return 10**(-0.4 * self.extinction_law(wave, av, self.r))

    def __call__(self, wave, spectra, av, deredden=False, normalize=True):
        ext = self.extinction(wave, av)
        if normalize:
            ext /= self.extinction(self.norm_wave, av)

        if spectra.ndim > 1:
            extra_dims = tuple(np.arange(0, spectra.ndim - 1, dtype=int))
            ext = np.expand_dims(ext, axis=extra_dims)

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
