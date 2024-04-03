import numpy as np
import matplotlib.pyplot as plt

from pst.SSP import BaseGM
import specBasics


ssp = BaseGM()

t = np.linspace(0, 13.7, 51)
tau = 1
mass_history = 1 - (t + tau) / tau * np.exp(-t/tau)


# Pressent mass = 1e10 Msun
mass_history *= 1e10 / mass_history[-1]
t_bins = (t[:-1] + t[1:]) / 2
mass_formed = mass_history[1:] - mass_history[:-1]
metallicity = np.full_like(t, fill_value=0.02)

plt.figure()
plt.plot(t_bins, mass_formed)
plt.plot(t, mass_history)
plt.show()


# print(ssp.wavelength)


# Degrade the resolution
velscale = 100
delta_lnwl = velscale / specBasics.constants.c.to('km/s').value
sigma = 150
sigma_pix = sigma / velscale

vel_offset = 200
vel_offset_pixels = vel_offset / velscale
redshift = np.exp(vel_offset /specBasics.constants.c.to('km/s').value) - 1 

newBorders = np.arange(
		np.log(ssp.wavelength[0]),
        np.log(ssp.wavelength[-1]),
        delta_lnwl)

ssp.interpolate_sed(np.exp(newBorders))

sed, weights = ssp.compute_SED(t * 1e9, mass_history, metallicity)

ori_sed = sed.copy()
ori_wl = ssp.wavelength.copy()

plt.figure()
plt.plot(ssp.wavelength, sed)
plt.show()

ref_spectra = sed

ppxf_sed, lnwl, _ = specBasics.log_rebin(ori_wl, ori_sed, velscale=velscale)

ref_spectra_sm = specBasics.smoothSpectrumFast(
    ref_spectra, sigma_pix)

print(np.diff(np.log(ssp.wavelength)), ssp.wavelength.size)
print(np.diff(lnwl), lnwl.size)

plt.figure()
plt.plot(ori_wl, ori_sed, '-', label='Original')
plt.plot(ssp.wavelength, ref_spectra, '-', label='Reinterpolated')
plt.plot(ssp.wavelength, ref_spectra_sm, '-', label='Smoothed')
plt.plot(ssp.wavelength * (1 + redshift), ref_spectra_sm, '-', label='Redshifted')
plt.legend()
plt.show()

print("saving spectra at: BaseGM_mock_spectra.dat")
np.savetxt("BaseGM_mock_spectra_delayed_tau.dat",
           np.array(
               [ssp.wavelength * (1 + redshift),
                ref_spectra_sm,
                ref_spectra_sm * 0.001]).T)

np.savetxt("BaseGM_mock_spectra_delayed_tau_real_weights.dat",
           weights)

