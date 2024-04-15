import numpy as np
import matplotlib.pyplot as plt

from pst.SSP import BaseGM
import specBasics


ssp = BaseGM()

velscale = 70 # km/s/pix
sigma = 150  # km/s
vel_offset = 200  # km/s
snr = 50
total_mass = 1e10
tau = 3
metallicity = 0.010
t = np.linspace(0, 13.7, 51)
mass_history = 1 - (t + tau) / tau * np.exp(-t/tau)

# Pressent mass = 1e10 Msun
mass_history *= total_mass / mass_history[-1]
t_bins = (t[:-1] + t[1:]) / 2
mass_formed = mass_history[1:] - mass_history[:-1]
metallicity_history = np.full_like(t, fill_value=metallicity)

plt.figure()
plt.bar(t_bins, mass_formed, width=t[1:] - t[:-1], label="Mass formed")
plt.bar(t_bins, mass_formed / (t[1:] - t[:-1]) / 1e9, width=t[1:] - t[:-1], label="SFR")
#plt.plot(t, mass_history, label="Mass history")
plt.yscale("log")
plt.legend()
plt.show()

# Degrade the resolution
delta_lnwl = velscale / specBasics.constants.c.to('km/s').value
sigma_pix = sigma / velscale
vel_offset_pixels = vel_offset / velscale
redshift = np.exp(vel_offset /specBasics.constants.c.to('km/s').value) - 1 

newBorders = np.arange(
		np.log(ssp.wavelength[0]),
        np.log(ssp.wavelength[-1]),
        delta_lnwl)

ssp.interpolate_sed(np.exp(newBorders))
sed, weights = ssp.compute_SED(t * 1e9, mass_history, metallicity_history)

ori_sed = sed.copy()
ori_wl = ssp.wavelength.copy()

ref_spectra = sed

ppxf_sed, lnwl, _ = specBasics.log_rebin(ori_wl, ori_sed, velscale=velscale)

ref_spectra_sm = specBasics.smoothSpectrumFast(
    ref_spectra, sigma_pix)

ref_spectra_sm_err = ref_spectra_sm / snr

print(np.diff(np.log(ssp.wavelength)), ssp.wavelength.size)
print(np.diff(lnwl), lnwl.size)

plt.figure()
plt.plot(ori_wl, ori_sed, '-', label='Original')
plt.plot(ssp.wavelength, ref_spectra, '-', label='Reinterpolated')
plt.plot(ssp.wavelength, ref_spectra_sm, label='Smoothed')
plt.errorbar(ssp.wavelength * (1 + redshift), ref_spectra_sm,
             yerr=ref_spectra_sm_err, errorevery=5, capsize=3, label='Redshifted')
plt.legend()
plt.show()

print("saving spectra at: BaseGM_mock_spectra_delayed_tau.dat")
np.savetxt("BaseGM_mock_spectra_delayed_tau.dat",
           np.array(
               [ssp.wavelength * (1 + redshift),
                ref_spectra_sm, ref_spectra_sm_err]).T)

np.savetxt("BaseGM_mock_spectra_delayed_tau_real_weights.dat",
           weights)

