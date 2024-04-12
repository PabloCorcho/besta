import numpy as np
import specBasics
import matplotlib.pyplot as plt

from pst.utils import flux_conserving_interpolation


obs_wl, obs_f, obs_e = np.loadtxt("BaseGM_mock_spectra_simple_ssp.dat", unpack=True)

ssp_data = np.loadtxt("BaseGM_mock_spectra_simple_ssp.dat_ssp_model_spectra.dat",
                           unpack=True)
print(ssp_data.shape)
ssp_wl, ssp_sed = ssp_data[0], ssp_data[1:]

# Load weights
ssp_weights = []
los_vel = None
sigma = None

with open("output/simple_ssp/means.txt", "r") as f:
    f.readline()
    for line in f.readlines():
        print(line)
        parname, val, _ = line.split("  ")[:-1]
        val = float(val)
        if 'ssp' in parname:
            ssp_weights.append(val)
        elif 'los_vel' in parname:
            los_vel = val
        elif 'sigma' in parname:
            sigma = val
        else:
            continue

redshift = np.exp(
    los_vel /specBasics.constants.c.to('km/s').value) - 1 
sigma_pix = sigma / (100 / 2)
ssp_weights = np.array(ssp_weights)

syn_spectra = np.sum(ssp_weights[:, np.newaxis] * ssp_sed, axis=0)

syn_spectra = specBasics.smoothSpectrumFast(
    syn_spectra, sigma_pix)

syn_spectra_smooth = syn_spectra.copy()
syn_spectra_smooth_wl = ssp_wl.copy()

syn_spectra = flux_conserving_interpolation(obs_wl, ssp_wl * (1 + redshift), syn_spectra)
ssp_wl = obs_wl

norm_obs = np.nanmean(obs_f[(obs_wl >= 4000) & (obs_wl <= 7000)])
norm_syn = np.nanmean(syn_spectra[(ssp_wl >= 4000) & (ssp_wl <= 7000)])

residuals = obs_f / norm_obs - syn_spectra / norm_syn


fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
ax = axs[0]
ax.errorbar(obs_wl, obs_f / norm_obs, yerr=obs_e/norm_obs, errorevery=5, label='obs')
ax.plot(syn_spectra_smooth_wl, syn_spectra_smooth / norm_syn, label='synth-sm')
ax.plot(ssp_wl, syn_spectra / norm_syn, label='synth-sm-vel')
ax.legend()
ax = axs[1]
ax.plot(obs_wl, residuals)
ax.set_ylim(np.nanpercentile(residuals[(obs_wl >= 4000) & (obs_wl <= 7000)], [5, 95]))
plt.show()
