import numpy as np
import specBasics
import matplotlib.pyplot as plt

from pst.utils import flux_conserving_interpolation
from pst.SSP import BaseGM

from utils import read_ini_file, read_chain_file, make_plot_chains, compute_chain_percentiles
from output import Reader

ini_file = read_ini_file("HBSPS_SFHdelayedtau.ini")

obs_wl, obs_f, obs_e = np.loadtxt("test/BaseGM_mock_spectra_delayed_tau.dat", unpack=True)

chain_file = "output/delayed_tau/delayed_tau_sfh.txt"
chain_results = read_chain_file(chain_file)
pct_results = compute_chain_percentiles(chain_results)
# make_plot_chains(chain_results)
ssp_data = np.loadtxt(
    "test/BaseGM_mock_spectra_delayed_tau.dat_ssp_model_spectra.dat", unpack=True)

ssp = BaseGM()
ssp.regrid(ini_file['HBSPS_SFH']['ageRange'], ini_file['HBSPS_SFH']['metRange'])
mlr = ssp.get_mass_lum_ratio(ini_file['HBSPS_SFH']['wlNormRange'])

ssp_wl, ssp_sed = ssp_data[0], ssp_data[1:]
original_ssp_wl = ssp_wl.copy()
# Load weights
ssp_weights = []
los_vel = None
sigma = None

with open("output/delayed_tau/means.txt", "r") as f:
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
ssp_weights = 10**np.array(ssp_weights)
ssp_weights = np.insert(ssp_weights, ssp_weights.size,
					   np.clip(1 - ssp_weights.sum(), a_min=0, a_max=None))
print("SSP weights: ", ssp_weights)
ssp_weights /= ssp_weights.sum()
log_met = np.sum(
    np.log10(ssp.metallicities) * ssp_weights.reshape(ssp.L_lambda.shape[:-1]).sum(axis=1)
    ) / np.sum(ssp_weights)
print("Av metallicity: ", 10**log_met)

syn_spectra = np.sum(ssp_weights[:, np.newaxis] * ssp_sed, axis=0)

syn_spectra = specBasics.smoothSpectrumFast(
    syn_spectra, sigma_pix)

syn_spectra_smooth = syn_spectra.copy()
syn_spectra_smooth_wl = ssp_wl.copy()

syn_spectra = flux_conserving_interpolation(obs_wl, ssp_wl * (1 + redshift), syn_spectra)
ssp_wl = obs_wl

norm_obs = np.nanmedian(obs_f[(obs_wl >= 5000) & (obs_wl <= 5500)])
norm_syn = np.nanmedian(syn_spectra[(ssp_wl >= 5000) & (ssp_wl <= 5500)])

residuals = obs_f / norm_obs - syn_spectra

#### STELLAR MASS
masses_mean = norm_obs * mlr.flatten() * ssp_weights / ssp_weights.sum()
print("Masses mean: ", masses_mean)

print("Weight sum: ", ssp_weights.sum())


masses = np.zeros((3, ssp_weights.size))
for n_ssp, n_mlr in zip(range(1, ssp_weights.size), mlr.flatten()):
    masses[:, n_ssp - 1] = (
        norm_obs * n_mlr * 10**pct_results[f"parameters--ssp{n_ssp}"][1],
        norm_obs * n_mlr * 10**pct_results[f"parameters--ssp{n_ssp}"][2],
        norm_obs * n_mlr * 10**pct_results[f"parameters--ssp{n_ssp}"][3]
                                 )

masses[1] = masses_mean
# Input mass history
t = np.linspace(0, 13.7, 51)
tau = 10
mass_history = 1 - (t + tau) / tau * np.exp(-t/tau)
# Pressent mass = 1e10 Msun
mass_history *= 1e10 / mass_history[-1]

recovered_mass_formed = masses.reshape((3, *ssp.L_lambda.shape[:-1])).sum(axis=1)

# From old to young SSP
mass_history_edges = np.interp(t[-1] - 10**(ini_file['HBSPS_SFH']['ageRange'] - 9), t, mass_history)

mass_formed = mass_history_edges[:-1] - mass_history_edges[1:]

print("Recovered: ",recovered_mass_formed[1], "Total mass: ", np.log10(recovered_mass_formed[1].sum()))
print("Model: ", mass_formed, "Total mass: ", np.log10(mass_formed.sum()))
print("Residuals: ", np.log10(recovered_mass_formed / mass_formed))

age_bins = (ini_file['HBSPS_SFH']['ageRange'][:-1] + ini_file['HBSPS_SFH']['ageRange'][1:]) / 2
age_width = ini_file['HBSPS_SFH']['ageRange'][1:] - ini_file['HBSPS_SFH']['ageRange'][:-1]
delta_age = 10**ini_file['HBSPS_SFH']['ageRange'][1:] - 10**ini_file['HBSPS_SFH']['ageRange'][:-1]

# Plot spectra
fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True)
ax = axs[0]
#for weight, sed in zip(ssp_weights, ssp_sed):
#    ax.plot(original_ssp_wl, sed / np.median(sed), lw=0.8, label=f'{weight:.4f}')
ax.errorbar(obs_wl, obs_f / norm_obs, yerr=obs_e / norm_obs, errorevery=5, label='obs')
ax.plot(syn_spectra_smooth_wl, syn_spectra_smooth, label='synth-sm')
ax.plot(ssp_wl, syn_spectra, label='synth-sm-vel')
ax.legend()
ax.set_ylabel(r"$\frac{L_\lambda}{<L_\lambda>_{5000-5500}}$")

#ax.set_yscale('log')
ax = axs[1]
ax.plot(obs_wl, residuals, alpha=0.7)
ax.set_ylim(np.nanpercentile(residuals[(obs_wl >= 4000) & (obs_wl <= 7000)], [5, 95]))
ax.set_xlabel("Wavelength (AA)")
twax = ax.twinx()
twax.plot(obs_wl, residuals**2 / (obs_e / norm_obs)**2, c='k', alpha=0.7)
twax.set_ylabel(r"$\chi^2$")
twax.set_yscale('log')
twax.set_ylim(0.001, 2)
plt.show()

# Plot SFHs
plt.figure()
plt.subplot(211)
plt.bar(age_bins, mass_formed, width=age_width, edgecolor='k', alpha=0.8)
plt.bar(age_bins, recovered_mass_formed[1], width=age_width, edgecolor='k',
        alpha=0.8)
# plt.errorbar(age_bins, recovered_mass_formed[1],
#              yerr=(recovered_mass_formed[1] - recovered_mass_formed[0],
#                    recovered_mass_formed[2] - recovered_mass_formed[1]),
#              xerr=age_width / 2, fmt='o', capsize=3, color='k')
mass_residuals = np.log10(mass_formed / recovered_mass_formed)

mass_res = [f"{r:.3f}" for r in mass_residuals[1]]
res = r"Residuals $\log_{10}(M/\hat{M})$=" + ", ".join(mass_res)
plt.annotate(res, xy=(0.05, 0.95), xycoords='axes fraction', va='top', ha='left')
plt.yscale("log")
plt.ylabel(r"Mass formed [M$_\odot$]")
plt.subplot(212)
plt.bar(age_bins, mass_formed / delta_age, width=age_width, edgecolor='k', alpha=0.8)
plt.bar(age_bins, recovered_mass_formed[1] / delta_age, width=age_width, edgecolor='k',
        alpha=0.8)
#plt.errorbar(age_bins, recovered_mass_formed[1] / delta_age,
#             yerr=((recovered_mass_formed[1] - recovered_mass_formed[0]) / delta_age,
#                   (recovered_mass_formed[2] - recovered_mass_formed[1]) / delta_age),
#             xerr=age_width / 2, fmt='o', capsize=3, color='k')

plt.ylabel(r"SFR [M$_\odot$/yr]")
plt.xlabel(r"$\rm \log_{10}(t_{lb} / yr)$")
plt.show()

plt.figure()
plt.subplot(121)
plt.hist(chain_results['parameters--los_vel'], weights=chain_results['weight'],
         bins=np.arange(0, 300, 300))
plt.hist(chain_results['parameters--los_vel'], weights=chain_results['weight'],
         bins=np.arange(0, 300, 30), histtype='step')
plt.axvline(200, c='k')
plt.xlabel("LOS vel (km/s)")

plt.subplot(122)
plt.hist(chain_results['parameters--sigma'], weights=chain_results['weight'],
         bins=np.arange(0, 300, 300))
plt.hist(chain_results['parameters--sigma'], weights=chain_results['weight'],
         bins=np.arange(0, 300, 30), histtype='step')
plt.axvline(150, c='k')
plt.xlabel(r"LOS $\sigma$ (km/s)")
plt.show()


# All masses
all_masses = np.zeros((chain_results['weight'].size, ssp_weights.size))
for n_ssp, n_mlr in zip(range(1, ssp_weights.size), mlr.flatten()):
    ssp_mass= norm_obs * n_mlr * 10**chain_results[f"parameters--ssp{n_ssp}"]
    plt.figure()
    plt.hist(np.log10(ssp_mass), weights=chain_results['weight'], bins=50)
    plt.show()
    
