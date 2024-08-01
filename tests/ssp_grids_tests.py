import numpy as np
import matplotlib.pyplot as plt

from pst.SSP import PyPopStar
from pst import models

from hbsps import specBasics
from hbsps import kinematics
from hbsps.sfh import PowerLawChemicalModel, ExponentialSFH

import extinction
import os
from astropy import table, units

ssp = PyPopStar("KRO")
ssp.cut_wavelength(3500, 9000)

#%%
lbtime = np.geomspace(1e-3, 13.7, 100)
time = 13.7 - lbtime[::-1]
time[-1] = 13.7

lbtime2 = np.geomspace(1e-3, 13.7, 30)
time2 = 13.7 - lbtime2[::-1]
time2[-1] = 13.7

dummy_t = np.geomspace(1e-3, 13.7, 1000) * 1e9 * units.yr

output_dir = "test/exponential"
# Degrade the resolution, 10.1367
velscale = 70
delta_lnwl = velscale / specBasics.constants.c.to('km/s').value
delta_lnwl = velscale / 3e5
snr = 50
# Default non-linear parameters
sigma = np.asarray([100])
vel_offset = np.asarray([0])
av = np.asarray([0.])

logtau = 9.5
logm_today = 10
logtau_z = 10
z_today = 0.02

mass_history_params = {'logtau': logtau, 'logm_today': logm_today}
met_history_params = {'alpha': 2, 'z_0': 0.02, 't_0': 5e9}

newBorders = np.arange(
        np.log(ssp.wavelength[0].value),
        np.log(ssp.wavelength[-1].value),
        delta_lnwl)

ssp.interpolate_sed(np.exp(newBorders))

sfh_model = ExponentialSFH(time=time * 1e9)
chemical_model = PowerLawChemicalModel(time=time * 1e9)

# Spare time sampling
sfh_model2 = ExponentialSFH(time=time2 * 1e9)
chemical_model2 = PowerLawChemicalModel(time=time2 * 1e9)

m_history = sfh_model.mass_history(mass_history_params)
z_history = chemical_model.metallicity_history(met_history_params)
# z_history = np.ones_like(m_history) * 0.02

m_history2 = sfh_model2.mass_history(mass_history_params)
z_history2 = chemical_model2.metallicity_history(met_history_params)
# z_history2 = np.ones_like(m_history2) * 0.02

tabular_model = models.Tabular_MFH(times=time * units.Gyr,
                                   masses=m_history * units.Msun,
                                   Z=z_history * units.dimensionless_unscaled)
tab_sed = tabular_model.compute_SED(ssp, t_obs=13.7 * units.Gyr)

tabular_model2 = models.Tabular_MFH(times=time2 * units.Gyr,
                                    masses=m_history2 * units.Msun,
                                   Z=z_history2 * units.dimensionless_unscaled)
tab_sed2 = tabular_model2.compute_SED(ssp, t_obs=13.7 * units.Gyr)

plt.figure()
plt.subplot(121)
plt.plot(chemical_model.time, z_history, 'o', label='Full sampling')
plt.plot(dummy_t, tabular_model.interpolate_Z(dummy_t), '-', label='Tab full')

plt.plot(chemical_model2.time, z_history2, 'o', label='Sparse sampling')
plt.plot(dummy_t, tabular_model2.interpolate_Z(dummy_t), '-', label='Tab sparse')
plt.legend()
plt.subplot(122)
plt.plot(sfh_model.time, m_history, 'o', label='Full sampling')
plt.plot(dummy_t,
         tabular_model.integral_SFR(dummy_t), '-', label='Tab full')

plt.plot(sfh_model2.time, m_history2, 'o', label='Sparse sampling')
plt.plot(dummy_t, tabular_model2.integral_SFR(dummy_t), '-',
           label='Tab sparse')
plt.legend()
plt.show()

# fig = plt.figure(constrained_layout=True)
# ax = fig.add_subplot(121)
# mappable = ax.pcolormesh(ssp.log_ages_yr.value - 9, ssp.metallicities,
#                          ssp_weights,
#                cmap='nipy_spectral', vmax=ssp_weights.max())
# plt.colorbar(mappable, ax=ax)
# ax = fig.add_subplot(122)
# mappable = ax.pcolormesh(ssp.log_ages_yr.value - 9, ssp.metallicities,
#                          ssp_weights2,
#                cmap='nipy_spectral', vmax=ssp_weights.max())
# plt.colorbar(mappable, ax=ax)



plt.figure()
plt.subplot(311)
plt.plot(ssp.wavelength, tab_sed, label='Tab')
plt.plot(ssp.wavelength, tab_sed2, label='Tab2')
plt.legend(ncol=2)
plt.subplot(312)
ratio = tab_sed / tab_sed2
plt.plot(ssp.wavelength, ratio)
#plt.ylim(ratio.min())
plt.axhline(1, c='k', ls='--', alpha=0.5)
plt.subplot(313)
plt.plot(sfh_model.time, m_history)
plt.plot(sfh_model2.time, m_history2)
plt.xscale('symlog')
#plt.yscale('symlog')
plt.show()


sigma_pix = sigma / velscale
vel_offset_pixels = vel_offset / velscale
redshift = np.exp(vel_offset /specBasics.constants.c.to('km/s').value) - 1 
convolved_spectra = kinematics.convolve_spectra(
    sed, vel_offset_pixels, sigma_pix)
ref_spectra_sm = convolved_spectra
ref_spectra_sm_err = convolved_spectra / snr

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(211, title=f"Av={av[0]:.2f}, LOS_v={vel_offset[0]:.1f}, LOS_sigma={sigma[0]:.1f}")
ax.plot(ssp.wavelength, sed, label='original', lw=0.7)
ax.plot(ssp.wavelength, convolved_spectra, label='Redshifted', lw=0.7)
ax.legend()

ax = fig.add_subplot(212)
mappable = ax.pcolormesh(ssp.log_ages_yr - 9, ssp.metallicities, ssp_weights,
               cmap='nipy_spectral', vmax=ssp_weights.max())
plt.colorbar(mappable, ax=ax, label='Normalized weights', extend='both')
ax.set_xlabel("SSP age")
ax.set_ylabel("SSP metallicity")
fig.savefig(os.path.join(output_dir, f"exponential_sfh.png"),
            dpi=200, bbox_inches='tight')
plt.close()

np.savetxt(os.path.join(output_dir, f"exponential.dat"),
            np.array(
                [ssp.wavelength * (1 + redshift),
                ref_spectra_sm, ref_spectra_sm_err]).T)

# properties_table.add_row(subhalo_prop)

# # properties_table.write(os.path.join(output_dir, "tng_properties_table.fits"),
# #                        overwrite=True)

# %%
