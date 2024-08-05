import numpy as np
import matplotlib.pyplot as plt

from pst.SSP import PyPopStar
from pst import models, observables

from hbsps import specBasics
from hbsps import kinematics
from hbsps.sfh import FixedTime_sSFR_SFH

import extinction
import os
from astropy import units as u

ssp = PyPopStar("KRO")
ssp.cut_wavelength(1000, 30000)
# Degrade the resolution, 10.1367
velscale = 150
delta_lnwl = velscale / specBasics.constants.c.to('km/s').value
delta_lnwl = velscale / 3e5
snr = 50
redshift = 0.0
# Default non-linear parameters

av = np.asarray([0.])
newBorders = np.arange(
        np.log(ssp.wavelength[0].value),
        np.log(ssp.wavelength[-1].value),
        delta_lnwl)

ssp.interpolate_sed(np.exp(newBorders))

list_of_filter_names = [
    "CFHT_MegaCam.u",
    "Subaru_HSC.g",
    "CFHT_MegaCam.r",
    "PANSTARRS_PS1.i",
    "Subaru_HSC.z",
    "Euclid_NISP.H",
    "Euclid_NISP.J",
    "Euclid_NISP.Y",
    "Euclid_VIS.vis"

]
filter_list = observables.load_photometric_filters(list_of_filter_names)

delta_time = np.array([0.1, 0.3, .5, 1., 3., 5.]) * u.Gyr

sfh = FixedTime_sSFR_SFH(delta_time=delta_time, redshift=redshift)

params = sfh.free_params.copy()
params['alpha'] = 0.0
params['z_today'] = 0.02

for k, v in params.items():
    if "logssfr" in k:
        params[k] = -10.

return_code = sfh.parse_free_params(params)
print("Valid parameters:", return_code)
if return_code == 0:
    raise ArithmeticError("Input parameters are not valid")

# Fix this parameter
sfh.free_params['alpha'] = [0]
print("Time at observation: ", sfh.today)

sed = sfh.model.compute_SED(ssp, t_obs=sfh.today)
output_dir = "/home/pcorchoc/Develop/HBSPS/test_data/photometry"

# Convert luminosity to flux at 10 pc
sed /= 4 * np.pi * (10 * u.pc)**2

photometry = np.zeros(len(filter_list)) * u.Jy

for ith, f in enumerate(filter_list):
    f.interpolate(ssp.wavelength * (1 + sfh.redshift))
    flux, _ = f.get_fnu(sed)
    photometry[ith] = flux
print(photometry)

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(211, title=f"Av={av[0]:.2f}")
twax = ax.twinx()
ax.plot(ssp.wavelength * (1 + sfh.redshift), sed.to('erg/s/angstrom/cm^2'),
        label='original', lw=0.7)
for ith, f in enumerate(filter_list):
    line = twax.plot(f.filter_wavelength, f.filter_resp)
    eff_wl = f.effective_wavelength()
    ax.plot(eff_wl, photometry[ith] / (3.34e4 * eff_wl**2), 'o',
            color=line[0].get_color(), mec='k')
ax.set_xlim(3000, 30000)

ax = fig.add_subplot(212)
ax.plot(sfh.time, sfh.model.integral_SFR(sfh.time))
ax.minorticks_on()
ax.grid(visible=True, which='both')
ax.set_ylabel("integral_SFR")

# fig.savefig(os.path.join(output_dir, f"input_spectra.png"),
#             dpi=200, bbox_inches='tight')
#plt.close()
plt.show()

# np.savetxt(os.path.join(output_dir, f"input_spectra.dat"),
#             np.array(
#                 [ssp.wavelength * (1 + redshift),
#                 ref_spectra_sm, ref_spectra_sm_err]).T)

sfh.make_ini(os.path.join(output_dir, f"fixedtime_ssfr_sfh_values.ini"))

np.savetxt(os.path.join(output_dir, f"fixedtime_ssfr_sfh_photometry.dat"),
           np.array([list_of_filter_names, photometry, photometry / snr],
                    dtype=object).T,
                    fmt="%s")