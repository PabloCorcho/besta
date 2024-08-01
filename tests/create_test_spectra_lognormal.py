import numpy as np
import matplotlib.pyplot as plt

from pst.SSP import PyPopStar
from pst import models

from hbsps import specBasics
from hbsps import kinematics
from hbsps import dust_extinction
from hbsps.sfh import LogNormalQuenchedSFH, LogNormalSFH

import extinction
import os
from astropy import units as u

ssp = PyPopStar("KRO")
ssp.cut_wavelength(3000, 9000)
# Degrade the resolution, 10.1367
velscale = 70
delta_lnwl = velscale / specBasics.constants.c.to('km/s').value
delta_lnwl = velscale / 3e5
snr = 100
# Default non-linear parameters
sigma = np.asarray([100])
vel_offset = np.asarray([230])
av = np.asarray([1.])

newBorders = np.arange(
        np.log(ssp.wavelength[0].value),
        np.log(ssp.wavelength[-1].value),
        delta_lnwl)

ssp.interpolate_sed(np.exp(newBorders))

params = {}

params['alpha'] = 1
params['z_today'] = 0.02
params['scale'] = .5
params['lnt0'] = 2.2
params['lnt_quench'] = 2.52  # 12.5
params['lntau_quench'] = -1.2

dummy_t = np.linspace(0, 13.7, 1000) * u.Gyr

sfh = LogNormalQuenchedSFH()
sfh.parse_free_params(params)

nonq_sfh = LogNormalSFH()
nonq_sfh.parse_free_params(params)


sed = sfh.model.compute_SED(ssp, t_obs=sfh.today)
nonq_sed = nonq_sfh.model.compute_SED(ssp, t_obs=nonq_sfh.today)
output_dir = "test/lognormal"


sigma_pix = sigma / velscale
vel_offset_pixels = vel_offset / velscale
redshift = np.exp(vel_offset /specBasics.constants.c.to('km/s').value) - 1 
convolved_spectra = kinematics.convolve_spectra(
    sed, vel_offset_pixels, sigma_pix)
ref_spectra_sm = convolved_spectra
ref_spectra_sm_err = convolved_spectra / snr

dust_model = dust_extinction.DustScreen('ccm89')

red_spectra = dust_model(ssp.wavelength, ref_spectra_sm, av=av, normalize=False)
red_spectra_err = dust_model(ssp.wavelength, ref_spectra_sm_err, av=av, normalize=False)

fig = plt.figure(constrained_layout=True)
ax = fig.add_subplot(211, title=f"Av={av[0]:.2f}, LOS_v={vel_offset[0]:.1f}, LOS_sigma={sigma[0]:.1f}")
ax.plot(ssp.wavelength, sed, label='original', lw=0.7)
ax.plot(ssp.wavelength, convolved_spectra, label='Kinematics', lw=0.7)
ax.errorbar(ssp.wavelength, red_spectra, yerr=red_spectra_err, label='Reddened', lw=0.7)

ax = fig.add_subplot(212)
ax.plot(dummy_t, nonq_sfh.model.integral_SFR(dummy_t).value)
ax.plot(dummy_t, sfh.model.integral_SFR(dummy_t).value)
#ax.set_yscale("symlog", linthresh=1e-3)
ax.minorticks_on()
#ax.set_xscale("log")
ax.grid(visible=True, which='both')

fig.savefig(os.path.join(output_dir, f"input_spectra.png"),
            dpi=200, bbox_inches='tight')
#plt.close()
plt.show()

np.savetxt(os.path.join(output_dir, f"input_spectra.dat"),
            np.array(
                [ssp.wavelength,
                 red_spectra, ref_spectra_sm_err]).T)

plt.figure()
plt.plot(ssp.wavelength, sed / nonq_sed, label='original', lw=0.7, c='k')
plt.show()
