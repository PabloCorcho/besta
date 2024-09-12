import numpy as np
import matplotlib.pyplot as plt

from pst.SSP import PyPopStar
from pst import models

from hbsps import specBasics
from hbsps import kinematics
from hbsps.sfh import ExponentialSFH

import extinction
import os
from astropy import units as u

ssp = PyPopStar("KRO")
ssp.cut_wavelength(3500, 9000)
# Degrade the resolution, 10.1367
velscale = 70
delta_lnwl = velscale / specBasics.constants.c.to('km/s').value
delta_lnwl = velscale / 3e5
snr = 50
# Default non-linear parameters
sigma = np.asarray([100])
vel_offset = np.asarray([0])
av = np.asarray([0.])
newBorders = np.arange(
        np.log(ssp.wavelength[0].value),
        np.log(ssp.wavelength[-1].value),
        delta_lnwl)

ssp.interpolate_sed(np.exp(newBorders))

params = {}
params['logtau'] = 0.5
params['alpha'] = 1
params['z_today'] = 0.02

sfh = ExponentialSFH()
sfh.parse_free_params(params)
sed = sfh.model.compute_SED(ssp, t_obs=sfh.today)
output_dir = "test/exponential"

print(sed)

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

ax = fig.add_subplot(212)
ax.plot(sfh.time, sfh.model.integral_SFR(sfh.time))
ax.minorticks_on()
ax.grid(visible=True, which='both')

fig.savefig(os.path.join(output_dir, f"input_spectra.png"),
            dpi=200, bbox_inches='tight')
#plt.close()
plt.show()

np.savetxt(os.path.join(output_dir, f"input_spectra.dat"),
            np.array(
                [ssp.wavelength * (1 + redshift),
                ref_spectra_sm, ref_spectra_sm_err]).T)
