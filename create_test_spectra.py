import numpy as np
import matplotlib.pyplot as plt

from pst.SSP import BaseGM
import specBasics


ssp = BaseGM()
print(ssp.wavelength)

ori_sed = ssp.L_lambda.copy()
ori_wl = ssp.wavelength.copy()

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

ref_spectra = ssp.L_lambda[2, 10]

ppxf_sed, lnwl, _ = specBasics.log_rebin(ori_wl, ori_sed[2, 10], velscale=velscale)

ref_spectra_sm = specBasics.smoothSpectrumFast(
    ref_spectra, sigma_pix)

print(np.diff(np.log(ssp.wavelength)), ssp.wavelength.size)
print(np.diff(lnwl), lnwl.size)

plt.figure()
plt.plot(ori_wl, ori_sed[2, 10], '-', label='Original')
plt.plot(ssp.wavelength, ref_spectra, '-', label='Reinterpolated')
plt.plot(ssp.wavelength, ref_spectra_sm, '-', label='Smoothed')
plt.plot(ssp.wavelength * (1 + redshift), ref_spectra_sm, '-', label='Redshifted')
plt.legend()
plt.show()

print("saving spectra at: BaseGM_mock_spectra.dat")
np.savetxt("BaseGM_mock_spectra.dat",
           np.array(
               [ssp.wavelength * (1 + redshift),
                ref_spectra_sm,
                ref_spectra_sm * 0.001]).T)
