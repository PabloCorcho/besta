import numpy as np
import matplotlib.pyplot as plt

from pst.SSP import PyPopStar
import specBasics
import h5py

ssp = PyPopStar("KRO")
ssp.cut_wavelength(3500, 9000)

tng_data = "/home/pcorchoc/Research/ageing_diagram_models/IllustrisTNG-Galaxies/TNG100-1/IllustrisTNG100-1_SFH_long.hdf5"
file = h5py.File(tng_data)

ages = file['lookbacktime']['lookbacktime'][:]
times = ages[-1] - ages
extension = 'twohalfmassrad'

subhalos = ['sub_id_588577', 'sub_id_69535']

# Degrade the resolution
velscale = 50
delta_lnwl = velscale / specBasics.constants.c.to('km/s').value
sigma = 100
vel_offset = 200
snr = 50

newBorders = np.arange(
        np.log(ssp.wavelength[0]),
        np.log(ssp.wavelength[-1]),
        delta_lnwl)

ssp.interpolate_sed(np.exp(newBorders))

for subhalo in subhalos:
    mass_history = file[subhalo][extension]['mass_history'][()] * 1e10
    met_history = file[subhalo][extension]['metallicity_history'][()]

    tng_sed, tng_ssp_weights = ssp.compute_SED(times[::-1] * 1e9, mass_history[::-1], met_history[::-1])

    plt.figure()
    plt.subplot(121)
    plt.plot(times, mass_history)
    plt.subplot(122)
    plt.plot(times, met_history)
    plt.show()

    plt.figure()
    plt.subplot(121)
    plt.plot(ssp.wavelength, tng_sed)
    plt.subplot(122)
    plt.imshow(np.log10(tng_ssp_weights), aspect='auto')
    plt.colorbar()
    plt.show()

    # Degrade the resolution
    delta_lnwl = velscale / specBasics.constants.c.to('km/s').value
    sigma_pix = sigma / velscale
    vel_offset_pixels = vel_offset / velscale
    redshift = np.exp(vel_offset /specBasics.constants.c.to('km/s').value) - 1 


    ori_sed = tng_sed.copy()
    ori_wl = ssp.wavelength.copy()

    ref_spectra = tng_sed

    ppxf_sed, lnwl, _ = specBasics.log_rebin(ori_wl, ori_sed, velscale=velscale)
    ref_spectra_sm = specBasics.smoothSpectrumFast(
        ref_spectra, sigma_pix)

    ref_spectra_sm_err = ref_spectra_sm / snr

    plt.figure()
    plt.plot(ori_wl, ori_sed, '-', label='Original')
    plt.plot(ssp.wavelength, ref_spectra, '-', label='Reinterpolated')
    plt.plot(ssp.wavelength, ref_spectra_sm, label='Smoothed')
    plt.errorbar(ssp.wavelength * (1 + redshift), ref_spectra_sm,
                 yerr=ref_spectra_sm_err, errorevery=5, capsize=3, label='Redshifted')
    plt.legend()
    plt.show()

    np.savetxt(f"test/tng_{subhalo}.dat",
               np.array(
                   [ssp.wavelength * (1 + redshift),
                    ref_spectra_sm, ref_spectra_sm_err]).T)
