import numpy as np
import matplotlib.pyplot as plt

from pst.SSP import PyPopStar
from hbsps import specBasics
import h5py

import extinction
import os

ssp = PyPopStar("KRO")
ssp.cut_wavelength(3500, 9000)

tng_data = "/home/pcorchoc/Research/ageing_diagram_models/IllustrisTNG-Galaxies/TNG100-1/IllustrisTNG100-1_SFH_long.hdf5"
file = h5py.File(tng_data)

ages = file['lookbacktime']['lookbacktime'][:]
times = ages[-1] - ages
extension = 'twohalfmassrad'

subhalos = ['sub_id_588577', 'sub_id_69535']
# subhalos = [k for k in file.keys() if "sub_id_" in k][::500]
print(len(subhalos))

output_dir = "test/tng"
# Degrade the resolution
velscale = 70
delta_lnwl = velscale / specBasics.constants.c.to('km/s').value
sigma = 100
vel_offset = 200
snr = 100

newBorders = np.arange(
        np.log(ssp.wavelength[0]),
        np.log(ssp.wavelength[-1]),
        delta_lnwl)

ssp.interpolate_sed(np.exp(newBorders))

for subhalo in subhalos:
    mass_history = file[subhalo][extension]['mass_history'][()] * 1e10
    met_history = file[subhalo][extension]['metallicity_history'][()]
    met_history = 0.02 * np.ones_like(met_history)
    tng_sed, tng_ssp_weights = ssp.compute_SED(
        times[::-1] * 1e9, mass_history[::-1], met_history[::-1])

    tng_dust_free = tng_sed.copy()
    tng_sed = extinction.apply(extinction.ccm89(ssp.wavelength, 1.0, 3.1),
                               tng_sed)

    fig = plt.figure(constrained_layout=True)
    ax =fig.add_subplot(
        121, title=f"Present mass: {np.log10(mass_history.max()):.2f}")
    ax.plot(times, np.log10(mass_history))

    ax = fig.add_subplot(122, sharex=ax)
    ax.plot(times, met_history)
    ax.set_xlabel("Cosmic time (Gyr)")
    fig.savefig(os.path.join(output_dir, f"{subhalo}_sfh.png"),
                dpi=200, bbox_inches='tight')

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

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(121)
    ax.plot(ori_wl, tng_dust_free, '-', label='Original')
    ax.plot(ori_wl, ori_sed, '-', label='Reddened')
    ax.plot(ssp.wavelength, ref_spectra, '-', label='Reinterpolated')
    ax.plot(ssp.wavelength, ref_spectra_sm, label='Smoothed')
    ax.errorbar(ssp.wavelength * (1 + redshift), ref_spectra_sm,
                 yerr=ref_spectra_sm_err, errorevery=5, capsize=3, label='Redshifted')
    ax.legend()
    
    ax = fig.add_subplot(122)
    mappable = ax.pcolormesh(ssp.metallicities, ssp.log_ages_yr,
                np.log10(tng_ssp_weights),
                vmax=np.log10(tng_ssp_weights.max()),
                vmin=np.log10(tng_ssp_weights.max()) - 4,
                cmap='nipy_spectral')
    plt.colorbar(mappable, ax=ax, label='Normalized weights', extend='both')
    fig.savefig(os.path.join(output_dir, f"{subhalo}_sed.png"),
                dpi=200, bbox_inches='tight')


    np.savetxt(os.path.join(output_dir, f"tng_{subhalo}.dat"),
               np.array(
                   [ssp.wavelength * (1 + redshift),
                    ref_spectra_sm, ref_spectra_sm_err]).T)
