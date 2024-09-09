import numpy as np
import matplotlib.pyplot as plt

from pst.SSP import PyPopStar
from hbsps import specBasics
from hbsps import kinematics
import h5py

import extinction
import os
from astropy import table

ssp = PyPopStar("KRO")
ssp.cut_wavelength(3500, 9000)

tng_data = "/home/pcorchoc/Research/ageing_diagram_models/IllustrisTNG-Galaxies/TNG100-1/IllustrisTNG100-1_SFH_long.hdf5"
file = h5py.File(tng_data)

ages = file['lookbacktime']['lookbacktime'][:]
times = ages[-1] - ages
mass_formed_lb_times = np.array([0, 0.01, 0.1, 0.3, 1.0, 3.0, 10.])

extension = 'twohalfmassrad'

subhalos = ['sub_id_588577', 'sub_id_69535']
#subhalos = [k for k in file.keys() if "sub_id_" in k][::500]
print(len(subhalos))

output_dir = "test/tng"
# Degrade the resolution
velscale = 70
delta_lnwl = velscale / specBasics.constants.c.to('km/s').value
snr = 300
# Default non-linear parameters
sigma = np.asarray([100])
vel_offset = np.asarray([0])
av = np.asarray([0.])

newBorders = np.arange(
        np.log(ssp.wavelength[0]),
        np.log(ssp.wavelength[-1]),
        delta_lnwl)

ssp.interpolate_sed(np.exp(newBorders))

properties_table = table.Table(names=['ID', 'MassFormed', 'MWAge', 'LWAge',
                                      'MWMetal', 'LWMetal',
                                      'av', 'los_vel', 'los_sigma'],
                               dtype=[str, np.ndarray] + [float] * 7)

for subhalo in subhalos:
    subhalo_prop = {'ID': subhalo.strip("sub_id")}
    mass_history = file[subhalo][extension]['mass_history'][()] * 1e10
    met_history = file[subhalo][extension]['metallicity_history'][()]
    #met_history = 0.02 * np.ones_like(mass_history)
    mass_at_t = 10**np.interp(mass_formed_lb_times, ages, np.log10(mass_history + 1))
    mass_formed = mass_at_t[0] - mass_at_t[1:]
    subhalo_prop['MassFormed'] = mass_formed

    tng_sed, tng_ssp_weights = ssp.compute_SED(
        times[::-1] * 1e9, mass_history[::-1], met_history[::-1],
        # plot_interpolation=True
        )
    mlr = ssp.get_mass_lum_ratio([5000, 5500]).T

    lw_age = np.sum(tng_ssp_weights * ssp.log_ages_yr[:, np.newaxis])
    lw_logmet = np.sum(tng_ssp_weights * np.log10(ssp.metallicities[np.newaxis, :]))
    subhalo_prop['LWAge'] = lw_age
    subhalo_prop['LWMetal'] = lw_logmet

    mw_age = np.sum(
        tng_ssp_weights * mlr * ssp.log_ages_yr[:, np.newaxis]
        ) / np.sum(mlr * tng_ssp_weights)
    mw_logmet = np.sum(
        tng_ssp_weights * mlr * np.log10(ssp.metallicities[np.newaxis, :])
                       ) / np.sum(mlr * tng_ssp_weights)
    subhalo_prop['MWAge'] = mw_age
    subhalo_prop['MWMetal'] = mw_logmet

    # Generate random parameters for non-linear properties
    # av = np.random.uniform(0, 3, 1)
    # vel_offset = np.random.uniform(-300, 300, 1)
    # sigma = np.random.uniform(velscale / 2, velscale * 4, 1)
    subhalo_prop['av'] = av
    subhalo_prop['los_vel'] = vel_offset
    subhalo_prop['los_sigma'] = sigma

    tng_dust_free = tng_sed.copy()
    tng_sed = extinction.apply(extinction.ccm89(ssp.wavelength, av, 3.1),
                               tng_sed)

    fig = plt.figure(constrained_layout=True)
    ax =fig.add_subplot(
        121, title=f"Present mass: {np.log10(mass_history.max()):.2f}")
    ax.plot(times, np.log10(mass_history))
    for m, t in zip(np.log10(mass_formed), mass_formed_lb_times[1:]):
        ax.axhline(m, color='k')
        ax.annotate(f"{t:.3f}", xy=(1, m + 0.01), xycoords='data')

    ax = fig.add_subplot(122, sharex=ax)
    ax.plot(times, met_history)
    ax.set_xlabel("Cosmic time (Gyr)")
    ax.axhline(10**lw_logmet, color='orange', label='LW-met')
    ax.axhline(10**mw_logmet, color='red', label='MW-met')
    ax.legend()

    fig.savefig(os.path.join(output_dir, f"{subhalo}_sfh.png"),
                dpi=200, bbox_inches='tight')
    plt.close()
    # Degrade the resolution
    sigma_pix = sigma / velscale
    vel_offset_pixels = vel_offset / velscale
    redshift = np.exp(vel_offset /specBasics.constants.c.to('km/s').value) - 1 


    ori_sed = tng_sed.copy()
    ori_wl = ssp.wavelength.copy()

    ref_spectra_sm = specBasics.smoothSpectrumFast(
        tng_sed, sigma_pix)

    convolved_spectra = kinematics.convolve_spectra(
        ori_sed, vel_offset_pixels, sigma_pix)
    ref_spectra_sm_err = ref_spectra_sm / snr

    fig = plt.figure(constrained_layout=True)
    ax = fig.add_subplot(211, title=f"Av={av[0]:.2f}, LOS_v={vel_offset[0]:.1f}, LOS_sigma={sigma[0]:.1f}")
    ax.plot(ori_wl, tng_dust_free, '-', label='Original', lw=0.7)
    ax.plot(ori_wl, ori_sed, '-', label='Reddened', lw=0.7)
    ax.plot(ssp.wavelength, convolved_spectra, label='Redshifted', lw=0.7)
    ax.legend()

    ax = fig.add_subplot(212)
    mappable = ax.pcolormesh(ssp.log_ages_yr, np.log10(ssp.metallicities / 0.02),
                np.log10(tng_ssp_weights.T),
                vmax=np.log10(tng_ssp_weights.max()),
                vmin=np.log10(tng_ssp_weights.max()) - 6,
                cmap='nipy_spectral')
    plt.colorbar(mappable, ax=ax, label='Normalized weights', extend='both')
    ax.set_xlabel("SSP age")
    ax.set_ylabel("SSP metallicity")
    fig.savefig(os.path.join(output_dir, f"{subhalo}_sed.png"),
                dpi=200, bbox_inches='tight')
    plt.close()

    np.savetxt(os.path.join(output_dir, f"tng_{subhalo}.dat"),
               np.array(
                   [ssp.wavelength * (1 + redshift),
                    ref_spectra_sm, ref_spectra_sm_err]).T)

    properties_table.add_row(subhalo_prop)

# properties_table.write(os.path.join(output_dir, "tng_properties_table.fits"),
#                        overwrite=True)
