#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 11:16:07 2024

@author: pcorchoc
"""

from glob import glob
import os
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

from FAENA.models.ageing_diagram import get_ssfr_ad_class, get_ssfr_activity_class

from ad_probability import get_all_from_results_file, SKEY, LKEY


cosmo = FlatLambdaCDM(H0=70., Om0=0.28)

output_dir = '../output/photometry/illustris'
output_dir = '../output/photometry/old_illustris_dust_and_redshift'
output_dir = '../output/photometry/illustris_dust_and_redshift'

files = glob(os.path.join(output_dir, "**", "processed_results.fits"))

ssfr_keys = ['logssfr_over_10.00_yr',
             'logssfr_over_9.70_yr',
             'logssfr_over_9.48_yr',
             'logssfr_over_9.00_yr',
             'logssfr_over_8.70_yr',
             'logssfr_over_8.48_yr',
             'logssfr_over_8.00_yr']

all_keys = ["a_v", "z_today", "normalization"] + ssfr_keys

all_ad_prob = []
all_ssfr_prob = []

all_results = {"input": []}
for key in all_keys:
    all_results[key] = {"p5": [], "p16": [], "p50": [], "p84": [],
                        "p95": [],
                        "real": [],
                        "int_to_real": []}

def get_real_values(path):
    real_values = {}
    with open(path, "r") as f:
        for line in f.readlines():

            line = line.strip("#").strip("\n").replace(" ", "")
            k, v = line.split("=")
            if "+/-" in v:
                 mag, magerr = v.split("+/-")
                 v = (float(mag), float(magerr))
            elif "ssfr" in k:
                v = np.log10(float(v) + 1e-14)
            elif "normalization" in k:
                v = float(v) * cosmo.luminosity_distance(real_values['redshift']
                                                         ).to_value('10 pc')**-2
            else:
                v = float(v)
            real_values[k] = v

            if "VIS" in line:
                break

    return real_values

for file in files:
    subhalo = os.path.basename(os.path.dirname(file))
    path_to_photo = os.path.join("..", "test_data", "photometry",
                                 "illustris_dust_and_redshift",
                                 f"{subhalo}_photometry.dat")
    
    ad_prob, ssfr_prob, figures = get_all_from_results_file(
        file.replace("processed_results.fits", "SFH_results.txt"))
    all_ad_prob.append(ad_prob)
    all_ssfr_prob.append(ssfr_prob)
    real_input = get_real_values(path_to_photo)
    all_results['input'].append(real_input)
    with fits.open(file) as hdul:
        for key in all_keys:
            p5, p16, p50, p84, p95 = hdul[2].data[f'{key}_pct']
            if key == "normalization":
                real = real_input['normalization']
                int_to_real = np.nan
            else:
                real = hdul[2].header[f'{key}_real']
                int_to_real = hdul[2].header[f'{key}_int_to_real']
            all_results[key]['p5'].append(p5)
            all_results[key]['p16'].append(p16)
            all_results[key]['p50'].append(p50)
            all_results[key]['p84'].append(p84)
            all_results[key]['p95'].append(p95)
            all_results[key]['real'].append(real)
            all_results[key]['int_to_real'].append(int_to_real)
        
        pdf_logssfr_9_logssfr_8 = hdul[3].data
        ssfr_9 = np.linspace(hdul[3].header['A0_INI'],
                    hdul[3].header['A0_END'],
                    pdf_logssfr_9_logssfr_8.shape[0])
        ssfr_8 = np.linspace(hdul[3].header['A1_INI'],
                    hdul[3].header['A1_END'],
                    pdf_logssfr_9_logssfr_8.shape[1])

redshift = np.array([p["redshift"] for p in all_results['input']])
umagerr = np.array([p["CFHT_MegaCam.u"][1] for p in all_results['input']])
highsnr = umagerr < 0.05
print("High SNR: ", np.count_nonzero(highsnr), " out of ", highsnr.size)
#%%

real_ssfr_long = np.array(all_results[LKEY.replace("parameters--", "")]['real'])
real_ssfr_short = np.array(all_results[SKEY.replace("parameters--", "")]['real'])

real_delta_ssfr = real_ssfr_short - real_ssfr_long


all_below = np.array([p['p_below'] for p in all_ssfr_prob])

# %%===========================================================================
# One to one
# =============================================================================
fig, axs = plt.subplots(nrows=4, ncols=len(ssfr_keys), sharex=False, sharey=False,
                        constrained_layout=True, figsize=(3 * len(ssfr_keys), 12),
                        height_ratios=[1, 3, 3, 3])

for ax, key in zip(axs[0], ssfr_keys):
    ax.set_title(key)
    offset = np.array(all_results[key]['real']) - np.array( all_results[key]['p50'])
    pct_offsets = np.nanpercentile(offset, [16, 50, 84])
    ax.hist(offset[highsnr], bins='auto', histtype='step', density=True)
    ax.hist(offset, bins='auto', histtype='step', density=True)
    for pct in pct_offsets:
        ax.axvline(pct, ls=':', color='k')
    ax.set_xlabel("Real - P50 (dex)")
    ax.annotate(f"{pct_offsets[1]:.2f} +/- ({pct_offsets[2] - pct_offsets[1]:.2f}, {pct_offsets[1] - pct_offsets[0]:.2f})",
                xy=(0.05, 0.95), xycoords='axes fraction', va='top')


for ax, key in zip(axs[1], ssfr_keys):
    # ax.set_title(key)
    ax.plot([-14, -9], [-14, -9], color='k', ls='--', alpha=0.8)
    ax.axhline(-11, color='k', ls=':')
    ax.axvline(-11, color='k', ls=':')
    mappable = ax.scatter(all_results[key]['real'], all_results[key]['p50'],
                          c=umagerr, vmin=0.01, vmax=0.2, alpha=0.7)
    ax.set_xlabel("sSFR Real")
    ax.set_ylabel("sSFR inferred")

plt.colorbar(mappable, ax=ax, label=r"$u$-band err (dex)")

for ax, key in zip(axs[2], ssfr_keys):
    # ax.set_title(key)
    ax.plot([-14, -9], [-14, -9], color='k', ls='--', alpha=0.8)
    ax.axhline(-11, color='k', ls=':')
    ax.axvline(-11, color='k', ls=':')
    mappable = ax.scatter(all_results[key]['real'], all_results[key]['p50'],
                          c=np.array(all_results["a_v"]["real"]) - np.array(all_results["a_v"]["p50"]),
                          vmin=0.01, vmax=0.7, alpha=0.7)
    ax.set_xlabel("sSFR Real")
    ax.set_ylabel("sSFR inferred")

plt.colorbar(mappable, ax=ax, label=r"Real $A_v$ - Inferred $A_v$")

for ax, key in zip(axs[3], ssfr_keys):
    # ax.set_title(key)
    ax.plot([-14, -9], [-14, -9], color='k', ls='--', alpha=0.8)
    ax.axhline(-11, color='k', ls=':')
    ax.axvline(-11, color='k', ls=':')
    mappable = ax.scatter(all_results[key]['real'], all_results[key]['p50'],
                          # c=np.array(all_results["z_today"]["real"]) / np.array(all_results["z_today"]["p50"]),
                          # vmin=0.5, vmax=1.5, alpha=0.7,
                          # cmap='seismic'
                          c=np.log10(np.array(all_results["z_today"]["p50"]) / 0.02),
                          vmin=-.5, vmax=0.2, alpha=0.7,
                          )
    ax.set_xlabel("sSFR Real")
    ax.set_ylabel("sSFR inferred")

plt.colorbar(mappable, ax=ax, label=r"Real $Z$ / Inferred $Z$")

#%%

fig, ax = plt.subplots()
ax.set_title("Dust extinction")
ax.plot([0, 2.5], [0, 2.5], color='k', ls='--', alpha=0.8)
ax.plot(all_results["a_v"]["real"], all_results["a_v"]["p50"],
        '.', alpha=0.7)
ax.set_xlabel("Av Real")
ax.set_ylabel("Av inferred")


fig, ax = plt.subplots()
ax.set_title("Metallicity")
ax.plot([0, 0.05], [0, 0.05], color='k', ls='--', alpha=0.8)
ax.plot(all_results["z_today"]["real"], all_results["z_today"]["p50"],
        # all_results["z_today"]["p50"],
        '.', alpha=0.7)
ax.set_xlabel("Z_today Real")
ax.set_ylabel("Z_today inferred")

# =============================================================================
# Mass
# =============================================================================

plt.figure()
delta_logm = np.log10(all_results['normalization']['real']) - np.log10(all_results['normalization']['p50'])
h, xedges, _ = plt.hist(delta_logm, range=[-2, 2], bins='auto')
p16, p50, p84 = np.nanpercentile(delta_logm, [16, 50, 84])
plt.annotate(f"{p50:.2f} +/- ({p84 - p50:.2f}, {p50 - p84:.2f})",
            xy=(0.05, 0.95), xycoords='axes fraction', va='top')
plt.xlabel("logM real - logM inferred")
# =============================================================================
# Offset histogram
# =============================================================================
fig, axs = plt.subplots(nrows=1, ncols=len(ssfr_keys), sharex=True, sharey=True,
                        constrained_layout=True, figsize=(3 * len(ssfr_keys), 3))

for ax, key in zip(axs, ssfr_keys):
    ax.set_title(key)
    offset = np.array(all_results[key]['real']) - np.array( all_results[key]['p50'])
    pct_offsets = np.nanpercentile(offset, [16, 50, 84])
    ax.hist(offset, bins='auto')
    for pct in pct_offsets:
        ax.axvline(pct, ls=':', color='k')
    ax.set_xlabel("Real - P50 (dex)")
    ax.annotate(f"{pct_offsets[1]:.2f} +/- ({pct_offsets[2] - pct_offsets[1]:.2f}, {pct_offsets[1] - pct_offsets[0]:.2f})",
                xy=(0.05, 0.95), xycoords='axes fraction', va='top')

# =============================================================================
# 2D Offset histogram
# =============================================================================
fig, axs = plt.subplots(nrows=1, ncols=len(ssfr_keys), sharex=True, sharey=True,
                        constrained_layout=True, figsize=(3 * len(ssfr_keys), 3))

axs[0].set_ylabel("Real - P50 (dex)")
for ax, key in zip(axs, ssfr_keys):
    ax.set_title(key)
    offset = np.array(all_results[key]['real']) - np.array( all_results[key]['p50'])
    pct_offsets = np.nanpercentile(offset[highsnr], [16, 50, 84])
    H, xedges, yedges = np.histogram2d(all_results[key]['real'],
                                       offset, bins=(10, 30),
                                       weights=highsnr,
                                       range=[[-14, -8], [-2, 2]])
    
    H_cum = np.cumsum(H * np.diff(yedges)[np.newaxis, :], axis=1)
    H_cum /= H_cum[:, -1][:, np.newaxis]
    
    xbins = (xedges[:-1] + xedges[1:]) / 2
    ybins = (yedges[:-1] + yedges[1:]) / 2
    mappable = ax.contourf(xbins, ybins, H_cum.T, vmin=0, vmax=1,
                           levels=[0, .16, .5, .84, 1.])
    
    # for pct in pct_offsets:
    #     ax.axhline(pct, ls=':', color='w')
    ax.grid(visible=True)
    ax.set_xlabel("Real sSFR")
    ax.annotate(f"{pct_offsets[1]:.2f} +/- ({pct_offsets[2] - pct_offsets[1]:.2f}, {pct_offsets[1] - pct_offsets[0]:.2f})",
                xy=(0.05, 0.95), xycoords='axes fraction', va='top')

plt.colorbar(mappable, ax=ax)

# =============================================================================
# Normalized Offset histogram
# =============================================================================
fig, axs = plt.subplots(nrows=1, ncols=len(ssfr_keys), sharex=True, sharey=True,
                        constrained_layout=True, figsize=(3 * len(ssfr_keys), 3))

for ax, key in zip(axs, ssfr_keys):
    ax.set_title(key)
    offset = (
        np.array(all_results[key]['real']) - np.array( all_results[key]['p50'])) / (
            np.array(all_results[key]['p84']) - np.array( all_results[key]['p16']))
    ax.hist(offset, range=[-3, 3], bins='auto')
    ax.set_xlabel(r"$\rm \frac{Real - P50}{P84 - P16}$")


# =============================================================================
#  probability integral transform
# =============================================================================
fig, axs = plt.subplots(nrows=1, ncols=len(ssfr_keys), sharex=True, sharey=True,
                        constrained_layout=True, figsize=(3 * len(ssfr_keys), 3))

for ax, key in zip(axs, ssfr_keys):
    ax.set_title(key)
    ax.hist(all_results[key]['int_to_real'], bins=30, range=[0, 1], density=True)
    ax.set_xlabel("CPF(x=true)")
ax.set_yscale("log")


# %%===========================================================================
# sSFR classification
# =============================================================================
for ith in range(0, len(ssfr_keys) - 1):
    for jth in range(ith + 1, len(ssfr_keys)):
        key_long = ssfr_keys[ith]
        key_short = ssfr_keys[jth]
        print(key_long, key_short)
        

        inferred_ssfr_class_short = get_ssfr_activity_class(all_results[key_short]["p50"])
        inferred_ssfr_class_long = get_ssfr_activity_class(all_results[key_long]["p50"])
        
        real_ssfr_class_short = get_ssfr_activity_class(all_results[key_short]["real"])
        real_ssfr_class_long = get_ssfr_activity_class(all_results[key_long]["real"])
        
        long_good_class = np.count_nonzero(
            inferred_ssfr_class_long == real_ssfr_class_long) / real_ssfr_class_long.size
        short_good_class = np.count_nonzero(
            inferred_ssfr_class_short == real_ssfr_class_short) / real_ssfr_class_short.size
        
        good_all = np.count_nonzero((inferred_ssfr_class_long == real_ssfr_class_long
                    ) & (inferred_ssfr_class_short == real_ssfr_class_short)
                                    ) / real_ssfr_class_long.size

        real_passive_short = real_ssfr_class_short <= 1
        real_sf_short = real_ssfr_class_short >= 2
        real_passive_long = real_ssfr_class_long <= 1
        real_sf_long = real_ssfr_class_long >= 2
        
        inferred_passive_short = inferred_ssfr_class_short <= 1
        inferred_sf_short = inferred_ssfr_class_short >= 2
        inferred_passive_long = inferred_ssfr_class_long <= 1
        inferred_sf_long = inferred_ssfr_class_long >= 2
        
        good_class_short = (
            real_passive_short & inferred_passive_short
            )  | (real_sf_short & inferred_sf_short)
        good_class_long = (
            real_passive_long & inferred_passive_long
            )  | (real_sf_long & inferred_sf_long)
        good_class_both = good_class_short & good_class_long

        n_real_sf_short = np.count_nonzero(real_sf_short)
        n_real_passive_short = np.count_nonzero(real_passive_short)
        n_real_sf_long = np.count_nonzero(real_sf_long)
        n_real_passive_long = np.count_nonzero(real_passive_long)
        
        n_inferred_sf_short = np.count_nonzero(inferred_sf_short)
        n_inferred_passive_short = np.count_nonzero(inferred_passive_short)
        n_inferred_sf_long = np.count_nonzero(inferred_sf_long)
        n_inferred_passive_long = np.count_nonzero(inferred_passive_long)
        
        sf_purity_short = np.count_nonzero(
            inferred_sf_short[real_sf_short]
            ) / (n_inferred_sf_short + 1e-2)
        sf_completeness_short = np.count_nonzero(
            real_sf_short[inferred_sf_short]
            ) / (n_real_sf_short + 1e-2)
        
        passive_purity_short = np.count_nonzero(
            inferred_passive_short[real_passive_short]
            ) / (n_inferred_passive_short + 1e-2)
        passive_completeness_short = np.count_nonzero(
            real_passive_short[inferred_passive_short]
            ) / (n_real_passive_short + 1e-2)
        
        sf_purity_long = np.count_nonzero(
            inferred_sf_long[real_sf_long]
            ) / (n_inferred_sf_long + 1e-2)
        sf_completeness_long = np.count_nonzero(
            real_sf_long[inferred_sf_long]
            ) / (n_real_sf_long + 1e-2)
        
        passive_purity_long = np.count_nonzero(
            inferred_passive_long[real_passive_long]
            ) / (n_inferred_passive_long + 1e-2)
        passive_completeness_long = np.count_nonzero(
            real_passive_long[inferred_passive_long]
            ) / (n_real_passive_long + 1e-2)
        
        
        print("Short timescale:\n",
              f"- SF purity: {sf_purity_short}\n",
              f"- SF completeness: {sf_completeness_short}\n",
              f"- Passive purity: {passive_purity_short}\n",
              f"- Passive completeness: {passive_completeness_short}")
        print("Long timescale:\n",
              f"- SF purity: {sf_purity_long}\n",
              f"- SF completeness: {sf_completeness_long}\n",
              f"- Passive purity: {passive_purity_long}\n",
              f"- Passive completeness: {passive_completeness_long}")
        print("#" * 50)
# %%===========================================================================
# AD classification
# =============================================================================
fig, axs = plt.subplots(3, 7, #constrained_layout=True,
                        figsize=(21, 12), sharex=True, sharey=True)
axs = axs.flatten()
ax_idx = 0
for ith in range(0, len(ssfr_keys) - 1):
    for jth in range(ith + 1, len(ssfr_keys)):
        key_long = ssfr_keys[ith]
        key_short = ssfr_keys[jth]
        
        # key_long = "logssfr_over_9.00_yr"
        # key_short = "logssfr_over_8.00_yr"
        print(key_long, key_short)
        
        inferred_ad_class = get_ssfr_ad_class(all_results[key_short]["p50"],
                                              all_results[key_long]["p50"])
        real_ad_class = get_ssfr_ad_class(all_results[key_short]["real"],
                                              all_results[key_long]["real"])

        real_ageing = real_ad_class == 1
        real_undet = real_ad_class == 2
        real_quenched = real_ad_class == 3
        real_retired = real_ad_class == 4
        
        inferred_ageing = inferred_ad_class == 1
        inferred_undet = inferred_ad_class == 2
        inferred_quenched = inferred_ad_class == 3
        inferred_retired = inferred_ad_class == 4
        
        n_real_ageing = np.count_nonzero(real_ageing)
        n_real_undet = np.count_nonzero(real_undet)
        n_real_quenched = np.count_nonzero(real_quenched)
        n_real_retired = np.count_nonzero(real_retired)

        n_inferred_ageing = np.count_nonzero(inferred_ageing)
        n_inferred_undet = np.count_nonzero(inferred_undet)
        n_inferred_quenched = np.count_nonzero(inferred_quenched)
        n_inferred_retired = np.count_nonzero(inferred_retired)

        n_good_class = np.count_nonzero(inferred_ad_class == real_ad_class)
        
        ageing_purity = np.count_nonzero(
            inferred_ageing[real_ageing]
            ) / (n_inferred_ageing + 1e-2)
        ageing_completeness = np.count_nonzero(
            real_ageing[inferred_ageing]
            ) / (n_real_ageing + 1e-2)
        
        undet_purity = np.count_nonzero(
            inferred_undet[real_undet]
            ) / (n_inferred_undet + 1e-2)
        undet_completeness = np.count_nonzero(
            real_undet[inferred_undet]
            ) / (n_real_undet + 1e-2)
        
        quenched_purity = np.count_nonzero(
            inferred_quenched[real_quenched]
            ) / (n_inferred_quenched + 1e-2)
        quenched_completeness = np.count_nonzero(
            real_quenched[inferred_quenched]
            ) / (n_real_quenched + 1e-2)

        retired_purity = np.count_nonzero(
            inferred_retired[real_retired]
            ) / (n_inferred_retired + 1e-2)
        retired_completeness = np.count_nonzero(
            real_retired[inferred_retired]
            ) / (n_real_retired + 1e-2)
        
        idx = ((ith + 1) * (jth + 1))
        axs[ax_idx].pie([n_real_ageing, n_real_undet, n_real_quenched, n_real_retired],
                        labels=['Ageing', "Undet", 'Quenched', 'Retired'],
                        colors=['b', 'green', 'orange', 'r'],
                        wedgeprops=dict(width=1, edgecolor='w'))
        axs[ax_idx].pie([n_inferred_ageing, n_inferred_undet, n_inferred_quenched, n_real_retired],
                        # labels=['Ageing', 'Quenched', 'retired'],
                        colors=['b', 'green', 'orange', 'r'],
                        wedgeprops=dict(width=0.5, edgecolor='w'))
        
        # axs[ax_idx].bar(['Ageing', 'Quenched', 'retired'],
        #                 [n_inferred_ageing, n_inferred_quenched, n_real_retired],
        #                 label='Inferred', color='none', edgecolor='r')
        axs[ax_idx].annotate(f"{key_long}/{key_short}"
                             + f"\nA(P/C) {ageing_purity:.2}/{ageing_completeness:.2}"
                             + f"\nU(P/C) {undet_purity:.2}/{undet_completeness:.2}"
                             + f"\nQ(P/C) {quenched_purity:.2}/{quenched_completeness:.2}"
                             + f"\nR(P/C) {retired_purity:.2}/{retired_completeness:.2}",
                             xy=(0.05, 1.), va='bottom',
                             xycoords='axes fraction',
                             fontsize=7)
        # axs[ax_idx].legend( reverse=True)
        ax_idx += 1

fig.savefig("illustris_ad_class.pdf", bbox_inches='tight')