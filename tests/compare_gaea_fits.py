#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 15:35:12 2024

@author: pcorchoc
"""

from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from astropy.table import Table

files = glob(f"/home/pcorchoc/Develop/HBSPS/output/photometry/*/processed_results.fits")

keys = ['a_v', 'logssfr_over_10.00_yr', 'logssfr_over_9.70_yr','logssfr_over_9.48_yr',
        'logssfr_over_9.00_yr', 'logssfr_over_8.70_yr', 'logssfr_over_8.48_yr',
        'logssfr_over_8.00_yr']
    
all_inferred = []
all_real = []
all_integral_to_real = []
for file in files:
    with fits.open(file) as hdul:
        real_values = [hdul[2].header[f"{key}_real"] for key in keys]
        integral_to_real = [hdul[2].header[f"{key}_int_to_real"] for key in keys]
        pct = [hdul[2].data[f"{key}_pct"][2] for key in keys]
    
    all_inferred.append(pct)
    all_real.append(real_values)
    all_integral_to_real.append(integral_to_real)

all_inferred = np.array(all_inferred)
all_real = np.array(all_real)
all_integral_to_real = np.array(all_integral_to_real)
# %%

fig, axs = plt.subplots(2, 7, constrained_layout=True,
                        figsize=(7 * 4,  8.5), sharex=True)


for ith, (title, ax) in enumerate(zip(keys[1:], axs[0])):
    ax.set_title(title)

    ax.plot([-14, -8], [-14, -8], 'k')
    ax.grid(visible=True)
    # ax.set_box_aspect(1)
    ax.set_ylabel("Inferred sSFR (log(sSFR/yr$^{-1}$))")

    ax.hist2d(all_real[:, ith + 1], all_inferred[:, ith + 1], bins=20,
              cmap='terrain_r', range=[[-14, -8], [-14, -8]],
              vmin=.5, vmax=10)
    ax.axvline(-11, color='k', ls=':')
    ax.axhline(-11, color='k', ls=':')

    real_passive = all_real[:, ith + 1] <= -11
    inferred_passive = all_inferred[:, ith + 1] <= -11
    
    real_ms = all_real[:, ith + 1] > -11
    inferred_ms = all_inferred[:, ith + 1] > -11
    
    passive_purity = np.count_nonzero(inferred_passive[real_passive]
                                      ) / np.count_nonzero(inferred_passive)
    passive_completeness = np.count_nonzero(real_passive[inferred_passive]
                                            ) / np.count_nonzero(real_passive)
    
    ms_purity = np.count_nonzero(inferred_ms[real_ms]
                                      ) / np.count_nonzero(inferred_ms)
    ms_completeness = np.count_nonzero(real_ms[inferred_ms]
                                            ) / np.count_nonzero(real_ms)
    
    ax.annotate(
        f"MS C/P={ms_completeness:.2f} / {ms_purity:.2f}\n"
        + f"Passive C/P={passive_completeness:.2f} / {passive_purity:.2f}",
        xy=(0.05, 0.95), xycoords='axes fraction', va='top')

for ith, (title, ax) in enumerate(zip(keys[1:], axs[1])):
    _, _, _, mappable = ax.hist2d(
        all_real[:, ith + 1], all_real[:, ith + 1] - all_inferred[:, ith + 1], bins=20,
              cmap='terrain_r', range=[[-14, -8], [-2, 2]], vmin=.5, vmax=10)
    ax.axvline(-11, color='k', ls=':')
    ax.axhline(0, color='k', ls=':')
    ax.set_ylabel("Real - Inferred (dex)")
    ax.set_xlabel(r"Real sSFR (log(sSFR/yr$^{-1}$))")
    
    if ith == 6:
        plt.colorbar(mappable, ax=ax, label='No. galaxies')
# %%
good = np.where(np.max(all_real, axis=1) > -12)[0]

fig, axs = plt.subplots(1, 7, constrained_layout=True,
                        figsize=(15,  5))

for ith, (title, ax) in enumerate(zip(keys[1:], axs)):
    ax.set_title(title)
    ax.set_box_aspect(1)
    delta_ssfr = all_real[:, ith + 1] - all_inferred[:, ith + 1]
    h, xedges, _ = ax.hist(delta_ssfr[good], bins='auto')
    percentiles = np.nanpercentile(delta_ssfr[good], [16, 50, 84])
    for p in percentiles:
        ax.axvline(p, color='k', ls=':')
    ax.annotate(f"{percentiles[1]:.2f}+/- ({percentiles[1] - percentiles[0]:.2f}, {percentiles[2] - percentiles[1]:.2f})",
                xy=(.05, .95), xycoords='axes fraction', va='top', ha='left')
    ax.set_xlabel("True - Inferred (dex)")

# %%
scale = keys[1:].copy()
scale = [k.replace("logssfr_over_", "") for k in scale]
fig, axs = plt.subplots(3, 7, #constrained_layout=True,
                        figsize=(21, 9), sharex=True, sharey=True)
axs = axs.flatten()
ax_idx = 0
for ith in range(0, 6):
    for jth in range(ith + 1, 7):
        idx = ((ith + 1) * (jth + 1))
        print(ax_idx)
        real_ratio = all_real[:, jth + 1] - all_real[:, ith + 1]
        inferred_ratio = all_inferred[:, jth + 1] - all_inferred[:, ith + 1]

        real_ms = (real_ratio > - 0.3) &  (real_ratio < 0.3)
        real_sb = real_ratio > 0.3
        real_q = real_ratio < -0.3
        
        inferred_ms = (inferred_ratio > - 0.3) &  (inferred_ratio < 0.3)
        inferred_sb = inferred_ratio > 0.3
        inferred_q = inferred_ratio < -0.3
        
        # 
        real_passive_long = all_real[:, ith + 1] <= -11
        inferred_passive_long = all_inferred[:, ith + 1] <= -11
        
        real_ms_long = all_real[:, ith + 1] > -11
        inferred_ms_long = all_inferred[:, ith + 1] > -11
        
        # 
        real_passive_short = all_real[:, jth + 1] <= -11
        inferred_passive_short = all_inferred[:, jth + 1] <= -11
        
        real_ms_short = all_real[:, jth + 1] > -11
        inferred_ms_short = all_inferred[:, jth + 1] > -11

        
        real_ageing = real_ms_long & real_ms_short
        real_quenched = real_ms_long & real_passive_short
        real_passive = real_passive_long & real_passive_short

        n_real_ageing = np.count_nonzero(real_ageing)
        n_real_quenched = np.count_nonzero(real_quenched)
        n_real_passive = np.count_nonzero(real_passive)
        
        n_real = n_real_ageing + n_real_quenched + n_real_passive
        
        inferred_ageing = inferred_ms_long & inferred_ms_short
        inferred_quenched = inferred_ms_long & inferred_passive_short
        inferred_passive = inferred_passive_long & inferred_passive_short
        
        n_inferred_ageing = np.count_nonzero(inferred_ageing)
        n_inferred_quenched = np.count_nonzero(inferred_quenched)
        n_inferred_passive = np.count_nonzero(inferred_passive)
        
        n_inferred = n_inferred_ageing + n_inferred_quenched + n_inferred_passive

        ageing_purity = np.count_nonzero(
            inferred_ageing[real_ageing]
            ) / (n_inferred_ageing + 1e-2)
        ageing_completeness = np.count_nonzero(
            real_ageing[inferred_ageing]
            ) / (n_real_ageing + 1e-2)
        
        quenched_purity = np.count_nonzero(
            inferred_quenched[real_quenched]
            ) / (n_inferred_quenched + 1e-2)
        quenched_completeness = np.count_nonzero(
            real_quenched[inferred_quenched]
            ) / (n_real_quenched + 1e-2)

        passive_purity = np.count_nonzero(
            inferred_passive[real_passive]
            ) / (n_inferred_passive + 1e-2)
        passive_completeness = np.count_nonzero(
            real_passive[inferred_passive]
            ) / (n_real_passive + 1e-2)
        
        axs[ax_idx].pie([n_real_ageing, n_real_quenched, n_real_passive],
                        labels=['Ageing', 'Quenched', 'Passive'],
                        colors=['b', 'orange', 'r'],
                        wedgeprops=dict(width=1, edgecolor='w'))
        axs[ax_idx].pie([n_inferred_ageing, n_inferred_quenched, n_real_passive],
                        # labels=['Ageing', 'Quenched', 'Passive'],
                        colors=['b', 'orange', 'r'],
                        wedgeprops=dict(width=0.5, edgecolor='w'))
        
        # axs[ax_idx].bar(['Ageing', 'Quenched', 'Passive'],
        #                 [n_inferred_ageing, n_inferred_quenched, n_real_passive],
        #                 label='Inferred', color='none', edgecolor='r')
        axs[ax_idx].annotate(f"{scale[jth]}/{scale[ith]}"
                             + f"\nAgeing(P/C) {ageing_purity:.2}/{ageing_completeness:.2}"
                             + f"\nQ(P/C) {quenched_purity:.2}/{quenched_completeness:.2}"
                             + f"\nP(P/C) {passive_purity:.2}/{passive_completeness:.2}",
                             xy=(0.05, 1.15), va='bottom',
                             xycoords='axes fraction')
        # axs[ax_idx].legend( reverse=True)
        ax_idx += 1
        
        
fig.savefig("ssfr_ratios.pdf", bbox_inches='tight')
