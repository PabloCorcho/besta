#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 13:41:13 2024

@author: pcorchoc
"""

from glob import glob
import os
import numpy as np
from matplotlib import pyplot as plt

output_dir = '/home/pcorchoc/Develop/HBSPS/test_data/photometry/gaea_examples'

files = glob(os.path.join(output_dir, "*_photometry.dat"))

galaxies = {}

all_ssfr = np.zeros((7, 2, 300))
all_av = np.zeros((2, 300))

for file in files:
    gid = os.path.basename(file).strip("_photometry.dat")
    if int(gid) > 299:
        continue
    real_ssfrs = {}
    with open(file, "r") as f:
        for line in f.readlines():
            if "av" in line:
                line = line.strip("#").strip("\n").replace(" ", "")
                k, v = line.split("=")
                all_av[0, int(gid)] = v
            if "ssfr" in line:
                line = line.strip("#").strip("\n").replace(" ", "")
                
                k, v = line.split("=")
                real_ssfrs[k.strip("ssfr_")] = float(v)
    
    galaxies[gid] = real_ssfrs
    
    ssfrs = np.array(list(real_ssfrs.values()))
    # Read results
    result = f"/home/pcorchoc/Develop/HBSPS/output/photometry/gaea_{gid}/means.txt"
    medians = np.loadtxt(result, usecols=1)
    all_av[1, int(gid)] = medians[0]
    result_ssfrs = medians[3:-3]

    for ith in range(all_ssfr.shape[0]):
        all_ssfr[ith, :, int(gid)] = [np.log10(ssfrs[ith] + 1e-14), result_ssfrs[ith]]

all_ssfr = np.array(all_ssfr)

fig, axs = plt.subplots(2, 7, constrained_layout=True,
                        figsize=(7 * 4,  8.5), sharex=True)


for ith, (scale, ax) in enumerate(zip(real_ssfrs, axs[0])):
    ax.set_title(f"tau / Gyr = {scale}")
    ax.plot([-14, -8], [-14, -8], 'k')
    ax.grid(visible=True)
    # ax.set_box_aspect(1)
    ax.set_ylabel("Inferred sSFR (log(sSFR/yr$^{-1}$))")

    ax.hist2d(all_ssfr[ith, 0], all_ssfr[ith, 1], bins=20,
              cmap='terrain_r', range=[[-14, -8], [-14, -8]],
              vmin=.5, vmax=10)
    ax.axvline(-11, color='k', ls=':')
    ax.axhline(-11, color='k', ls=':')

    real_passive = all_ssfr[ith, 0] <= -11
    inferred_passive = all_ssfr[ith, 1] <= -11
    
    real_ms = all_ssfr[ith, 0] > -11
    inferred_ms = all_ssfr[ith, 1] > -11
    
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

for ith, (scale, ax) in enumerate(zip(real_ssfrs, axs[1])):
    _, _, _, mappable = ax.hist2d(
        all_ssfr[ith, 0], all_ssfr[ith, 0] - all_ssfr[ith, 1], bins=20,
              cmap='terrain_r', range=[[-14, -8], [-2, 2]], vmin=.5, vmax=10)
    ax.axvline(-11, color='k', ls=':')
    ax.axhline(0, color='k', ls=':')
    ax.set_ylabel("Real - Inferred (dex)")
    ax.set_xlabel(r"Real sSFR (log(sSFR/yr$^{-1}$))")
    
    if ith == 6:
        plt.colorbar(mappable, ax=ax, label='No. galaxies')

# %%
good = np.where(np.max(all_ssfr[:, 0, :], axis=0) > -12)[0]

fig, axs = plt.subplots(1, 7, constrained_layout=True,
                        figsize=(15,  5))

for ith, (scale, ax) in enumerate(zip(real_ssfrs.keys(), axs)):
    ax.set_title(f"tau / Gyr = {scale}")
    ax.set_box_aspect(1)
    delta_ssfr = all_ssfr[ith, 0] - all_ssfr[ith, 1]
    h, xedges, _ = ax.hist(delta_ssfr[good], bins=30)
    percentiles = np.nanpercentile(delta_ssfr[good], [16, 50, 84])
    for p in percentiles:
        ax.axvline(p, color='k', ls=':')
    ax.annotate(f"{percentiles[1]:.2f}+/- ({percentiles[1] - percentiles[0]:.2f}, {percentiles[2] - percentiles[1]:.2f})",
                xy=(.05, .95), xycoords='axes fraction', va='top', ha='left')
    ax.set_xlabel("True - Inferred (dex)")

# %%
scale = list(real_ssfrs.keys())
fig, axs = plt.subplots(3, 7, #constrained_layout=True,
                        figsize=(21, 9), sharex=True, sharey=True)
axs = axs.flatten()
ax_idx = 0
for ith in range(0, 6):
    for jth in range(ith + 1, 7):
        idx = ((ith + 1) * (jth + 1))
        print(ax_idx)
        real_ratio = all_ssfr[jth, 0]- all_ssfr[ith, 0]
        inferred_ratio = all_ssfr[jth, 1]- all_ssfr[ith, 1]

        real_ms = (real_ratio > - 0.3) &  (real_ratio < 0.3)
        real_sb = real_ratio > 0.3
        real_q = real_ratio < -0.3
        
        inferred_ms = (inferred_ratio > - 0.3) &  (inferred_ratio < 0.3)
        inferred_sb = inferred_ratio > 0.3
        inferred_q = inferred_ratio < -0.3
        
        ms_purity = np.count_nonzero(inferred_ms[real_ms]
                                     ) / np.count_nonzero(inferred_ms)
        ms_completeness = np.count_nonzero(real_ms[inferred_ms]
                                                ) / np.count_nonzero(real_ms)

        # sb_purity = np.count_nonzero(inferred_sb[real_sb]
        #                              ) / np.count_nonzero(inferred_sb)
        # sb_completeness = np.count_nonzero(real_sb[inferred_sb]
        #                                         ) / np.count_nonzero(real_sb)

        q_purity = np.count_nonzero(inferred_q[real_q]
                                     ) / np.count_nonzero(inferred_q)
        q_completeness = np.count_nonzero(real_q[inferred_q]
                                                ) / np.count_nonzero(real_q)

        axs[ax_idx].plot(real_ratio, inferred_ratio, ',')
        axs[ax_idx].annotate(f"{scale[jth]}/{scale[ith]}"
                             + f"\nMS(P/C) {ms_purity:.2}/{ms_completeness:.2}"
                             + f"\nQ(P/C) {q_purity:.2}/{q_completeness:.2}",
                             xy=(0.05, 0.95), va='top',
                             xycoords='axes fraction')
        ax_idx += 1
fig.savefig("ssfr_ratios.pdf", bbox_inches='tight')
