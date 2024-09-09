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

output_dir = '../test_data/photometry/illustris_examples'
results_dir = '../output/photometry/illustris'

files = glob(os.path.join(output_dir, "*_photometry.dat"))

galaxies = {}

fig, axs = plt.subplots(1, 7, constrained_layout=True,
                        figsize=(15,  5))

for ith, ax in enumerate(axs):
    ax.plot([-14, -8], [-14, -8], 'k')
    ax.grid(visible=True)
    ax.set_box_aspect(1)
    ax.set_xlabel("Real sSFR")
    ax.set_ylabel("Inferred sSFR")

all_ssfr = np.zeros((7, 2, len(files)))

for nfile, file in enumerate(files):
    print(file)
    gid = os.path.basename(file).strip("_photometry.dat")
    real_ssfrs = {}
    with open(file, "r") as f:
        while True:
            line = f.readline()
            if "ssfr" in line:
                line = line.strip("#").strip("\n").replace(" ", "")
                
                k, v = line.split("=")
                real_ssfrs[k.strip("ssfr_")] = float(v)
            if "CFHT" in line:
                break
    
    galaxies[gid] = real_ssfrs
    
    ssfrs = np.array(list(real_ssfrs.values()))
    print(ssfrs)
    # Read results
    result = os.path.join(results_dir, f"{gid}", "medians.txt")
    result_ssfrs = np.loadtxt(result, usecols=1)[2:-3]
    result_ssfrs_err = np.loadtxt(result, usecols=2)[2:-3]
    
    for ith, ax in enumerate(axs):
        ax.errorbar(np.log10(ssfrs[ith] + 1e-14), result_ssfrs[ith],
                    yerr=result_ssfrs_err[ith], marker='.')
        all_ssfr[ith, :, nfile] = [np.log10(ssfrs[ith] + 1e-14), result_ssfrs[ith]]

all_ssfr = np.array(all_ssfr)
for scale, ax in zip(real_ssfrs, axs):
    ax.set_title(f"tau / Gyr = {scale}")
fig.savefig("illustris_ssfr_comparison.png", dpi=250, bbox_inches='tight')

# %%
fig, axs = plt.subplots(1, 7, constrained_layout=True,
                        figsize=(15,  5))

for ith, (scale, ax) in enumerate(zip(real_ssfrs.keys(), axs)):
    ax.set_title(f"tau / Gyr = {scale}")
    ax.set_box_aspect(1)
    delta_ssfr = all_ssfr[ith, 0] - all_ssfr[ith, 1]
    h, xedges, _ = ax.hist(delta_ssfr, bins=30)
    percentiles = np.nanpercentile(delta_ssfr, [16, 50, 84])
    for p in percentiles:
        ax.axvline(p, color='k', ls=':')
    ax.annotate(f"{percentiles[1]:.2f}+/- ({percentiles[1] - percentiles[0]:.2f}, {percentiles[2] - percentiles[1]:.2f})",
                xy=(.05, .95), xycoords='axes fraction', va='top', ha='left')
    ax.set_xlabel("True - Inferred (dex)")

fig.savefig("illustris_ssfr_hist_comparison.png", dpi=250, bbox_inches='tight')
plt.show()
