#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 15:18:00 2024

@author: pcorchoc
"""
import numpy as np
import os
from astropy.table import Table
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

from FAENA.models import ageing_diagram

LKEY = "parameters--logssfr_over_9.00_yr"
SKEY = "parameters--logssfr_over_8.00_yr"

ssfr_limits = ageing_diagram.ssfr_activity_levels()

def get_ad_probability(ssfr_long, ssfr_short, weights,
                       ssfr_binedges=np.arange(-14, -8, 0.1),
                       gauss_smooth=1, plot=False):

    
    ssfr_bins = (ssfr_binedges[:-1] + ssfr_binedges[1:]) / 2
    H, _, _ = np.histogram2d(ssfr_short, ssfr_long, weights=weights,
                             bins=ssfr_binedges, density=True)
    if gauss_smooth is not None:
        H = gaussian_filter(H, sigma=gauss_smooth)
        H /= np.nansum(H * np.diff(ssfr_binedges)[np.newaxis]**2)

    H_cum = np.cumsum(
        np.cumsum(H * np.diff(ssfr_binedges)[np.newaxis], axis=0)
        *np.diff(ssfr_binedges)[np.newaxis], axis=1)
    interpolator = RegularGridInterpolator((ssfr_bins, ssfr_bins), H_cum)


    ssfr_l_pts, ssfr_s_pts = np.meshgrid(
        np.insert(ssfr_limits, [0, ssfr_limits.size], [ssfr_bins[[0, -1]]]),
        np.insert(ssfr_limits, [0, ssfr_limits.size], [ssfr_bins[[0, -1]]]),
        indexing='ij')

    cmf_pts = interpolator(np.array([ssfr_l_pts.flatten(),
                                     ssfr_s_pts.flatten()]).T)
    cmf_pts = cmf_pts.reshape(ssfr_l_pts.shape)

    p_retired = cmf_pts[1, 1] - cmf_pts[0, 0]

    p_trans_trans = (cmf_pts[2, 2] - cmf_pts[1, 1]
                     - (cmf_pts[1, 2] - cmf_pts[1, 1])
                     - (cmf_pts[2, 1] - cmf_pts[1, 1]))

    p_ms_ms = (cmf_pts[3, 3] - cmf_pts[2, 2]
                     - (cmf_pts[3, 2] - cmf_pts[2, 2])
                     - (cmf_pts[2, 3] - cmf_pts[2, 2]))
    p_ageing = p_trans_trans + p_ms_ms
    p_quenched = (
        cmf_pts[0, -1] - cmf_pts[0, 0]
        + cmf_pts[1, -1] - cmf_pts[1, 1]
        )
    p_rejuv = (
        cmf_pts[-1, 2]
        - cmf_pts[1, 1]
        - (cmf_pts[2, 2] - cmf_pts[2, 1])
        )
    
    if plot:
        fig, ax = plt.subplots()
        mappable  =ax.pcolormesh(ssfr_bins, ssfr_bins, H_cum, cmap="Greys")
        plt.colorbar(mappable, ax=ax, label='Cum prob.')
        ax.contour(ssfr_bins, ssfr_bins, H, cmap='jet')
        for limit, color in zip(ssfr_limits, ["r", "g", "b"]):
            ax.axvline(limit, color=color)
            ax.axhline(limit, color=color)
        ax.plot([-14, -8], [-14, -8], color='lime')
        ax.plot([-14, -8], [-13.3, -7.3], color='lime')
        ax.plot([-14, -8], [-14.7, -8.7], color='lime')
    else:
        fig, ax = None, None
    
    return {'p_ageing': p_ageing,
            'p_retired': p_retired,
            'p_quenched': p_quenched,
            'p_rejuv': p_rejuv}, (fig, ax)

def get_delta_ssfr_probability(ssfr_long, ssfr_short, weights,
                       ssfr_binedges=np.arange(-14, -8, 0.1),
                       delta_ssfr_binedges = np.arange(-3, 3, .1),
                       gauss_smooth=1, plot=False):

    ssfr_bins = (ssfr_binedges[:-1] + ssfr_binedges[1:]) / 2
    delta_ssfr_bins = (delta_ssfr_binedges[:-1] + delta_ssfr_binedges[1:]) / 2
    
    H, _, _ = np.histogram2d(ssfr_short - ssfr_long,
                             ssfr_long, weights=weights,
                             bins=[delta_ssfr_binedges, ssfr_binedges], density=True)
    if gauss_smooth is not None:
        H = gaussian_filter(H, sigma=gauss_smooth)
        H /= np.nansum(H
                       * np.diff(delta_ssfr_binedges)[:, np.newaxis]
                       * np.diff(ssfr_binedges)[np.newaxis])

    H_cum = np.cumsum(
        np.cumsum(H * np.diff(delta_ssfr_binedges)[:, np.newaxis], axis=0)
        *np.diff(ssfr_binedges)[np.newaxis], axis=1)

    interpolator = RegularGridInterpolator((delta_ssfr_bins, ssfr_bins), H_cum)

    ssfr_vertex = np.insert(ssfr_limits, [0, ssfr_limits.size], [ssfr_bins[[0, -1]]])
    delta_ssfr_vertex = np.array([delta_ssfr_bins[0], -0.7, 0.7, delta_ssfr_bins[-1]])

    delta_ssfr_pts, ssfr_l_pts = np.meshgrid(delta_ssfr_vertex, ssfr_vertex,
                                         indexing='ij')

    cmf_pts = interpolator(np.array([delta_ssfr_pts.flatten(), ssfr_l_pts.flatten()]).T)
    cmf_pts = cmf_pts.reshape(ssfr_l_pts.shape)

    p_below = cmf_pts[1, -1]
    p_above = cmf_pts[-1, -1] - cmf_pts[2, -1]
    p_between = 1 - p_below - p_above
    
    if plot:
        fig, ax = plt.subplots()
        mappable = ax.pcolormesh(ssfr_bins, delta_ssfr_bins, H_cum, cmap="Greys")
        plt.colorbar(mappable, ax=ax)
        ax.contour(ssfr_bins, delta_ssfr_bins, H, cmap='jet')
        ax.axhline(0, color='r')
        ax.axhline(0.7, color='r')
        ax.axhline(-0.7, color='r')
    else:
        fig, ax = None, None

    return {"p_below": p_below, 
            "p_above": p_above,
            "p_between": p_between}, (fig, ax)

def read_results_file(path):
    with open(path, "r") as f:
        header = f.readline().strip("#")
        columns = header.replace("\n", "").split("\t")
    matrix = np.loadtxt(path)
    
    table = Table()
    for ith, c in enumerate(columns):
        table.add_column(matrix.T[ith], name=c.lower())
    return table

def get_all_from_results_file(path, ad_kwargs={}, ssfr_kwargs={}):
    table = read_results_file(path)
    posterior = np.exp(
            table["post"].value - np.nanmax(table["post"].value)) 
    posterior /= np.nansum(posterior)
    
    ssfr_binedges = np.arange(-14, -8, 0.1)
    ssfr_bins = (ssfr_binedges[:-1] + ssfr_binedges[1:]) / 2
    
    delta_ssfr_binedges = np.arange(-3, 3, .1)
    delta_ssfr_bins = (delta_ssfr_binedges[:-1] + delta_ssfr_binedges[1:]) / 2
    
    p_ad, (ad_fig, ad_ax) = get_ad_probability(
        table[LKEY].value, table[SKEY].value, weights=posterior, **ad_kwargs)
    
    p_ssfr, (ssfr_fig, ssfr_ax) = get_delta_ssfr_probability(
        table[LKEY].value, table[SKEY].value, weights=posterior, **ssfr_kwargs)
    return p_ad, p_ssfr, (ad_fig, ad_ax, ssfr_fig, ssfr_ax)
    
    
if __name__ == "__main__":
    p_ad, p_ssfr, figures = get_all_from_results_file("../output/photometry/illustris_dust_and_redshift/subhalo_63926/SFH_results.txt")
