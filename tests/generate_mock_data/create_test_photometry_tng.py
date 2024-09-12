from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize, ListedColormap

from matplotlib import cm
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from FAENA.math.special_func import schechter, double_power_law
from FAENA.math.stat import get_cmf

from diagram_bins import (color_bin_edges, color_bins, ew_bin_edges, ew_bins,
                          ew_offset, d4000_bin_edges, d4000_bins)

from utils import exp_sequence
from bayes import compute_fraction_from_map
import h5py
from datasets import MaNGA, CALIFA, TNG

ageing_sequence_params, ageing_sequence_params_d4000 = np.loadtxt(
    'data/ageing_sequence_params', unpack=True)
quenched_sequence_params, quenched_sequence_params_d4000 = np.loadtxt(
    'data/quenched_sequence_params', unpack=True)

plt.style.use('plotconfig.mplstyle')
# %%
h5_sfhs = h5py.File(
    'IllustrisTNG-Galaxies/TNG100-1/IllustrisTNG100-1_SFH_linear_bins.hdf5',
    'r')

last_gyr = 3.

log_lookbacktime = h5_sfhs['log_lookbacktime']['lookbacktime_bins'][()]
log_lookbacktime_edges = h5_sfhs['log_lookbacktime']['lookbacktime_edges'][()]
log_lbt_1Gyr = np.digitize(last_gyr, log_lookbacktime)
renorm_log_lbt_1Gyr = np.diff(log_lookbacktime_edges
                              )[:log_lbt_1Gyr].sum() / last_gyr
log_t_univ = (log_lookbacktime_edges.max() - log_lookbacktime) * 1e9


delta_t = 0.001
dummy_lookbacktime_edges = np.arange(log_lookbacktime_edges.min(),
                                     log_lookbacktime_edges.max(),
                                     delta_t)
dummy_lookbacktime = dummy_lookbacktime_edges[:-1] + delta_t/2
dummy_time = dummy_lookbacktime.max() - dummy_lookbacktime
dummy_last_gyr = np.digitize(last_gyr, dummy_lookbacktime)

dummy_ssfrt_edges = np.logspace(-3, 3, 100)
dummy_ssfrt_bins = (
    np.log10(dummy_ssfrt_edges)[:-1] + np.log10(dummy_ssfrt_edges)[1:]) / 2

cross_boundaries = [0, 0.05, 0.5, 2, 1e8]

# Lower limits for SF activity phases
t_strong = 2
t_normal = 0.5
t_mild = 0.05
t_residual = 0

t_quench_idx = np.digitize(t_mild, cross_boundaries) - 1
t_ms_idx = np.digitize(t_normal, cross_boundaries) - 1
t_sb_idx = np.digitize(t_strong, cross_boundaries) - 1

phase_color = ['red', 'yellowgreen', 'cornflowerblue', 'purple']


def interpolate(new_x, x, y, kind='linear', axis=0, fill_value='extrapolate',
                extra_args=dict()):
    """Call scipy.interpolate.interp1d function."""
    interpolator = interp1d(x, y, axis=axis, kind=kind, fill_value=fill_value,
                            **extra_args)
    interpolation = interpolator(new_x)
    del interpolator
    return interpolation


def create_mass_history(ssp_masses):
    """Compute the cumulative mass function from an array of SSP masses."""
    if len(ssp_masses.shape) > 1:
        # Add a 0 before the older SSP
        mass = np.c_[(ssp_masses, np.zeros(ssp_masses.shape[0]))]
        # Compute the cumulative mass (from old to young SSP) and reverse order
        mass_history = np.cumsum(mass[:, ::-1], axis=1)[:, ::-1]
    else:
        mass = np.hstack((ssp_masses, 0))
        mass_history = np.cumsum(mass[::-1])[::-1]
    return mass_history


def compute_ssfr_history(newlookbacktime, lookbacktime, masses,
                         kind='linear', axis=1, log_mass=False):
    """Compute the sfr and ssfr history from the cumulative mass."""
    if not log_mass:
        mass_interp = interpolate(newlookbacktime, lookbacktime, masses,
                                  kind=kind, axis=axis)
        sfr = (mass_interp[:, :-1] - mass_interp[:, 1:]
               ) / np.diff(newlookbacktime)
        mass = (mass_interp[:, :-1] + mass_interp[:, 1:]) / 2
        ssfr = sfr / mass
    else:
        mass_interp = 10**(interpolate(newlookbacktime, lookbacktime,
                                       np.log10(masses + 1),
                                       kind=kind, axis=axis))
        sfr = (mass_interp[:, :-1] - mass_interp[:, 1:]
               ) / np.diff(newlookbacktime)
        mass = (mass_interp[:, :-1] + mass_interp[:, 1:]) / 2
        ssfr = sfr / mass
    return mass, sfr, ssfr


def get_birthrate_times(birthrate_hist, t_bin_width, birthrate_values,
                        kind='linear', interp_args=dict(bounds_error=False)):
    """..."""
    srt_b, cmf_b = get_cmf(birthrate_hist, weights=t_bin_width)
    t_cross = interpolate(birthrate_values, srt_b, cmf_b,
                          axis=0,
                          kind=kind,
                          fill_value=(0, cmf_b[-1]),
                          extra_args=interp_args)
    return srt_b, cmf_b, t_cross


# %%
# =============================================================================
# MANGA data
# =============================================================================
manga = MaNGA()

# %%
log_manga_mass_history = create_mass_history(manga.ssp_masses)
log_manga_mass_history, log_manga_sfr_history, log_manga_ssfr_history = (
    compute_ssfr_history(log_lookbacktime_edges, log_lookbacktime_edges,
                         log_manga_mass_history,
                         kind='linear')
    )

log_manga_birthrate_hist = log_manga_ssfr_history * log_t_univ / 1e9

log_interp_manga_ssfr_history = interpolate(dummy_lookbacktime,
                                            log_lookbacktime,
                                            log_manga_ssfr_history,
                                            axis=1,
                                            kind='linear')

log_interp_manga_birthrate_hist = log_interp_manga_ssfr_history * dummy_time

# %%
# =============================================================================
# CALIFA data
# =============================================================================
califa = CALIFA()
# %%

log_califa_mass_history = create_mass_history(califa.ssp_masses)
log_califa_mass_history, log_califa_sfr_history, log_califa_ssfr_history = (
    compute_ssfr_history(log_lookbacktime_edges, log_lookbacktime_edges,
                         log_califa_mass_history,
                         kind='linear')
    )

log_califa_birthrate_hist = log_califa_ssfr_history * log_t_univ / 1e9

log_interp_califa_ssfr_history = interpolate(dummy_lookbacktime,
                                             log_lookbacktime,
                                             log_califa_ssfr_history,
                                             axis=1,
                                             kind='linear')

log_interp_califa_birthrate_hist = log_interp_califa_ssfr_history * dummy_time
# %%
# TNG
tng = TNG(model='TNG100-1')
# %%
tng_cum_T = np.zeros((dummy_ssfrt_edges.size))
tng_cum_T_1Gyr = np.zeros((dummy_ssfrt_edges.size))
tng_bound_times = np.full(
    (tng.subhaloid.size, len(cross_boundaries)), fill_value=np.nan)
tng_bound_times_1Gyr = np.full(
    (tng.subhaloid.size, len(cross_boundaries)), fill_value=np.nan)

tng_cum_T_minres = np.zeros((dummy_ssfrt_edges.size))
tng_cum_T_1Gyr_minres = np.zeros((dummy_ssfrt_edges.size))
tng_bound_times_minres = np.full(
    (tng.subhaloid.size, len(cross_boundaries)), fill_value=np.nan)
tng_bound_times_1Gyr_minres = np.full(
    (tng.subhaloid.size, len(cross_boundaries)), fill_value=np.nan)
# MaNGA
manga_cum_T = np.zeros_like(dummy_ssfrt_edges)
manga_cum_T_1Gyr = np.zeros_like(dummy_ssfrt_edges)
manga_bound_times = np.full(
    (log_manga_ssfr_history.shape[0], len(cross_boundaries)), fill_value=np.nan)
manga_bound_times_1Gyr = np.full(
    (log_manga_ssfr_history.shape[0], len(cross_boundaries)), fill_value=np.nan)
manga_bound_times_1Gyr_lin = np.full(
    (log_manga_ssfr_history.shape[0], len(cross_boundaries)), fill_value=np.nan)
# CALIFA
califa_cum_T = np.zeros_like(dummy_ssfrt_edges)
califa_cum_T_1Gyr = np.zeros_like(dummy_ssfrt_edges)
califa_bound_times = np.full(
    (log_califa_ssfr_history.shape[0], len(cross_boundaries)), fill_value=np.nan)
califa_bound_times_1Gyr = np.full(
    (log_califa_ssfr_history.shape[0], len(cross_boundaries)), fill_value=np.nan)

# %% LOOPS
for i, birthrate_hist in enumerate(log_interp_califa_birthrate_hist):
    print('Iter {}, CALIFA'.format(i))
    br_sort, cmf_br, cross_times = get_birthrate_times(
        birthrate_hist,
        t_bin_width=np.ones_like(birthrate_hist) * delta_t,
        birthrate_values=cross_boundaries)

    califa_bound_times[i] = cross_times

    # Only the SFH during the last Gyr
    br_sort_1Gyr, cmf_br_1Gyr, cross_times_1Gyr = get_birthrate_times(
        birthrate_hist[:dummy_last_gyr],
        t_bin_width=np.ones_like(birthrate_hist[:dummy_last_gyr]) * delta_t,
        birthrate_values=cross_boundaries)

    califa_bound_times_1Gyr[i] = cross_times_1Gyr

for i, birthrate_hist in enumerate(log_interp_manga_birthrate_hist):
    print('Iter {}, MaNGA'.format(i))
    br_sort, cmf_br, cross_times = get_birthrate_times(
        birthrate_hist,
        t_bin_width=np.ones_like(birthrate_hist) * delta_t,
        birthrate_values=cross_boundaries)

