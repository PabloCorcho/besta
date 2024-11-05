import numpy as np
import matplotlib.pyplot as plt
import os

from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table

cosmology = FlatLambdaCDM(H0=70., Om0=0.28)

def read_chain_file(path):
    with open(path, "r") as f:
        header = f.readline().strip("#")
        columns = header.replace("\n", "").split("\t")
    matrix = np.loadtxt(path)
    results = {}
    ssp_weights = np.zeros(matrix.shape[0])
    last_ssp = 0
    for ith, par in enumerate(columns):
        results[par] = matrix[:, ith]
        if "ssp" in par:
            last_ssp += 1
            ssp_weights += 10 ** matrix[:, ith]
    results[f"parameters-ssp{last_ssp + 1}"] = np.log10(
        np.clip(1 - ssp_weights, a_min=1e-4, a_max=None)
    )
    return results


def make_plot_chains(chain_results, truth_values=None, output="."):
    parameters = [par for par in chain_results.keys() if "parameters" in par]
    if truth_values is None:
        truth_values = [np.nan] * len(parameters)
    all_figs = []
    for par, truth in zip(parameters, truth_values):
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.plot(chain_results[par], ",", c="k")
        ax.axhline(truth, c="r")
        inax = ax.inset_axes((1.05, 0, 0.5, 1))
        inax.hist(chain_results[par], weights=chain_results["weight"], bins=100)
        inax.axvline(truth, c="r")
        plt.show()
        all_figs.append(fig)
    return all_figs


def compute_chain_percentiles(chain_results, pct=[0.5, 0.16, 0.50, 0.84, 0.95]):
    parameters = [par for par in chain_results.keys() if "parameters" in par]
    pct_resutls = {}
    for par in parameters:
        sort_pos = np.argsort(chain_results[par])
        cum_distrib = np.cumsum(chain_results["weight"][sort_pos])
        cum_distrib /= cum_distrib[-1]
        pct_resutls[par] = np.interp(pct, cum_distrib, chain_results[par][sort_pos])
    return pct_resutls
