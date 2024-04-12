import numpy as np
import matplotlib.pyplot as plt
import os

def read_ini_file(path):
    ini_info = {}
    with open(path, "r") as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            if len(line) == 0:
                continue
            if line[0] == '[':
                module = line.strip("[]")
                ini_info[module] = {}
            else:
                components = line.split("=")
                name = components[0].strip(" ")
                str_value = components[1].replace(" ", "")
                str_value = str_value.replace(".", "")
                str_value = str_value.replace("e", "")
                str_value = str_value.replace("-", "")
                if not str_value.isnumeric():
                    ini_info[module][name] = components[1].strip(" ")
                else:
                    numbers = [n for n in components[1].split(" ") if len(n) > 0]
                    if ("." in components[1]) or ("e" in components[1]):
                        # Float number
                        ini_info[module][name] = np.array(
                            numbers, dtype=float)
                    else:
                        # int number
                        ini_info[module][name] = np.array(
                            numbers, dtype=int)
    return ini_info

def read_chain_file(path):
    with open(path, "r") as f:
        header = f.readline().strip("#")
        columns = header.replace("\n", "").split("\t")
    matrix = np.loadtxt(path)
    results = {}
    for ith, par in enumerate(columns):
        results[par] = matrix[:, ith]
    return results

def make_plot_chains(chain_results, truth_values=None, output='.'):
    parameters = [par for par in chain_results.keys() if "parameters" in par]
    if truth_values is None:
        truth_values = [np.nan] * len(parameters)
    all_figs = []
    for par, truth in zip(parameters, truth_values):
        fig = plt.figure(constrained_layout=True)
        ax = fig.add_subplot(111)
        ax.plot(chain_results[par], ',', c='k')
        ax.axhline(truth, c='r')
        inax = ax.inset_axes((1.05, 0, .5, 1))
        inax.hist(chain_results[par], weights=chain_results['weight'],
                  bins=100)
        inax.axvline(truth, c='r')
        plt.show()
        all_figs.append(fig)
    return all_figs

def compute_chain_percentiles(chain_results, pct=[.5, .16, .50, .84, .95]):
    parameters = [par for par in chain_results.keys() if "parameters" in par]
    pct_resutls = {}
    for par in parameters:
        sort_pos = np.argsort(chain_results[par])
        cum_distrib = np.cumsum(chain_results['weight'][sort_pos])
        cum_distrib /= cum_distrib[-1]
        pct_resutls[par] = np.interp(
            pct, cum_distrib, chain_results[par][sort_pos])
    return pct_resutls
