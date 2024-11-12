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
