import numpy as np
import hbsps.specBasics as specBasics
import matplotlib.pyplot as plt

from pst.utils import flux_conserving_interpolation
from pst.SSP import BaseGM

from hbsps.utils import read_ini_file, read_chain_file, make_plot_chains, compute_chain_percentiles
from io import Reader

#ini_file = read_ini_file("HBSPS_SFHdelayedtau.ini")

reader = Reader("HBSPS_SFH_tng.ini")

pct_results = reader.get_chain_percentiles()
reader.compute_solution_from_pct(pct_results)
