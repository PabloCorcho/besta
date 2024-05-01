import numpy as np
import matplotlib.pyplot as plt

from hbsps.utils import read_ini_file, read_chain_file, make_plot_chains, compute_chain_percentiles
from hbsps.output import Reader
from hbsps.sfh import reconstruct_sfh_from_table

reader = Reader("SFH_auto.ini")

reader.load_observation()
reader.load_ssp_model()
reader.load_extinction_model()
reader.load_chain()

results = reconstruct_sfh_from_table(reader.config, reader.table_chain)

w = reader.table_chain['post'].value
w -= w.max()
w = np.exp(w / 2)

plt.figure()
plt.hist(np.log10(results['total_mass']), bins=100, weights=w)
plt.show()