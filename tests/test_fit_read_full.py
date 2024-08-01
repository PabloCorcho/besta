from glob import glob
from pathlib import Path
import os
import time
from astropy import table
import numpy as np
from matplotlib import pyplot as plt

from hbsps.utils import read_ini_file, read_chain_file, make_plot_chains, compute_chain_percentiles
from hbsps.output import Reader
from hbsps.sfh import reconstruct_sfh_from_table

ref_table = table.Table.read("test/tng/tng_properties_table.fits")

mass_formed_lb_times = np.array([0.01, 0.1, 0.3, 1.0, 3.0, 13.])

output_dir = Path("output/tng_all")
for path in output_dir.glob("*/SFH_auto.ini"):
    gal_id = os.path.basename(os.path.dirname(str(path))).replace(
        "tng_sub_id_", "")
    print("reading: ", path, "Subhalo ID: ", gal_id)
    
    ref_tab_pos = np.where(ref_table['ID'] == str(gal_id))[0][0]
    print(ref_tab_pos)

    # Read the results
    reader = Reader(str(path))
    reader.load_observation()
    reader.load_ssp_model()
    reader.load_extinction_model()
    reader.load_chain()


    results = reconstruct_sfh_from_table(reader.config, reader.table_chain,
                                     av=reader.ini['SFH']['av'])

    w = reader.table_chain['post'].value
    w -= w.max()
    w = np.exp(w / 2)

    mass_history = results['mass_formed_history']
    ages = results['ages']
    ages = (ages[:-1] + ages[1:]) / 2
    median_mass_history = np.nanmedian(mass_history, axis=1)
    pct_mass_history = np.nanpercentile(mass_history, [16, 84], axis=1)
    plt.figure()
    plt.plot(ages, np.cumsum(median_mass_history), 'o-')
    plt.plot(ages, np.cumsum(pct_mass_history[0]), 'o-')
    plt.plot(ages, np.cumsum(pct_mass_history[1]), 'o-')
    plt.plot(np.log10(mass_formed_lb_times) + 9,
             ref_table['MassFormed'][ref_tab_pos], 'o-')
    plt.yscale('log')
    plt.show()

    total_mass = np.log10(ref_table['MassFormed'][ref_tab_pos][-1])
    plt.figure()
    plt.hist(np.log10(results['total_mass']), bins=100, weights=w)
    plt.axvline(total_mass, color='k')
    plt.xlim(total_mass - 0.3, total_mass + 0.3)
    plt.show()