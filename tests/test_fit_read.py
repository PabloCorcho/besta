from glob import glob
from pathlib import Path
import os
import time
from astropy import table
import numpy as np
from matplotlib import pyplot as plt

ref_table = table.Table.read("test/tng/tng_properties_table.fits")

results = np.array([
    [0, 112.3, 70.00, 2.845],
    [86, -131.69, 50.0, 0.43],
    [86, -131.69, 50.0, 0.43],
    [128393, 106., 58.077, 0.92],
    [172727, -306.90, 143.75, 1.173],
    [194960, 36.887, 205.28, 1.3189],
    [342354, 63.627, 100.27, 2.17],
    [409086, 146.78438, 50.0, 0.897],
    [421032, 253.829, 210.00, 2.1324],
    [459084, 75.680, 80.125, 2.8141],
    [476624, 51.6710, 138.448, 2.544],
    [560767, 82.2265, 207.270, 0.0341],
    [569518, -20.579, 192.3176, 0.24637],
    [577764, -84.0053, 156.701, 2.2471],
    [585537, 20.014, 144.2742, 0.3247],
    #[588577, 254.7396, 252.0112, 0.10814],
    [592690, -68.1495, 57.2775, 2.4952],
    [599569, -146.893, 54.7234, 1.76797],
])


all_offsets = []
for gal in results:
    gal_id = str(int(gal[0]))
    pos = np.where(ref_table['ID'] == gal_id)[0][0]
    print(gal_id, pos)
    #print(ref_table[pos])
    all_offsets.append(
        [
        ref_table[pos]['los_vel'] - gal[1],
        ref_table[pos]['los_sigma'] - gal[2],
        ref_table[pos]['av'] - gal[3]]
    )

all_offsets = np.array(all_offsets)
fig, axs = plt.subplots(nrows=2, ncols=2)
axs[0, 1].axis('off')
axs[0, 0].hist(all_offsets[:, 0], bins='auto')
axs[0, 0].set_xlabel(r"$v$ offset [km/s]")
axs[1, 0].hist(all_offsets[:, 1], bins='auto')
axs[1, 0].set_xlabel(r"$\sigma$ offset [km/s]")
axs[1, 1].hist(all_offsets[:, 2], bins='auto')
axs[1, 1].set_xlabel(r"$A_v$ offset [dex]")

plt.show()

fig, axs = plt.subplots(nrows=2, ncols=2)
axs[0, 1].axis('off')
axs[0, 0].plot(results[:, 1], all_offsets[:, 0], 'o')
axs[0, 0].set_ylabel(r"$v$ offset [km/s]")
axs[1, 0].plot(results[:, 2], all_offsets[:, 1], 'o')
axs[1, 0].set_ylabel(r"$\sigma$ offset [km/s]")
axs[1, 1].plot(results[:, 3], all_offsets[:, 2], 'o')
axs[1, 1].set_ylabel(r"$A_v$ offset [dex]")
plt.show()


# results = Path("/output/tng_all")

# results.glob("**/SFH_auto.ini")