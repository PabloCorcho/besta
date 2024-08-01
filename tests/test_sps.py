from hbsps.sfh import FixedGridSFH
import numpy as np
from matplotlib import pyplot as plt

from pst.SSP import PyPopStar

lb = np.array([6, 7, 8, 9, 10])
logz = np.array([-3.0, -2.5, -1.0])

sfh = FixedGridSFH(lb, logz)
sfh.build_free_params()

print(sfh.loglbtime_bins, sfh.loglbtime_bin_edges,
      np.log10(sfh.time))

mass_formed = np.array([1, 1, 1, 1, 1, 1, 1e9, 1e9])

mh, zh = sfh.mass_history(mass_formed)

ssp = PyPopStar(IMF='KRO')
spec, w = ssp.compute_SED(
		    sfh.time[::-1], mh[::-1], zh[::-1])

spec1, _ = ssp.compute_SED(
		    sfh.time[::-1], mh[::-1] / 2,
            np.ones_like(sfh.time) * 10**sfh.logz_bins[0])
spec2, _ = ssp.compute_SED(
		    sfh.time[::-1], mh[::-1] / 2,
            np.ones_like(sfh.time) * 10**sfh.logz_bins[1])
spec_tot = spec1 + spec2

plt.figure()
plt.plot(ssp.wavelength, spec)
plt.plot(ssp.wavelength, spec_tot)
plt.xlim(3000, 9000)
plt.show()
