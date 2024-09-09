import os
from pst import dust, SSP, models, observables
import numpy as np
from matplotlib import pyplot as plt

from astropy import constants
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.io import fits

cosmology = FlatLambdaCDM(H0=70., Om0=0.28)

basedir = "../test_data/photometry/illustris_dust_and_redshift/subhalo_167396_photometry.dat"

fluxes, fluxe_err = np.loadtxt(basedir, usecols=(1, 2), unpack=True)
filters = np.loadtxt(basedir, usecols=(0), unpack=True, dtype=str)

photometric_filters = []
for filter_name in filters:
    print(f"Loading photometric filter: {filter_name}")
    if os.path.exists(filter_name):
        f = observables.Filter(filter_path=filter_name)
    else:
        f = observables.Filter(filter_name=filter_name)
    photometric_filters.append(f)

ssp = SSP.PyPopStar("KRO")
velscale = 200.
oversampling = 1

dlnlam = velscale / constants.c.to("km/s").value
dlnlam /= oversampling
ln_wl_edges = np.log(ssp.wavelength[[0, -1]].to_value("angstrom"))
extra_offset_pixel = 0

lnlam_bin_edges = np.arange(
    ln_wl_edges[0]
    - dlnlam * extra_offset_pixel * oversampling
    - 0.5 * dlnlam,
    ln_wl_edges[-1]
    + dlnlam * (1 + extra_offset_pixel) * oversampling
    + 0.5 * dlnlam,
    dlnlam,
)
ssp.interpolate_sed(np.exp(lnlam_bin_edges))


dust_model = dust.DustScreen("ccm89")
a_v_array = np.linspace(0, 3, 30)
ssps = [dust_model.redden_ssp_model(ssp, a_v=av) for av in a_v_array]
all_photometry = np.zeros(
		(a_v_array.size, len(photometric_filters), *ssp.L_lambda.shape[:-1])
							 ) * u.Quantity("3631e-9 Jy / Msun")

for j, ssp in enumerate(ssps):
    photo = ssp.compute_photometry(
				filter_list=photometric_filters, z_obs=0
				).to("3631e-9 Jy / Msun")
    all_photometry[j] = photo


class SFHBase():
    free_params = {}
    def __init__(self, *args, **kwargs):
        print("Initialising Star Formation History model")
        
        self.redshift = kwargs.get("redshift", 0.0)
        self.today = kwargs.get(
            "today", cosmology.age(self.redshift))

    def make_ini(self, ini_file):
        print("Making ini file: ", ini_file)
        with open(ini_file, "w") as file:
            file.write("[parameters]\n")
            for k, v in self.free_params.items():
                if len(v) > 1:
                    file.write(f"{k} = {v[0]} {v[1]} {v[2]}\n")
                else:
                    file.write(f"{k} = {v[0]}\n")


class FixedTime_sSFR_SFH(SFHBase):
    free_params = {'alpha': [0, 1, 10], 'z_today': [0.005, 0.01, 0.08]}

    def __init__(self, delta_time, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[SFH] Initialising FixedGrid-sSFR-SFH model")
        if not isinstance(delta_time, u.Quantity):
            print("Assuming input times are in Gyr: ", delta_time)
            delta_time = np.array(delta_time) * u.Gyr
        self.delta_time = np.sort(delta_time)[::-1]
        self.today = cosmology.age(self.redshift)

        self.time = np.sort(self.today - self.delta_time)
        self.time = np.insert(self.time, [0, self.time.size],
                              [0 * u.Gyr, self.today])
        for dt in self.delta_time.to_value('yr'):
            max_logssfr = np.min((np.log10(1 / dt), -8.0))
            self.free_params[
                    f'logssfr_over_{np.log10(dt):.2f}_yr'] = [-14, -10, max_logssfr]

        self.model = models.Tabular_ZPowerLaw(
            times=self.time, masses=np.ones(self.time.size) * u.Msun,
            z_today=kwargs.get("z_today", 0.02)  * u.dimensionless_unscaled,
            alpha=kwargs.get("alpha", 0.0))

    def parse_free_params(self, free_params):
        dt_yr = self.delta_time.to_value('yr')
        ssfr_over_last = np.array(
            [free_params[f'logssfr_over_{np.log10(dt):.2f}_yr'] for dt in dt_yr],
            dtype=float)
        mass_frac = 1 - dt_yr * 10**ssfr_over_last
        mass_frac = np.insert(mass_frac, [0, mass_frac.size], [0.0, 1.0])
        if (mass_frac[1:] - mass_frac[:-1] < 0).any():
            return 0
        # Update the mass of the tabular model
        self.model.table_M =  mass_frac * u.Msun
        self.model.alpha = free_params['alpha']
        self.model.z_today = free_params['z_today'] * u.dimensionless_unscaled
        return 1

    def parse_datablock(self, datablock):
        dt_yr = self.delta_time.to_value('yr')
        ssfr_over_last = np.array(
            [datablock["parameters", f'logssfr_over_{np.log10(dt):.2f}_yr'] for dt in dt_yr],
            dtype=float)
        mass_frac = 1 - dt_yr * 10**ssfr_over_last
        mass_frac = np.insert(mass_frac, [0, mass_frac.size], [0.0, 1.0])
        if (mass_frac[1:] - mass_frac[:-1] < 0).any():
            return 0
        # Update the mass of the tabular model
        self.model.table_M =  mass_frac * u.Msun
        self.model.alpha = datablock["parameters",'alpha']
        self.model.z_today = datablock["parameters",'z_today'] * u.dimensionless_unscaled
        return 1

sfh = FixedTime_sSFR_SFH([0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0])

# best_fit_values = {
#     'alpha':0.5532074960831881,
#     'z_today': 0.012220741977759825,
#     'logssfr_over_10.00_yr':-10.02738336039512,
#     'logssfr_over_9.70_yr': -10.982550237601957,
#     'logssfr_over_9.48_yr': -10.765533342991736,
#     'logssfr_over_9.00_yr': -10.288442258336438,
#     'logssfr_over_8.70_yr':	-9.98741226840214,
#     'logssfr_over_8.48_yr': -10.479772533935853,
# 	'logssfr_over_8.00_yr': -10.002671435215348
#     }
# normalization = 59018793731.38986


# from the medians
best_fit_values = {
    'alpha':1.3702906134536736,
    'z_today': 0.015567951944085825,
    'logssfr_over_10.00_yr':-10.229540951393194,
    'logssfr_over_9.70_yr': -10.140589318260336,
    'logssfr_over_9.48_yr': -10.313681756932677,
    'logssfr_over_9.00_yr': -10.124007285476837,
    'logssfr_over_8.70_yr':	-10.11625982521806,
    'logssfr_over_8.48_yr': -10.285853338186678,
	'logssfr_over_8.00_yr': -10.181167652188803
    }
normalization = 53391966268.356865

sfh.parse_free_params(best_fit_values)

# =============================================================================
# Real values
# =============================================================================
# ssfr_0.1=9.31e-11
# ssfr_0.3=9.05e-11
# ssfr_0.5=8.90e-11
# ssfr_1.0=8.46e-11
# ssfr_3.0=1.06e-10
# ssfr_5.0=9.60e-11
# ssfr_10.0=9.77e-11


flux_model = sfh.model.compute_photometry(
		ssp,
		t_obs=sfh.today,
		allow_negative=False,
		photometry=all_photometry[0])
flux_model = flux_model.to_value("3631e-9 Jy")

normalization = np.mean(fluxes / flux_model)

plt.figure()
plt.plot(fluxes)
plt.plot(flux_model * normalization)

chi2 = (fluxes - flux_model * normalization)**2 / fluxe_err**2
print("Mean chi2: ", np.mean(chi2))
plt.figure()
plt.subplot(121)
plt.plot(sfh.time, np.log10(sfh.model.Z))
plt.subplot(122)
plt.plot(sfh.time, sfh.model.table_M.value * normalization)


# %%=============================================================================
# # Results from manual processing
# =============================================================================
results = fits.open("/home/pcorchoc/Develop/HBSPS/output/photometry/illustris_dust_and_redshift/subhalo_167396/processed_results.fits")
manual_best_fit = best_fit_values.copy()

for key in manual_best_fit.keys():
    manual_best_fit[key] = results[2].data[f'{key}_pct'][2]

normalization = results[2].data['normalization_pct'][2]

sfh.parse_free_params(manual_best_fit)

flux_model = sfh.model.compute_photometry(
		ssp,
		t_obs=sfh.today,
		allow_negative=False,
		photometry=all_photometry[0])
flux_model = flux_model.to_value("3631e-9 Jy")

normalization = np.mean(fluxes / flux_model)

plt.figure()
plt.errorbar(x=np.arange(0, fluxes.size), y=fluxes, yerr=fluxe_err,
             lw=0, marker='.', capsize=2)
plt.plot(flux_model * normalization)

chi2 = (fluxes - flux_model * normalization)**2 / fluxe_err**2
print("Mean chi2: ", np.mean(chi2))
plt.figure()
plt.subplot(121)
plt.plot(sfh.time, np.log10(sfh.model.Z + 1e-4))
plt.subplot(122)
plt.plot(sfh.time, sfh.model.table_M.value * normalization)


