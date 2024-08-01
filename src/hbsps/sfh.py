import numpy as np
from hbsps.utils import cosmology
import pst
from astropy import units as u
from scipy.special import erf, hyp2f1, betainc

def composite_stellar_population(config, weights):
    if type(weights) is dict:
        w_array = np.array([weights[w] for w in weights.keys() if "ssp" in w])
    else:
        w_array = weights
    w_array = 10**w_array
    composite_spectra = np.sum(config["ssp_sed"] * w_array[:, np.newaxis], axis=0)
    return composite_spectra

def reconstruct_sfh(config, weights, av=None):
    print("Reconstructing SFH from input solution")
    if type(weights) is dict:
        w_array = np.array([weights[w] for w in weights.keys() if "ssp" in w])
    else:
        w_array = np.asarray(weights)
    w_array = 10**w_array

    ssp_met, ssp_age = config['ssp_metals_edges'], config['ssp_ages_edges']
    # Geometric mean
    ssp_met_bins = (ssp_met[:-1] + ssp_met[1:]) / 2
    ssp_age_bins = (ssp_age[:-1] + ssp_age[1:]) / 2
    ssp_mlr = config['ssp_mlr']
    norm_flux = config['norm_flux']
    if av is not None:
    # Deredden the observed flux norm
       extinction_law = config["extinction_law"]
       ext = extinction_law.extinction(extinction_law.norm_wave, av)
       norm_flux *= 1 / ext

    w_array = w_array.reshape(ssp_mlr.shape)
    print("Light fraction matrix shape: ", w_array.shape)
    ssp_mass_formed = norm_flux * w_array * ssp_mlr
    mass_history = np.nansum(ssp_mass_formed, axis=0)
    total_mass = np.nansum(mass_history)
    mean_age = 10**(np.sum(ssp_mass_formed * ssp_age_bins[np.newaxis, :]
                      ) / total_mass)
    mean_metals = 10**(np.sum(ssp_mass_formed * ssp_met_bins[:, np.newaxis]
                         ) / total_mass)
    results = {'ages': ssp_age, 'metals': ssp_met,
               'ssp_mass_formed': ssp_mass_formed,
               'mass_formed_history': mass_history, 'total_mass': total_mass,
               'mean_age': mean_age, 'mean_metals': mean_metals}
    return results

def reconstruct_sfh_from_table(config, table, av=None):
    print("Reconstructing SFH from input solution")
    weights = 10**np.array([table[k].value for k in table.keys() if "ssp" in k])

    ssp_met, ssp_age = config['ssp_metals_edges'], config['ssp_ages_edges']
    # Geometric mean
    ssp_mlr = config['ssp_mlr']
    norm_flux = config['norm_flux']
    if av is not None:
    # Deredden the observed flux norm
       extinction_law = config["extinction_law"]
       ext = extinction_law.extinction(extinction_law.norm_wave, av)
       norm_flux *= 1 / ext

    weights = weights.reshape((*ssp_mlr.shape, weights.shape[-1]))

    print("Light fraction matrix shape: ", weights.shape)
    ssp_mass_formed = norm_flux * weights * ssp_mlr[:, :, np.newaxis]
    mass_history = np.nansum(ssp_mass_formed, axis=0)
    total_mass = np.nansum(mass_history, axis=0)
    # mean_age = 10**(np.sum(ssp_mass_formed * ssp_age_bins[np.newaxis, :]
    #                   ) / total_mass)
    # mean_metals = 10**(np.sum(ssp_mass_formed * ssp_met_bins[:, np.newaxis]
    #                      ) / total_mass)
    results = {'ages': ssp_age, 'metals': ssp_met,
               'ssp_mass_formed': ssp_mass_formed,
               'mass_formed_history': mass_history, 'total_mass': total_mass,
                #'mean_age': mean_age, 'mean_metals': mean_metals
               }
    return results

# Star formation history models

class SFHBase():
    free_params = {}
    def __init__(self, *args, **kwargs):
        print("Initialising Star Formation History model")
        
        self.today = kwargs.get(
            "today", cosmology.age(kwargs.get("redshift", 0.0)))

    def make_ini(self, ini_file):
        print("Making ini file: ", ini_file)
        with open(ini_file, "w") as file:
            file.write("[parameters]\n")
            for k, v in self.free_params.items():
                file.write(f"{k} = {v[0]} {v[1]} {v[2]}\n")

class FixedTimeSFH(SFHBase):
    free_params = {'alpha': [0, 1, 10], 'z_today': [0.005, 0.01, 0.08]}

    def __init__(self, time_bins, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[SFH] Initialising FixedGridSFH model")
        self.time = np.sort(time_bins)
        if not isinstance(self.time, u.Quantity):
            self.time = self.time * u.Gyr
        self.today = cosmology.age(0)

        if self.time[-1] > self.today:
            self.today = self.time[-1] + 1 * u.yr

        self.time = np.insert(
            self.time, (0, self.time.size),
            (0 * u.Gyr, self.today.to(self.time.unit)))

        delta_t = self.time[1:] - self.time[:-1]
        for t, dt in zip(self.time[1:-1].to_value('Gyr'), delta_t[:-1]):
            max_frac = dt.to_value('Gyr') / self.today.to_value('Gyr')
            self.free_params[
                    f'logmass_at_{t:.3f}'] = [0.0, max_frac / 2, min((2 * max_frac, 1))]
        

        self.model = pst.models.Tabular_ZPowerLaw(
            times=self.time, masses=np.ones(self.time.size) * u.Msun,
            z_today=kwargs.get("z_today", 0.02)  * u.dimensionless_unscaled,
            alpha=kwargs.get("alpha", 0.0))

    def parse_free_params(self, free_params):
        m_formed = np.array(
            [free_params[f'logmass_at_{t:.3f}'] for t in self.time[1:-1].to_value('Gyr')],
            dtype=float)
        cumulative = np.cumsum(m_formed)
        if cumulative[-1] > 1.0:
            return 0
        cumulative = np.insert(cumulative, (0, cumulative.size), (0, 1))
        # Update the mass of the tabular model
        self.model.table_M =  cumulative * u.Msun
        self.model.alpha = free_params['alpha']
        self.model.z_today = free_params['z_today'] * u.dimensionless_unscaled
        return 1

class FixedMassFracSFH(SFHBase):
    free_params = {'alpha': [0, 1, 10], 'z_today': [0.005, 0.01, 0.08]}

    def __init__(self, mass_fraction, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[SFH] Initialising FixedMassFracSFH model")
        self.mass_fractions = np.sort(mass_fraction)
        self.mass_fractions = np.insert(
            self.mass_fractions, [0, self.mass_fractions.size], [0, 1])
        self.today = cosmology.age(0)

        delta_f = self.mass_fractions[1:] - self.mass_fractions[:-1]
        for f, df in zip(self.mass_fractions[1:-1], delta_f[:-1]):
            max_time = min(self.today * df * 5, self.today)
            self.free_params[
                    f't_at_frac_{f:.4f}'] = [
                        -4,
                        np.log10((self.today * df).to_value("Gyr")),
                        np.log10(max_time.to_value("Gyr"))]

        self.model = pst.models.Tabular_ZPowerLaw(
            times=np.ones(self.mass_fractions.size) * u.Gyr,
            masses=self.mass_fractions * u.Msun,
            z_today=kwargs.get("z_today", 0.02)  * u.dimensionless_unscaled,
            alpha=kwargs.get("alpha", 0.0))

    def parse_free_params(self, free_params):
        times = np.array(
            [free_params[f't_at_frac_{f:.4f}'] for f in self.mass_fractions[1:-1]],
            dtype=float)
        cumulative = np.cumsum(10**times)
        if cumulative[-1] > self.today.to_value("Gyr"):
            return 0
        cumulative = np.insert(cumulative, (0, cumulative.size),
                               (0, self.today.to_value("Gyr")))
        # Update the mass of the tabular model
        self.model.table_t = cumulative * u.Gyr
        self.model.alpha = free_params['alpha']
        self.model.z_today = free_params['z_today'] * u.dimensionless_unscaled
        return 1

    def parse_datablock(self, datablock):
        times = np.array(
            [datablock["parameters", f't_at_frac_{f:.4f}'] for f in self.mass_fractions[1:-1]],
            dtype=float)
        cumulative = np.cumsum(10**times)
        if cumulative[-1] > self.today.to_value("Gyr"):
            return 0
        cumulative = np.insert(cumulative, (0, cumulative.size),
                               (0, self.today.to_value("Gyr")))
        # Update the mass of the tabular model
        self.model.table_t = cumulative * u.Gyr
        self.model.alpha = datablock["parameters",'alpha']
        self.model.z_today = datablock["parameters",'z_today'] * u.dimensionless_unscaled
        return 1

class ExponentialSFH(SFHBase):
    free_params = {'alpha': [0, 1, 10], 'z_today': [0.005, 0.01, 0.08],
                   "logtau": [-1, 0.5, 1.7]}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[SFH] Initialising FixedGridSFH model")
        self.time = kwargs.get("time")
        if self.time is None:
            self.time = self.today - np.geomspace(1e-5, 1, 200) * self.today
        self.time = np.sort(self.time)

        self.model = pst.models.Tabular_ZPowerLaw(
            times=self.time,
            masses=np.ones(self.time.size) * u.Msun,
            z_today=kwargs.get("z_today", 0.02)  * u.dimensionless_unscaled,
            alpha=kwargs.get("alpha", 0.0))

    def parse_free_params(self, free_params):
        self.tau = 10**free_params['logtau']
        m = 1 - np.exp(-self.time.to_value("Gyr") / self.tau)
        self.model.table_M = m / m[-1] * u.Msun
        self.model.alpha = free_params['alpha']
        self.model.z_today = free_params['z_today'] * u.dimensionless_unscaled
        return 1

    def parse_datablock(self, datablock):
        self.tau = 10**datablock['parameters', 'logtau']
        m = 1 - np.exp(-self.time.to_value("Gyr") / self.tau)
        self.model.table_M = m / m[-1] * u.Msun
        self.model.alpha = datablock['parameters', 'alpha']
        self.model.z_today = datablock['parameters', 'z_today'] * u.dimensionless_unscaled
        return 1
    
class LogNormalSFH(SFHBase):
    free_params = {'alpha': [0, 1, 10], 'z_today': [0.005, 0.01, 0.08],
                   "scale": [0.1, 3.0, 50], "lnt0": [-1, 0.5, 1.7]}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[SFH] Initialising LogNormalSFH model")
        self.free_params['lnt0'] = [np.log(0.1),
                                    np.log(self.today.to_value("Gyr") / 2),
                                    np.log(self.today.to_value("Gyr"))]
        self.model = pst.models.LogNormal_MFH(
            alpha=kwargs.get("alpha", 0.0),
            z_today=kwargs.get("z_today", 0.02)  * u.dimensionless_unscaled,
            lnt0=1., scale=1.)

    def parse_free_params(self, free_params):
        self.model.alpha = free_params['alpha']
        self.model.z_today = free_params['z_today'] * u.dimensionless_unscaled
        self.model.lnt0 = free_params['lnt0']
        self.model.scale = free_params['scale']
        return 1

    def parse_datablock(self, datablock):
        self.model.alpha = datablock['parameters', 'alpha']
        self.model.z_today = datablock['parameters', 'z_today'] * u.dimensionless_unscaled
        self.model.lnt0 = datablock['parameters', 'lnt0']
        self.model.scale = datablock['parameters', 'scale']
        return 1

class LogNormalQuenchedSFH(SFHBase):
    free_params = {'alpha': [0.0, 1.0, 3.0], 'z_today': [0.005, 0.01, 0.08],
                   "scale": [0.1, 3.0, 50], "lnt0": [-1, 0.5, 1.7],
                   "lnt_quench": [0, 1, 10], "lntau_quench": [-1, 0.5, 1.7]}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[SFH] Initialising FixedGridSFH model")
        self.free_params['lnt0'] = [np.log(0.1),
                                     np.log(self.today.to_value("Gyr") / 2),
                                     2 * np.log(self.today.to_value("Gyr"))]
        self.free_params['lnt_quench'] = [np.log(0.3),
                                          np.log(self.today.to_value("Gyr")),
                                          np.log(2 * self.today.to_value("Gyr"))
                                        ]
        self.free_params['lntau_quench'] = [np.log(0.1),
                                            np.log(0.5),
                                            np.log(self.today.to_value("Gyr"))]

        self.model = pst.models.LogNormalQuenched_MFH(
            alpha=kwargs.get("alpha", 0.0),
            z_today=kwargs.get("z_today", 0.02)  * u.dimensionless_unscaled,
            lnt0=1., scale=1.,
            t_quench=1., tau_quench=1.)

    def parse_free_params(self, free_params):
        self.model.alpha = free_params['alpha']
        self.model.z_today = free_params['z_today'] * u.dimensionless_unscaled
        self.model.lnt0 = free_params['lnt0']
        self.model.scale = free_params['scale']
        self.model.t_quench = np.exp(free_params['lnt_quench']) << u.Gyr
        self.model.tau_quench = np.exp(free_params['lntau_quench']) << u.Gyr
        return 1

    def parse_datablock(self, datablock):
        self.model.alpha = datablock['parameters', 'alpha']
        self.model.z_today = datablock['parameters', 'z_today'] * u.dimensionless_unscaled
        self.model.lnt0 = datablock['parameters', 'lnt0']
        self.model.scale = datablock['parameters', 'scale']
        self.model.t_quench = np.exp(datablock['parameters', 'lnt_quench']) << u.Gyr
        self.model.tau_quench = np.exp(datablock['parameters', 'lntau_quench']) << u.Gyr
        return 1

# class DoublePowerLawSFH(SFHBase):
#     free_params = {'alpha': [0, 1, 10], 'z_today': [0.005, 0.01, 0.08],
#                    "beta": [-1, 0.5, 1.7], "gama": [-1, 0.5, 1.7],
#                    'logtau': [-1, 0.7, 1]}

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         print("[SFH] Initialising FixedGridSFH model")
#         self.time = kwargs.get("time")
#         if self.time is None:
#             self.time = self.today - np.geomspace(1e-5, 0.99, 200) * self.today
#         self.time = np.sort(self.time)
#         self.free_params['logtau'] = [
#             np.log10(0.1),
#             np.log10(self.today.to_value("Gyr") / 2),
#             np.log10(self.today.to_value("Gyr"))]

#         self.model = pst.models.Tabular_ZPowerLaw(
#             times=self.time,
#             masses=np.ones(self.time.size) * u.Msun,
#             z_today=kwargs.get("z_today", 0.02)  * u.dimensionless_unscaled,
#             alpha=kwargs.get("alpha", 0.0))

#     @staticmethod
#     def cmf(t, tau, beta, gamma):
#         x = t / tau
#         bc = beta + gamma
#         f_x_b_c = x**(1 - beta) * (
#             (beta - 1) * beta * x**bc * hyp2f1(
#                 1, (gamma + 1) / bc, (gamma + 1) / bc + 1, -x**bc)
#             + (gamma + 1) * gamma * hyp2f1(1, (1 - beta) / bc, (gamma + 1) / bc,
#                                            -x**bc)
#             + gamma * (gamma + 1)
#         ) / ((beta - 1) * (gamma + 1) * bc)
#         return tau * f_x_b_c

#     def parse_free_params(self, free_params):
#         scale = free_params['scale']
#         logt0 = free_params['logt0']
#         m = self.cmf(self.logt, logt0, scale)
#         self.model.table_M = m / m[-1] * u.Msun
#         self.model.alpha = free_params['alpha']
#         self.model.z_today = free_params['z_today'] * u.dimensionless_unscaled
#         return 1

#     def parse_datablock(self, datablock):
#         scale = datablock['parameters', 'scale']
#         logt0 = datablock['parameters', 'logt0']
#         m = self.cmf(self.logt, logt0, scale)
#         self.model.table_M = m / m[-1] * u.Msun
#         self.model.alpha = datablock['parameters', 'alpha']
#         self.model.z_today = datablock['parameters', 'z_today'] * u.dimensionless_unscaled
#         return 1


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    ssp = pst.SSP.BaseGM()
    tau = 3e9 * u.yr
    times = np.linspace(1e-3, 13.7, 300) * u.Gyr
    sfh = FixedTimeSFH(times, today=13.7 * u.Gyr)
    sfh.model.z_today = 0.02
    sfh.model.alpha = 1.0
    sfh.model.table_M = (1 - np.exp(-sfh.time / tau)) * u.Msun
        
    times_2 = np.linspace(0.001, 1, 10)**2 * 13.7 * u.Gyr
    sfh_2 = FixedTimeSFH(times_2, today=13.7 * u.Gyr)
    sfh_2.model.z_today = 0.02
    sfh_2.model.alpha = 1.0
    sfh_2.model.table_M = (1 - np.exp(-sfh_2.time / tau)) * u.Msun
    


    plt.figure()
    plt.subplot(211)
    plt.plot(sfh.time, sfh.model.integral_SFR(sfh.time), '^-')
    plt.plot(sfh_2.time, sfh.model.integral_SFR(sfh_2.time), '+-')
    plt.plot(sfh_2.time, sfh_2.model.integral_SFR(sfh_2.time), 'o-')
    plt.plot(sfh_2.time, 1 - np.exp(-sfh_2.time.to_value('yr') / 3e9))
    plt.subplot(212)
    plt.plot(sfh.time, sfh.model.Z)
    plt.plot(sfh_2.time, sfh_2.model.Z)
    plt.plot(sfh_2.time, sfh.model.integral_Z_SFR(sfh_2.time), '+--')
    plt.plot(sfh_2.time, sfh_2.model.integral_Z_SFR(sfh_2.time), '--')
    plt.show()

    sed = sfh.model.compute_SED(ssp, t_obs=sfh.today)
    sed_2 = sfh_2.model.compute_SED(ssp, t_obs=sfh_2.today)

    plt.figure()
    plt.plot(ssp.wavelength, sed)
    plt.plot(ssp.wavelength, sed_2)
    plt.show()