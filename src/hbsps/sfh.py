"""
Star formation history fitting module
"""
import numpy as np

from cosmosis import DataBlock

import pst.utils
from hbsps.utils import cosmology
import pst
from astropy import units as u

# Star formation history models

class SFHBase():
    """Star formation history model.
    
    Description
    -----------
    This class serves as an interface between the sampler and PST SFH models.

    Attributes
    ----------
    free_params : dict
        Dictionary containing the free parameters of the model and their
        respective range of validity.
    redshift : float, optional, default=0.0
        Cosmological redshift at the time of the observation.
    today : astropy.units.Quantity
        Age of the universe at the time of observation. If not provided,
        it is computed using the default cosmology and the value of ``redshift``.
    """
    free_params = {}
    def __init__(self, *args, **kwargs):        
        self.redshift = kwargs.get("redshift", 0.0)
        self.today = kwargs.get("today", cosmology.age(self.redshift))

    def make_ini(self, ini_file):
        """Create a cosmosis .ini file.
        
        Parameters
        ----------
        ini_file : str
            Path to the output .ini file.
        """
        print("Making ini file: ", ini_file)
        with open(ini_file, "w") as file:
            file.write("[parameters]\n")
            for k, v in self.free_params.items():
                if len(v) > 1:
                    file.write(f"{k} = {v[0]} {v[1]} {v[2]}\n")
                else:
                    file.write(f"{k} = {v[0]}\n")

    def parse_free_params(self, free_params : dict):
        return self.parse_datablock(DataBlock.from_dict(
            dict(parameters=free_params)))

    def parse_datablock(self):
        pass


class ZPowerLawMixin:
    """Metallicity evolution as a power law in terms of the stellar mass formed."""
    free_params = {'alpha_powerlaw': [0, 0.5, 3], 'ism_metallicity_today': [0.005, 0.01, 0.08]}


class PieceWiseSFHMixin:

    @property
    def sfh_bin_keys(self):
        return self._sfh_bin_keys
    
    @sfh_bin_keys.setter
    def sfh_bin_keys(self, value):
        self._sfh_bin_keys = value

    def get_sfh_parameters_array(self, datablock : DataBlock, dtype=float):
        return np.array([datablock["parameters", key] for key in self.sfh_bin_keys],
                        dtype=dtype)

class FixedTimeSFH(SFHBase, ZPowerLawMixin, PieceWiseSFHMixin):
    """A SFH model with fixed time bins.
    
    Description
    -----------
    The SFH of a galaxy is modelled as a stepwise function where the free
    parameters correspond to the mass fraction formed on each bin.
    
    Upon initializaiton, the free parameters are set between -8 to 0 in terms
    of log(M/Msun). The starting point corresponds to the mass fraction formed
    assuming a constant star formation history.

    Attributes
    ----------
    lookback_time : astropy.units.Quantity
        Lookback time bin edges.

    """

    def __init__(self, lookback_time_bins, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[SFH] Initialising FixedGridSFH model")
        # From the begining of the Universe to the present date
        self.lookback_time = pst.utils.check_unit(
            np.sort(lookback_time_bins)[::-1], u.Gyr)

        self.lookback_time = np.insert(
            self.lookback_time, (0, self.lookback_time.size),
            (self.today.to(self.lookback_time.unit),
             0 << self.lookback_time.unit))

        self.time = self.today - self.lookback_time
        if (self.time < 0).any():
            print("[SFH] Warning: lookback time bin larger the age of the Universe")

        logm_min = kwargs.get('logmass_min', -6)
        print("[SFH] Setting up free parameters")
        print(f"[SFH] Minimum log(M/Msun)={logm_min}")
        self.sfh_bin_keys = []
        for lb in self.lookback_time[1:-1].to_value("Gyr"):
            # Initialise parameters assuming a constant star formation history
            k = f'logmass_at_{lb:.3f}'
            self.sfh_bin_keys.append(k)
            self.free_params[k] = [logm_min, self.today._to_value("Gyr") / lb, 0.0]

        # Initialise PST 
        self.model = pst.models.TabularCEM_ZPowerLaw(
            times=self.time,
            masses=np.ones(self.time.size) * u.Msun,
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02)  << u.dimensionless_unscaled,
            alpha_powerlaw=kwargs.get("alpha", 0.0))

    def parse_datablock(self, datablock : DataBlock):
        logm_formed = self.get_sfh_parameters_array(datablock)
        cumulative = np.cumsum(10**logm_formed)
        if cumulative[-1] > 1.0:
            return 0, cumulative[-1]
        cumulative = np.insert(cumulative, (0, cumulative.size), (0, 1))
        # Update the mass of the tabular model
        self.model.table_mass =  cumulative * u.Msun
        self.model.alpha_powerlaw = datablock["parameters", "alpha_powerlaw"]
        self.model.ism_metallicity_today = datablock[
            "parameters", "ism_metallicity_today"] << u.dimensionless_unscaled
        return 1, None


class FixedTime_sSFR_SFH(SFHBase, ZPowerLawMixin, PieceWiseSFHMixin):
    """A SFH model with fixed time bins.
    
    Description
    -----------
    The SFH of a galaxy is modelled as a stepwise function where the free
    parameters correspond to the average sSFR over the last ``lookback_time``.

    Attributes
    ----------
    lookback_time : astropy.units.Quantity
        Lookback time bin edges.

    """
    def __init__(self, lookback_time, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[SFH] Initialising FixedGrid-sSFR-SFH model")
        self.lookback_time = pst.utils.check_unit(
            np.sort(lookback_time)[::-1], u.Gyr)

        self.lookback_time = np.insert(
            self.lookback_time, (0, self.lookback_time.size),
            (self.today.to(self.lookback_time.unit),
             0 << self.lookback_time.unit))

        self.time = self.today - self.lookback_time
        self.sfh_bin_keys = []
        for lt in self.lookback_time[1:-1].to_value('yr'):
            k = f'logssfr_over_{np.log10(lt):.2f}_logyr'
            self.sfh_bin_keys.append(k)
            max_logssfr = np.min((np.log10(1 / lt), -8.0))
            self.free_params[k] = [-14.0,
                                   np.log10(1 / self.today.to_value("yr")),
                                   max_logssfr]

        self.model = pst.models.TabularCEM_ZPowerLaw(
            times=self.time,
            masses=np.ones(self.time.size) * u.Msun,
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02
                                             )  << u.dimensionless_unscaled,
            alpha_powerlaw=kwargs.get("alpha", 0.0))

    def parse_datablock(self, datablock : DataBlock):
        lt_yr = self.lookback_time[1:-1].to_value('yr')
        ssfr_over_last = self.get_sfh_parameters_array(datablock)
        mass_frac = 1 - lt_yr * 10**ssfr_over_last
        mass_frac = np.insert(mass_frac, [0, mass_frac.size], [0.0, 1.0])
        if (mass_frac[1:] - mass_frac[:-1] < 0).any():
            dm = mass_frac[1:] - mass_frac[:-1]
            return 0, 1 + np.abs(dm[dm < 0].sum())
        # Update the mass of the tabular model
        self.model.table_mass =  mass_frac * u.Msun
        self.model.alpha_powerlaw = datablock["parameters",'alpha_powerlaw']
        self.model.ism_metallicity_today = datablock["parameters",'ism_metallicity_today'] << u.dimensionless_unscaled
        return 1, None


class FixedMassFracSFH(SFHBase, ZPowerLawMixin, PieceWiseSFHMixin):
    """A SFH model with fixed mass fraction bins.
    
    Description
    -----------
    The SFH of a galaxy is modelled as a stepwise function where the free
    parameters correspond to the time at which a given fraction of the total stellar mass was
    formed.

    Attributes
    ----------
    mass_fractions : np.ndarray
        SFH mass fractions.
    lookback_time : astropy.units.Quantity
        Lookback time bin edges.

    """
    def __init__(self, mass_fraction, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[SFH] Initialising FixedMassFracSFH model")
        self.mass_fractions = np.sort(mass_fraction)
        self.mass_fractions = np.insert(
            self.mass_fractions, [0, self.mass_fractions.size], [0, 1])

        self.sfh_bin_keys = []
        for f in self.mass_fractions[1:-1]:
            k = f't_at_frac_{f:.4f}'
            self.sfh_bin_keys.append(k)
            self.free_params[k] = [0, f * self.today.to_value("Gyr"),
                                   self.today.to_value("Gyr")]

        self.model = pst.models.TabularCEM_ZPowerLaw(
            times=np.ones(self.mass_fractions.size) * u.Gyr,
            masses=self.mass_fractions * u.Msun,
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02
                                             )  << u.dimensionless_unscaled,
            alpha_powerlaw=kwargs.get("alpha", 0.0))

    def parse_datablock(self, datablock : DataBlock):
        times = self.get_sfh_parameters_array(datablock)
        dt = times[1:] - times[:-1]
        if (dt < 0).any():
            return 0, 1 + np.abs(dt[dt < 0].sum())
        times = np.insert(times, (0, times.size),
                          (0, self.today.to_value("Gyr")))
        # Update the mass of the tabular model
        self.model.table_t = times * u.Gyr
        self.model.alpha_powerlaw = datablock["parameters",'alpha_powerlaw']
        self.model.ism_metallicity_today = datablock["parameters",'ism_metallicity_today'] << u.dimensionless_unscaled
        return 1, None

# Analytical star formation histories

class ExponentialSFH(SFHBase, ZPowerLawMixin):
    """An analytical exponentially declining SFH model.
    
    Description
    -----------
    The SFH of a galaxy is modelled as an exponentially declining function.

    Attributes
    ----------
    time : astropy.units.Quantity
        Time bins to evaluate the SFH.
    lookback_time : astropy.units.Quantity

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[SFH] Initialising ExponentialSFH model")
        self.time = kwargs.get("time")
        if self.time is None:
            self.time = self.today - np.geomspace(1e-5, 1, 200) * self.today
        self.time = np.sort(self.time)

        # Initialise the free parameter
        self.free_params["logtau"] = kwargs.get("logtau", [-1, 0.5, 1.7])

        self.model = pst.models.TabularCEM_ZPowerLaw(
            times=self.time,
            mass_today = 1 << u.Msun,
            masses=np.ones(self.time.size) * u.Msun,
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02)  << u.dimensionless_unscaled,
            alpha_powerlaw=kwargs.get("alpha", 0.0))

    def parse_datablock(self, datablock : DataBlock):
        self.tau = 10**datablock['parameters', 'logtau']
        m = 1 - np.exp(-self.time.to_value("Gyr") / self.tau)
        self.model.table_mass = m / m[-1] * u.Msun
        self.model.alpha_powerlaw = datablock['parameters', 'alpha_powerlaw']
        self.model.ism_metallicity_today = datablock['parameters', 'ism_metallicity_today'] << u.dimensionless_unscaled
        return 1, None


class LogNormalSFH(SFHBase, ZPowerLawMixin):
    """An analytical log-normal declining SFH model.
    
    Description
    -----------
    The SFH of a galaxy is modelled as an log-normal declining function.

    Attributes
    ----------
    time : astropy.units.Quantity
        Time bins to evaluate the SFH.
    lookback_time : astropy.units.Quantity

    """

    free_params = {'alpha_powerlaw': [0, 1, 10],
                   'ism_metallicity_today': [0.005, 0.01, 0.08],
                   "scale": [0.1, 0.5, 3.0], "t0": [0.1, 3.0, 30.0]}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[SFH] Initialising LogNormalSFH model")
        self.model = pst.models.LogNormalZPowerLawCEM(
            today=self.today,
            mass_today=1.0 << u.Msun,
            alpha_powerlaw=kwargs.get("alpha", 0.0),
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02
                                             ) << u.dimensionless_unscaled,
            t0=1., scale=1.)

    def parse_free_params(self, free_params : dict):
        return self.parse_datablock(DataBlock.from_dict(
            dict(parameters=free_params)))

    def parse_datablock(self, datablock):
        self.model = pst.models.LogNormalZPowerLawCEM(
            today=self.today,
            mass_today=1.0 << u.Msun,
            alpha_powerlaw=datablock['parameters', 'alpha_powerlaw'],
            ism_metallicity_today=datablock['parameters', 'ism_metallicity_today'] << u.dimensionless_unscaled,
            t0=datablock['parameters', 't0'] << u.Gyr,
            scale=datablock['parameters', 'scale'])
        return 1, None


class LogNormalQuenchedSFH(SFHBase, ZPowerLawMixin):
    """An analytical log-normal declining SFH model including a quenching event.
    
    Description
    -----------
    The SFH of a galaxy is modelled as an log-normal declining function. A quenching
    event is modelled as an additional exponentially declining function that is 
    applied after the time of quenching.

    Attributes
    ----------
    time : astropy.units.Quantity
        Time bins to evaluate the SFH.
    lookback_time : astropy.units.Quantity

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[SFH] Initialising LogNorMalQuenched model")
        self.time = kwargs.get("time")
        if self.time is None:
            self.time = self.today - np.geomspace(1e-5, 1, 200) * self.today
        self.time = np.sort(self.time)

        self.free_params["scale"] = kwargs.get("scale", [0.1, 3.0, 50])
        self.free_params['t0'] = kwargs.get(
            "t0", [0.1, self.today.to_value("Gyr") / 2,
                   self.today.to_value("Gyr")])
        self.free_params['quenching_time'] = kwargs.get(
            "quenching_time", [0.3, self.today.to_value("Gyr"),
                               2 * self.today.to_value("Gyr")])

        self.model = pst.models.LogNormalQuenchedCEM(
            today=self.today,
            mass_today=1.0 << u.Msun,
            t0=1., scale=1.,
            alpha_powerlaw=kwargs.get("alpha", 0.0),
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02
                                             )  << u.dimensionless_unscaled,
            quenching_time=self.today)

    def parse_datablock(self, datablock : DataBlock):
        self.model.alpha_powerlaw = datablock['parameters', 'alpha_powerlaw']
        self.model.ism_metallicity_today = datablock['parameters', 'ism_metallicity_today'] << u.dimensionless_unscaled
        self.model.t0 = datablock['parameters', 't0'] << u.Gyr
        self.model.scale = datablock['parameters', 'scale']
        self.model.quenching_time = datablock['parameters', 'quenching_time'] << u.Gyr
        return 1, None


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    ssp = pst.SSP.BaseGM()
    tau = 3e9 * u.yr
    times = np.linspace(1e-3, 13., 300) * u.Gyr
    sfh = FixedTimeSFH(times, today=13.7 * u.Gyr)
    sfh.model.ism_metallicity_today = 0.02
    sfh.model.alpha_powerlaw = 1.0
    sfh.model.table_mass = (1 - np.exp(-sfh.time / tau)) * u.Msun

    times_2 = np.linspace(0.001, 1, 10)**2 * 13. * u.Gyr
    sfh_2 = FixedTimeSFH(times_2, today=13.7 * u.Gyr)
    sfh_2.model.ism_metallicity_today = 0.02
    sfh_2.model.alpha_powerlaw = 1.0
    sfh_2.model.table_mass = (1 - np.exp(-sfh_2.time / tau)) * u.Msun
    


    plt.figure()
    plt.subplot(211)
    plt.plot(sfh.time, sfh.model.stellar_mass_formed(sfh.time), '^-')
    plt.plot(sfh_2.time, sfh.model.stellar_mass_formed(sfh_2.time), '+-')
    plt.plot(sfh_2.time, sfh_2.model.stellar_mass_formed(sfh_2.time), 'o-')
    plt.plot(sfh_2.time, 1 - np.exp(-sfh_2.time.to_value('yr') / 3e9))
    plt.subplot(212)
    plt.plot(sfh_2.time, sfh.model.ism_metallicity(sfh_2.time), '+--')
    plt.plot(sfh_2.time, sfh_2.model.ism_metallicity(sfh_2.time), '--')
    plt.show()

    sed = sfh.model.compute_SED(ssp, t_obs=sfh.today)
    sed_2 = sfh_2.model.compute_SED(ssp, t_obs=sfh_2.today)

    plt.figure()
    plt.plot(ssp.wavelength, sed)
    plt.plot(ssp.wavelength, sed_2)
    plt.show()