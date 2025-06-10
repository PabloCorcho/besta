"""
Star formation history fitting module
"""
from abc import ABC, abstractmethod
import numpy as np
from astropy import units as u

from cosmosis import DataBlock
import pst

from besta.config import cosmology


# Star formation history models


class SFHBase(ABC):
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
        with open(ini_file, "w", encoding="utf-8") as file:
            file.write("[parameters]\n")
            for key, val in self.free_params.items():
                if len(val) > 1:
                    file.write(f"{key} = {val[0]} {val[1]} {val[2]}\n")
                else:
                    file.write(f"{key} = {val[0]}\n")

    def parse_free_params(self, free_params: dict):
        """Parse the SFH model free parameters from a dictionary.

        Parameters
        ----------
        free_params : dict
            Dictionary containing the SFH model free parameters.
        """
        db = DataBlock.from_dict({"parameters": free_params})
        return self.parse_datablock(db)

    @abstractmethod
    def parse_datablock(self, *args):
        """Parse the SFH model free parameters from a DataBlock."""


class ZPowerLawMixin:
    """Metallicity evolution as a power law in terms of the stellar mass formed."""

    free_params = {
        "alpha_powerlaw": [0, 0.5, 3],
        "ism_metallicity_today": [0.005, 0.01, 0.08],
    }


class PieceWiseSFHMixin:
    """Piece-wise star formation history model mixin.

    This mixing provides the common properties of piece-wise SFH models.
    """

    @property
    def sfh_bin_keys(self):
        """Keys associated to the bins of the SFH model."""
        return self._sfh_bin_keys

    @sfh_bin_keys.setter
    def sfh_bin_keys(self, value):
        self._sfh_bin_keys = value

    def get_sfh_parameters_array(self, datablock: DataBlock, dtype=float):
        """Get an array containing the values of each bin of the SFH.

        Parameters
        ----------
        databloc : DataBlock
            The datablock containing the values of each parameter of the SFH.
        """
        return np.array(
            [datablock["parameters", key] for key in self.sfh_bin_keys], dtype=dtype
        )


class FixedTimeSFH(ZPowerLawMixin, SFHBase, PieceWiseSFHMixin):
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
            np.sort(lookback_time_bins)[::-1], u.Gyr
        )

        self.lookback_time = np.insert(
            self.lookback_time,
            (0, self.lookback_time.size),
            (self.today.to(self.lookback_time.unit), 0 << self.lookback_time.unit),
        )

        self.time = self.today - self.lookback_time
        if (self.time < 0).any():
            print("[SFH] Warning: lookback time bin larger the age of the Universe")

        logm_min = kwargs.get("logmass_min", -6)
        print("[SFH] Setting up free parameters")
        print(f"[SFH] Minimum log(M/Msun)={logm_min}")
        self.sfh_bin_keys = []
        for lbt in self.lookback_time[1:-1].to_value("Gyr"):
            # Initialise parameters assuming a constant star formation history
            k = f"logmass_at_{lbt:.3f}"
            self.sfh_bin_keys.append(k)
            self.free_params[k] = [logm_min, self.today._to_value("Gyr") / lbt, 0.0]

        # Initialise PST
        self.model = pst.models.TabularCEM_ZPowerLaw(
            times=self.time,
            masses=np.ones(self.time.size) << u.Msun,
            today=self.today,
            mass_today=1 << u.Msun,
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02)
            << u.dimensionless_unscaled,
            alpha_powerlaw=kwargs.get("alpha", 0.0),
        )

    def parse_datablock(self, datablock: DataBlock):
        logm_formed = self.get_sfh_parameters_array(datablock)
        cumulative = np.cumsum(10**logm_formed)
        if cumulative[-1] > 1.0:
            return 0, cumulative[-1]
        cumulative = np.insert(cumulative, (0, cumulative.size), (0, 1))
        # Update the mass of the tabular model
        self.model.table_mass = cumulative << u.Msun
        self.model.alpha_powerlaw = datablock["parameters", "alpha_powerlaw"]
        self.model.ism_metallicity_today = (
            datablock["parameters", "ism_metallicity_today"] << u.dimensionless_unscaled
        )
        return 1, None


class FixedCosmicTimeSFH(ZPowerLawMixin, SFHBase, PieceWiseSFHMixin):
    """A SFH model with fixed time bins.

    Description
    -----------
    The SFH of a galaxy is modelled as a stepwise function where the free
    parameters correspond to ... #TODO

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
            np.sort(lookback_time_bins)[::-1], u.Gyr
        )

        # Lookbacktime [T_univ, t1, ... tn, 0]
        self.lookback_time = np.insert(
            self.lookback_time,
            (0, self.lookback_time.size),
            (self.today.to(self.lookback_time.unit), 0 << self.lookback_time.unit),
        )

        self.time = self.today - self.lookback_time
        # Massed fraction formed on each bin [0, t1], [t1, t2], ..., [tn, today]
        self.bin_masses = np.zeros(self.time.size - 2)

        if (self.time < 0).any():
            print("[SFH] Warning: lookback time bin larger the age of the Universe")

        print("[SFH] Setting up free parameters")
        self.sfh_bin_keys = []
        for lbt in self.lookback_time[1:-1].to_value("Gyr"):
            # Initialise parameters assuming a constant star formation history
            k = f"coeff_at_{lbt:.3f}"
            self.sfh_bin_keys.append(k)
            self.free_params[k] = [0.0, 0.5, 1.0]

        # Initialise PST
        self.model = pst.models.TabularCEM_ZPowerLaw(
            times=self.time,
            masses=np.ones(self.time.size) << u.Msun,
            today=self.today,
            mass_today=1 << u.Msun,
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02)
            << u.dimensionless_unscaled,
            alpha_powerlaw=kwargs.get("alpha", 0.0),
        )

    def update_mass(self, i, coeff):
        """Update a bin of the stellar mass fraction formed."""
        self.bin_masses[i] = coeff * (1 - np.sum(self.bin_masses[:i]))

    def parse_datablock(self, datablock: DataBlock):
        coefficients = self.get_sfh_parameters_array(datablock)
        # The first coefficient correspond to the mass fraction between
        # the origin of the universe and the first input time (largest lookback time)
        self.bin_masses[0] = coefficients[0]
        # Update the rest of the elements
        _ = [
            self.update_mass(i, coefficients[i]) for i in range(1, self.bin_masses.size)
        ]
        # Mass formation history. The last bin uses the remanining fraction)
        cumulative = np.insert(
            np.cumsum(self.bin_masses), (0, self.bin_masses.size), (0, 1)
        )
        # Update the mass of the tabular model
        self.model.table_mass = cumulative << u.Msun
        self.model.alpha_powerlaw = datablock["parameters", "alpha_powerlaw"]
        self.model.ism_metallicity_today = (
            datablock["parameters", "ism_metallicity_today"] << u.dimensionless_unscaled
        )
        return 1, None


class FlexibleCosmicTimeSFH(ZPowerLawMixin, SFHBase, PieceWiseSFHMixin):
    """A SFH model with fixed time bins.

    Description
    -----------
    The SFH of a galaxy is modelled as a stepwise function where the free
    parameters correspond to ... #TODO

    Attributes
    ----------
    lookback_time : astropy.units.Quantity
        Lookback time bin edges.

    """

    def __init__(self, n_times, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[SFH] Initialising FixedGridSFH model")
        # From the begining of the Universe to the present date
        self.lookback_time = (
            np.geomspace(5e-2, self.today.to_value("Gyr"), n_times + 1)[::-1] << u.Gyr
        )

        # Lookbacktime [T_univ, t1, ... tn, 0]
        self.lookback_time = np.insert(
            self.lookback_time, self.lookback_time.size, 0 << self.lookback_time.unit
        )
        print("Automatic choice of lookback times: ", self.lookback_time)
        self.time = self.today - self.lookback_time
        # Massed fraction formed on each bin [0, t1], [t1, t2], ..., [tn, today]
        self.bin_masses = np.zeros(self.time.size - 2)

        if (self.time < 0).any():
            print("[SFH] Warning: lookback time bin larger the age of the Universe")

        print("[SFH] Setting up free parameters")
        self.sfh_bin_keys = []
        for i, lbt in enumerate(self.lookback_time[1:-1].to_value("Gyr")):
            # Initialise parameters assuming a constant star formation history
            print("Lookback time: ", lbt, " --> coeff :", i + 1)
            k = f"coeff_{i + 1}"
            self.sfh_bin_keys.append(k)
            self.free_params[k] = [0.0, 0.5, 1.0]

        # Initialise PST
        self.model = pst.models.TabularCEM_ZPowerLaw(
            times=self.time,
            masses=np.ones(self.time.size) << u.Msun,
            today=self.today,
            mass_today=1 << u.Msun,
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02)
            << u.dimensionless_unscaled,
            alpha_powerlaw=kwargs.get("alpha", 0.0),
        )

    def update_mass(self, i, coeff):
        """Update a bin of the stellar mass fraction formed."""
        self.bin_masses[i] = coeff * (1 - np.sum(self.bin_masses[:i]))

    def parse_datablock(self, datablock: DataBlock):
        coefficients = self.get_sfh_parameters_array(datablock)
        # The first coefficient correspond to the mass fraction between
        # the origin of the universe and the first input time (largest lookback time)
        self.bin_masses[0] = coefficients[0]
        # Update the rest of the elements
        _ = [
            self.update_mass(i, coefficients[i]) for i in range(1, self.bin_masses.size)
        ]
        # Mass formation history. The last bin uses the remanining fraction)
        cumulative = np.insert(
            np.cumsum(self.bin_masses), (0, self.bin_masses.size), (0, 1)
        )
        # Update the mass of the tabular model
        self.model.table_mass = cumulative << u.Msun
        self.model.alpha_powerlaw = datablock["parameters", "alpha_powerlaw"]
        self.model.ism_metallicity_today = (
            datablock["parameters", "ism_metallicity_today"] << u.dimensionless_unscaled
        )
        return 1, None


class FixedTime_sSFR_SFH(ZPowerLawMixin, SFHBase, PieceWiseSFHMixin):
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
        self.lookback_time = pst.utils.check_unit(np.sort(lookback_time)[::-1], u.Gyr)

        self.lookback_time = np.insert(
            self.lookback_time,
            (0, self.lookback_time.size),
            (self.today.to(self.lookback_time.unit), 0 << self.lookback_time.unit),
        )

        self.time = self.today - self.lookback_time
        self.sfh_bin_keys = []
        for lbt in self.lookback_time[1:-1].to_value("yr"):
            k = f"logssfr_over_{np.log10(lbt):.2f}_logyr"
            self.sfh_bin_keys.append(k)
            max_logssfr = np.min((np.log10(1 / lbt), -8.0))
            self.free_params[k] = [
                -14.0,
                np.log10(1 / self.today.to_value("yr")),
                max_logssfr,
            ]

        self.model = pst.models.TabularCEM_ZPowerLaw(
            times=self.time,
            masses=np.ones(self.time.size) << u.Msun,
            today=self.today,
            mass_today=1 << u.Msun,
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02)
            << u.dimensionless_unscaled,
            alpha_powerlaw=kwargs.get("alpha", 0.0),
        )

    def parse_datablock(self, datablock: DataBlock):
        lt_yr = self.lookback_time[1:-1].to_value("yr")
        ssfr_over_last = self.get_sfh_parameters_array(datablock)
        mass_frac = 1 - lt_yr * 10**ssfr_over_last
        mass_frac = np.insert(mass_frac, [0, mass_frac.size], [0.0, 1.0])
        if (mass_frac[1:] - mass_frac[:-1] < 0).any():
            delta_m = mass_frac[1:] - mass_frac[:-1]
            return 0, 1 + np.abs(delta_m[delta_m < 0].sum())
        # Update the mass of the tabular model
        self.model.table_mass = mass_frac << u.Msun
        self.model.alpha_powerlaw = datablock["parameters", "alpha_powerlaw"]
        self.model.ism_metallicity_today = (
            datablock["parameters", "ism_metallicity_today"] << u.dimensionless_unscaled
        )
        return 1, None


class FixedMassFracSFH(ZPowerLawMixin, SFHBase, PieceWiseSFHMixin):
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
            self.mass_fractions, [0, self.mass_fractions.size], [0, 1]
        )

        self.sfh_bin_keys = []
        for frc in self.mass_fractions[1:-1]:
            k = f"t_at_frac_{frc:.4f}"
            self.sfh_bin_keys.append(k)
            self.free_params[k] = [
                0,
                frc * self.today.to_value("Gyr"),
                self.today.to_value("Gyr"),
            ]

        self.model = pst.models.TabularCEM_ZPowerLaw(
            times=np.ones(self.mass_fractions.size) << u.Gyr,
            masses=self.mass_fractions << u.Msun,
            today=self.today,
            mass_today=1 << u.Msun,
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02)
            << u.dimensionless_unscaled,
            alpha_powerlaw=kwargs.get("alpha", 0.0),
        )

    def parse_datablock(self, datablock: DataBlock):
        times = self.get_sfh_parameters_array(datablock)
        delta_t = times[1:] - times[:-1]
        if (delta_t < 0).any():
            return 0, 1 + np.abs(delta_t[delta_t < 0].sum())
        times = np.insert(times, (0, times.size), (0, self.today.to_value("Gyr")))
        # Update the mass of the tabular model
        self.model.table_t = times * u.Gyr
        self.model.alpha_powerlaw = datablock["parameters", "alpha_powerlaw"]
        self.model.ism_metallicity_today = (
            datablock["parameters", "ism_metallicity_today"] << u.dimensionless_unscaled
        )
        return 1, None


# class FixedMassFracLinSFH(ZPowerLawMixin, SFHBase, PieceWiseSFHMixin):
#     """A SFH model with fixed mass fraction bins.

#     Description
#     -----------
#     The SFH of a galaxy is modelled as a stepwise function where the free
#     parameters correspond to the time at which a given fraction of the total stellar mass was
#     formed.

#     Attributes
#     ----------
#     mass_fractions : np.ndarray
#         SFH mass fractions.
#     lookback_time : astropy.units.Quantity
#         Lookback time bin edges.

#     """

#     def __init__(self, mass_fraction, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         print("[SFH] Initialising FixedMassFracSFH model")
#         # Convert from mass fraction formed in the last t to mass fraction
#         # at time (today - t)
#         self.mass_fractions = 1 - np.sort(mass_fraction)[::-1]
#         self.mass_fractions = np.insert(
#             self.mass_fractions, [0, self.mass_fractions.size], [0, 1]
#         )

#         self.bin_times_frac = np.zeros(self.mass_fractions.size - 2)
#         for m in self.mass_fractions[1:-1]:
#             # Initialise parameters assuming a constant star formation history
#             k = f"coeff_at_{m:.3f}"
#             self.sfh_bin_keys.append(k)
#             self.free_params[k] = [0.0, 0.5, 1.0]

#         self.model = pst.models.TabularCEM_ZPowerLaw(
#             times=np.ones(self.mass_fractions.size) << u.Gyr,
#             masses=self.mass_fractions << u.Msun,
#             mass_today=1 << u.Msun,
#             ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02)
#             << u.dimensionless_unscaled,
#             alpha_powerlaw=kwargs.get("alpha", 0.0),
#         )

#     def update_time_fraction(self, i, coeff):
#         """Update a bin of the stellar mass fraction formed."""
#         self.bin_times_frac[i] = coeff * (1 - np.sum(self.bin_times_frac[:i]))

#     def parse_datablock(self, datablock: DataBlock):
#         coefficients = self.get_sfh_parameters_array(datablock)
#         # The first coefficient correspond to the mass fraction between
#         # the origin of the universe and the first input time (largest lookback time)
#         self.bin_times_frac[0] = coefficients[0]
#         # Update the rest of the elements
#         _ = [
#             self.update_time_fraction(i, coefficients[i])
#             for i in range(1, self.bin_times_frac.size)
#         ]
#         # Update the mass of the tabular model
#         t_frac = np.isert(
#             np.cumsum(self.bin_times_frac), (0, self.bin_times_frac.size), (0, 1)
#         )
#         self.model.table_t = t_frac * self.today
#         self.model.alpha_powerlaw = datablock["parameters", "alpha_powerlaw"]
#         self.model.ism_metallicity_today = (
#             datablock["parameters", "ism_metallicity_today"] << u.dimensionless_unscaled
#         )
#         return 1, None

#     def parse_datablock(self, datablock: DataBlock):
#         times = self.get_sfh_parameters_array(datablock)
#         dt = times[1:] - times[:-1]
#         if (dt < 0).any():
#             return 0, 1 + np.abs(dt[dt < 0].sum())
#         times = np.insert(times, (0, times.size), (0, self.today.to_value("Gyr")))
#         # Update the mass of the tabular model
#         self.model.table_t = times * u.Gyr
#         self.model.alpha_powerlaw = datablock["parameters", "alpha_powerlaw"]
#         self.model.ism_metallicity_today = (
#             datablock["parameters", "ism_metallicity_today"] << u.dimensionless_unscaled
#         )
#         return 1, None


# Analytical star formation histories


class ExponentialSFH(ZPowerLawMixin, SFHBase):
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
            today=self.today,
            mass_today=1 << u.Msun,
            masses=np.ones(self.time.size) << u.Msun,
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02)
            << u.dimensionless_unscaled,
            alpha_powerlaw=kwargs.get("alpha_powerlaw", 0.0),
        )

    def parse_datablock(self, datablock: DataBlock):
        tau = 10 ** datablock["parameters", "logtau"]
        mass = 1 - np.exp(-self.time.to_value("Gyr") / tau)
        self.model.table_mass = mass / mass[-1] << u.Msun
        self.model.alpha_powerlaw = datablock["parameters", "alpha_powerlaw"]
        self.model.ism_metallicity_today = (
            datablock["parameters", "ism_metallicity_today"] << u.dimensionless_unscaled
        )
        return 1, None


class DelayedTauSFH(ZPowerLawMixin, SFHBase):
    r"""An exponentially declining delayed-tau SFH model.

    Description
    -----------
    The SFH of a galaxy is modelled as an exponentially declining function.

    .. math::
        M_\star(t) = M_{inf} \cdot (1 - e^{-t/\tau} \frac{t + \tau}{\tau})

    Attributes
    ----------
    time : astropy.units.Quantity
        Time bins to evaluate the SFH.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[SFH] Initialising DelayedTauSFH model")
        # Initialise the free parameter
        self.free_params["logtau"] = kwargs.get("logtau", [-1, 0.5, 1.7])

        self.model = pst.models.ExponentialDelayedZPowerLawCEM(
            today=self.today,
            mass_today=1 << u.Msun,
            tau=1 << u.Gyr,
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02)
            << u.dimensionless_unscaled,
            alpha_powerlaw=kwargs.get("alpha_powerlaw", 0.0),
        )

    def parse_datablock(self, datablock: DataBlock):
        self.model = pst.models.ExponentialDelayedZPowerLawCEM(
            today=self.today,
            mass_today=1.0 << u.Msun,
            tau=10 ** datablock["parameters", "logtau"],
            alpha_powerlaw=datablock["parameters", "alpha_powerlaw"],
            ism_metallicity_today=datablock["parameters", "ism_metallicity_today"]
            << u.dimensionless_unscaled,
        )
        return 1, None


class DelayedTauQuenchedSFH(ZPowerLawMixin, SFHBase):
    r"""An exponentially declining delayed-tau SFH model with a quenching event.

    Description
    -----------
    The SFH of a galaxy is modelled as an exponentially declining function.

    .. math::
        M_\star(t) = M_{inf} \cdot (1 - e^{-t/\tau} \frac{t + \tau}{\tau})

    After the quenching event, taking place at :math:`t_{quench}`, the
    stellas mass will be :math:`M_\star(t)=M_\star(t_{quench})` for all times
    larget than :math:`t_{quench}`.

    Attributes
    ----------
    time : astropy.units.Quantity
        Time bins to evaluate the SFH.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[SFH] Initialising DelayedTauQuenchedSFH model")
        # Initialise the free parameter
        self.free_params["logtau"] = kwargs.get("logtau", [-1, 0.5, 1.7])
        self.free_params["quenching_time"] = kwargs.get(
            "quenching_time", [0, self.today / 2, self.today]
        )

        self.model = pst.models.ExponentialDelayedQuenchedCEM(
            today=self.today,
            mass_today=1 << u.Msun,
            tau=1 << u.Gyr,
            quenching_time=1 << u.Gyr,
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02)
            << u.dimensionless_unscaled,
            alpha_powerlaw=kwargs.get("alpha_powerlaw", 0.0),
        )

    def parse_datablock(self, datablock: DataBlock):
        self.model = pst.models.ExponentialDelayedQuenchedCEM(
            today=self.today,
            mass_today=1.0 << u.Msun,
            tau=10 ** datablock["parameters", "logtau"],
            quenching_time=datablock["parameters", "quenching_time"],
            alpha_powerlaw=datablock["parameters", "alpha_powerlaw"],
            ism_metallicity_today=datablock["parameters", "ism_metallicity_today"]
            << u.dimensionless_unscaled,
        )
        return 1, None


class LogNormalSFH(ZPowerLawMixin, SFHBase):
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

    free_params = {
        "alpha_powerlaw": [0, 1, 10],
        "ism_metallicity_today": [0.005, 0.01, 0.08],
        "scale": [0.1, 0.5, 3.0],
        "t0": [0.1, 3.0, 30.0],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("[SFH] Initialising LogNormalSFH model")
        self.model = pst.models.LogNormalZPowerLawCEM(
            today=self.today,
            mass_today=1.0 << u.Msun,
            alpha_powerlaw=kwargs.get("alpha_powerlaw", 0.0),
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02)
            << u.dimensionless_unscaled,
            t0=1.0,
            scale=1.0,
        )

    def parse_datablock(self, datablock: DataBlock):
        self.model = pst.models.LogNormalZPowerLawCEM(
            today=self.today,
            mass_today=1.0 << u.Msun,
            alpha_powerlaw=datablock["parameters", "alpha_powerlaw"],
            ism_metallicity_today=datablock["parameters", "ism_metallicity_today"]
            << u.dimensionless_unscaled,
            t0=datablock["parameters", "t0"] << u.Gyr,
            scale=datablock["parameters", "scale"],
        )
        return 1, None


class LogNormalQuenchedSFH(ZPowerLawMixin, SFHBase):
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
        print("[SFH] Initialising LogNormalQuenched model")
        self.time = kwargs.get("time")
        if self.time is None:
            self.time = self.today - np.geomspace(1e-5, 1, 200) * self.today
        self.time = np.sort(self.time)

        self.free_params["scale"] = kwargs.get("scale", [0.1, 3.0, 50])
        self.free_params["t0"] = kwargs.get(
            "t0", [0.1, self.today.to_value("Gyr") / 2, self.today.to_value("Gyr")]
        )
        self.free_params["quenching_time"] = kwargs.get(
            "quenching_time",
            [0.3, self.today.to_value("Gyr"), 2 * self.today.to_value("Gyr")],
        )

        self.model = pst.models.LogNormalQuenchedCEM(
            today=self.today,
            mass_today=1.0 << u.Msun,
            t0=1.0 << u.Gyr,
            scale=1.0,
            alpha_powerlaw=kwargs.get("alpha_powerlaw", 0.0),
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02)
            << u.dimensionless_unscaled,
            quenching_time=self.today,
        )

    def parse_datablock(self, datablock: DataBlock):
        self.model = pst.models.LogNormalQuenchedCEM(
            today=self.today,
            mass_today=1.0 << u.Msun,
            alpha_powerlaw=datablock["parameters", "alpha_powerlaw"],
            ism_metallicity_today=datablock["parameters", "ism_metallicity_today"]
            << u.dimensionless_unscaled,
            t0=datablock["parameters", "t0"] << u.Gyr,
            scale=datablock["parameters", "scale"],
            quenching_time=datablock["parameters", "quenching_time"] << u.Gyr,
        )
        return 1, None


# Mr Krtxo \(ﾟ▽ﾟ)/
