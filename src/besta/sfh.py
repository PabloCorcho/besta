"""
Star formation history fitting module
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import lgamma
import numpy as np
from astropy import units as u

from cosmosis import DataBlock
from pst import cem
from pst.model import Parameter
from pst.utils import check_unit
from besta.config import cosmology
from besta.logging import get_logger

logger = get_logger(__name__)


def _softmax(x):
    """Numerically stable softmax."""
    x = np.asarray(x, dtype=float)
    x_shift = x - np.max(x)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x)


def _sigmoid(x):
    """Simple sigmoid to clamp values into (0, 1)."""
    return 1.0 / (1.0 + np.exp(-x))


def _logit(x):
    """Inverse of sigmoid on (0,1)."""
    x = np.asarray(x, dtype=float)
    if np.any((x <= 0) | (x >= 1)):
        raise ValueError("Logit is only defined for values strictly within (0, 1).")
    return np.log(x) - np.log1p(-x)


def validate_monotonic(array, *, strict=True, name="array"):
    """Validate that the input array is monotonic increasing."""
    arr = np.asarray(array)
    diff = np.diff(arr)
    if strict:
        ok = np.all(diff > 0)
    else:
        ok = np.all(diff >= 0)
    if not ok:
        raise ValueError(f"{name} must be monotonically increasing.")
    return True


# SFH Priors

@dataclass(frozen=True)
class SFHSmoothnessPrior:
    r"""Legacy index-based Gaussian smoothness prior.

    This class preserves the original BESTA smoothness prior for
    reproducibility. New fits use :class:`SFHRobustTimeCurvaturePrior` by
    default when SFH smoothness regularisation is enabled.

    This prior regularises a piecewise star formation history by penalising
    variations in the logarithm of the bin-averaged star formation rate.

    For a set of stellar masses, or mass fractions, formed in disjoint time
    bins, the average star formation rate in bin :math:`i` is defined as

    .. math::

        \mathrm{SFR}_i = \frac{\Delta M_i}{\Delta t_i},

    where :math:`\Delta M_i` is the stellar mass formed in the bin and
    :math:`\Delta t_i` is its duration. The logarithmic star formation rate is

    .. math::

        y_i = \log_{10}\left(
            \mathrm{SFR}_i + \mathrm{SFR}_{\mathrm{min}}
        \right),

    where :math:`\mathrm{SFR}_{\mathrm{min}}` is a small positive number used
    to avoid evaluating the logarithm at zero.

    For a first-order prior, the residuals are the differences between
    neighbouring logarithmic star formation rates,

    .. math::

        r_i^{(1)} = y_{i+1} - y_i,

    and the logarithmic prior is

    .. math::

        \ln p_{\mathrm{smooth}}
        =
        -\frac{1}{2}
        \sum_i
        \left(
            \frac{r_i^{(1)}}{\sigma_{\mathrm{dex}}}
        \right)^2.

    This form favours similar star formation rates in adjacent bins and
    therefore tends to favour approximately constant star formation histories.

    For a second-order prior, the residuals are

    .. math::

        r_i^{(2)}
        =
        y_{i+2} - 2y_{i+1} + y_i,

    and the logarithmic prior is

    .. math::

        \ln p_{\mathrm{smooth}}
        =
        -\frac{1}{2}
        \sum_i
        \left(
            \frac{r_i^{(2)}}{\sigma_{\mathrm{dex}}}
        \right)^2.

    This form penalises changes in the local logarithmic SFR gradient. It
    therefore suppresses isolated spikes while allowing smoothly rising or
    declining star formation histories.

    The current implementation defines finite differences with respect to the
    bin index rather than physical time. Consequently, the second-order prior
    corresponds exactly to a curvature prior only when the time-bin centres
    are approximately uniformly spaced. For strongly irregular time grids, a
    time-weighted finite-difference prior may be more appropriate.

    Parameters
    ----------
    sigma_dex : float, optional
        Characteristic allowed variation in logarithmic star formation rate,
        in dex. For ``order=1``, this controls differences between adjacent
        values of :math:`\log_{10}(\mathrm{SFR})`. For ``order=2``, it controls
        second differences in :math:`\log_{10}(\mathrm{SFR})`. Smaller values
        impose stronger smoothing. The default is 0.5.
    order : {1, 2}, optional
        Order of the finite-difference regularisation.

        * ``1`` penalises differences between adjacent logarithmic SFRs.
        * ``2`` penalises second differences in logarithmic SFR.

        The default is 2.
    min_sfr : float, optional
        Positive numerical floor added to the bin-averaged SFR before taking
        its base-10 logarithm. Its units must be consistent with the units of
        ``mass_per_bin / delta_t`` used when evaluating the prior. The default
        is ``1e-12``.

    Notes
    -----
    The prior is invariant under multiplication of all bin masses by the same
    positive constant because finite differences remove the corresponding
    additive offset in :math:`\log_{10}(\mathrm{SFR})`.

    The returned quantity is an additive log-prior contribution and should be
    combined with the remaining terms in the posterior as

    .. math::

        \ln p(\boldsymbol{\theta}\mid D)
        =
        \ln \mathcal{L}(D\mid\boldsymbol{\theta})
        + \ln p_{\mathrm{base}}(\boldsymbol{\theta})
        + \ln p_{\mathrm{smooth}}(\boldsymbol{\theta}).

    The normalisation constant of the Gaussian prior is omitted because it is
    independent of the sampled SFH parameters when ``sigma_dex`` is fixed.
    """
    sigma_dex: float = 0.5
    order: int = 2
    min_sfr: float = 1e-12

    def __post_init__(self):
        if self.sigma_dex <= 0:
            raise ValueError("sigma_dex must be strictly positive.")
        if self.order not in (1, 2):
            raise ValueError("order must be either 1 or 2.")
        if self.min_sfr <= 0:
            raise ValueError("min_sfr must be strictly positive.")

    def __call__(self, mass_per_bin, time_edges) -> float:
        mass_per_bin = np.asarray(mass_per_bin, dtype=float)
        time_edges = np.asarray(time_edges, dtype=float)

        if mass_per_bin.ndim != 1 or time_edges.ndim != 1:
            raise ValueError("Inputs must be one-dimensional.")

        if time_edges.size != mass_per_bin.size + 1:
            raise ValueError(
                "time_edges must have length len(mass_per_bin) + 1."
            )

        if (
            not np.all(np.isfinite(mass_per_bin))
            or not np.all(np.isfinite(time_edges))
            or np.any(mass_per_bin < 0)
        ):
            return -1e20

        delta_t = np.diff(time_edges)

        if np.any(delta_t <= 0):
            return -1e20

        sfr = mass_per_bin / delta_t
        log_sfr = np.log10(sfr + self.min_sfr)

        if log_sfr.size <= self.order:
            return 0.0

        residuals = np.diff(log_sfr, n=self.order)

        return float(
            -0.5 * np.dot(residuals, residuals) / self.sigma_dex**2
        )


@dataclass(frozen=True)
class SFHRobustTimeCurvaturePrior:
    r"""Robust smoothness prior defined on an irregular physical-time grid.

    The prior first converts the bin-averaged SFRs into dimensionless values
    relative to the lifetime-averaged SFR,

    .. math::

        y_i = \log_{10}\left[
            \frac{(\Delta M_i / \Delta t_i)}
                 {(\sum_j \Delta M_j / \sum_j \Delta t_j)}
            + \epsilon
        \right].

    For each interior bin, it then compares :math:`y_i` with the value
    obtained by linearly interpolating its two neighbours at the physical
    bin-centre time. This residual is zero for any log-SFR history that is
    linear in physical time, even when the time bins are irregular.

    The residuals follow a Student-t distribution. Its heavy tails retain
    regularisation around smooth solutions without effectively excluding
    genuine bursts or quenching transitions.

    Parameters
    ----------
    sigma_dex : float, optional
        Student-t scale of the local interpolation residuals in dex. The
        default is 0.3 dex.
    dof : float, optional
        Degrees of freedom of the Student-t distribution. The default is 3,
        which gives heavy tails and finite variance.
    relative_sfr_floor : float, optional
        Positive floor added to SFR divided by lifetime-averaged SFR. The
        default is 1e-4 and is independent of the input mass units.
    """

    sigma_dex: float = 0.3
    dof: float = 3.0
    relative_sfr_floor: float = 1e-4

    def __post_init__(self):
        if not np.isfinite(self.sigma_dex) or self.sigma_dex <= 0:
            raise ValueError("sigma_dex must be finite and strictly positive.")
        if not np.isfinite(self.dof) or self.dof <= 0:
            raise ValueError("dof must be finite and strictly positive.")
        if (
            not np.isfinite(self.relative_sfr_floor)
            or self.relative_sfr_floor <= 0
        ):
            raise ValueError(
                "relative_sfr_floor must be finite and strictly positive."
            )

    def __call__(self, mass_per_bin, time_edges) -> float:
        mass_per_bin = np.asarray(mass_per_bin, dtype=float)
        time_edges = np.asarray(time_edges, dtype=float)

        if mass_per_bin.ndim != 1 or time_edges.ndim != 1:
            raise ValueError("Inputs must be one-dimensional.")
        if time_edges.size != mass_per_bin.size + 1:
            raise ValueError(
                "time_edges must have length len(mass_per_bin) + 1."
            )
        if (
            not np.all(np.isfinite(mass_per_bin))
            or not np.all(np.isfinite(time_edges))
            or np.any(mass_per_bin < 0)
        ):
            return -1e20

        delta_t = np.diff(time_edges)
        total_mass = np.sum(mass_per_bin)
        total_time = np.sum(delta_t)
        if np.any(delta_t <= 0) or total_mass <= 0 or total_time <= 0:
            return -1e20

        if mass_per_bin.size <= 2:
            return 0.0

        sfr = mass_per_bin / delta_t
        mean_sfr = total_mass / total_time
        relative_sfr = sfr / mean_sfr
        log_sfr = np.log10(relative_sfr + self.relative_sfr_floor)

        time_centres = 0.5 * (time_edges[:-1] + time_edges[1:])
        left_span = time_centres[1:-1] - time_centres[:-2]
        right_span = time_centres[2:] - time_centres[1:-1]
        neighbour_span = left_span + right_span
        interpolated_log_sfr = (
            right_span * log_sfr[:-2] + left_span * log_sfr[2:]
        ) / neighbour_span
        residuals = log_sfr[1:-1] - interpolated_log_sfr

        nu = self.dof
        sigma = self.sigma_dex
        log_normalization = (
            lgamma(0.5 * (nu + 1.0))
            - lgamma(0.5 * nu)
            - 0.5 * np.log(nu * np.pi)
            - np.log(sigma)
        )
        log_shape = -0.5 * (nu + 1.0) * np.log1p(
            residuals**2 / (nu * sigma**2)
        )
        return float(residuals.size * log_normalization + np.sum(log_shape))


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
        self.sect_name = kwargs.get("sect_name", "stars.sfh")
        self.redshift = kwargs.get("redshift", 0.0)
        self.today = kwargs.get("today", cosmology.age(self.redshift))
        self.use_transforms = kwargs.get("use_transforms", False)
        self.use_mass_normalization = kwargs.get("use_mass_normalization", True)

        self.free_params = self.free_params.copy()
        self.sfh_smoothness_prior = self._make_sfh_smoothness_prior(
            **kwargs
        )

    # Prior setup
    @staticmethod
    def _make_sfh_smoothness_prior(**kwargs):
        """Construct the optional SFH smoothness prior."""
        use_prior = kwargs.get("use_sfh_smoothness_prior", False)

        if not use_prior:
            return None

        prior_type = str(
            kwargs.get(
                "sfh_smoothness_prior_type",
                kwargs.get("sfh_smoothness_prior", "robust_time_curvature"),
            )
        ).strip().lower()

        if prior_type in {
            "robust_time_curvature",
            "time_curvature",
            "robust",
        }:
            sigma_dex = float(kwargs.get("sfh_smoothness_sigma_dex", 0.3))
            dof = float(kwargs.get("sfh_smoothness_dof", 3.0))
            relative_floor = float(
                kwargs.get("sfh_smoothness_relative_floor", 1e-4)
            )
            logger.info(
                "Enabling robust physical-time SFH curvature prior with "
                "sigma_dex=%s, dof=%s, relative_sfr_floor=%s",
                sigma_dex,
                dof,
                relative_floor,
            )
            return SFHRobustTimeCurvaturePrior(
                sigma_dex=sigma_dex,
                dof=dof,
                relative_sfr_floor=relative_floor,
            )

        if prior_type in {
            "legacy_index_gaussian",
            "index_gaussian",
            "legacy",
        }:
            sigma_dex = float(kwargs.get("sfh_smoothness_sigma_dex", 0.5))
            order = int(kwargs.get("sfh_smoothness_order", 2))
            min_sfr = float(kwargs.get("sfh_smoothness_min_sfr", 1e-12))
            logger.info(
                "Enabling legacy index-based Gaussian SFH smoothness prior "
                "with sigma_dex=%s, order=%s, min_sfr=%s",
                sigma_dex,
                order,
                min_sfr,
            )
            return SFHSmoothnessPrior(
                sigma_dex=sigma_dex,
                order=order,
                min_sfr=min_sfr,
            )

        raise ValueError(
            "Unknown sfh_smoothness_prior_type "
            f"{prior_type!r}; expected 'robust_time_curvature' or "
            "'legacy_index_gaussian'."
        )

    @property
    def use_sfh_smoothness_prior(self):
        """Whether the smoothness prior is enabled."""
        return self.sfh_smoothness_prior is not None

    def evaluate_sfh_smoothness_prior(
        self,
        mass_per_bin,
        time_edges,
    ) -> float:
        """Evaluate the optional SFH smoothness prior.

        Parameters
        ----------
        mass_per_bin
            Mass or mass fraction formed in each disjoint time bin.
        time_edges
            Increasing cosmic-time bin edges. The preferred unit is Gyr.

        Returns
        -------
        float
            Additive log-prior contribution.
        """
        if self.sfh_smoothness_prior is None:
            return 0.0

        return self.sfh_smoothness_prior(
            mass_per_bin=mass_per_bin,
            time_edges=time_edges,
        )

    # --- Transform hooks ---
    def to_physical(self, latent):
        """Map latent parameters to physical space (default: identity)."""
        return latent

    def to_latent(self, physical):
        """Map physical parameters back to latent space (default: identity)."""
        return physical

    def make_ini(self, ini_file, mode="a"):
        """Create a cosmosis .ini file.

        Parameters
        ----------
        ini_file : str
            Path to the output .ini file.
        mode : str, optional, default="a"
            Mode for opening the file.
        """
        logger.info("Making ini file: %s", ini_file)

        free_params = self.free_params.copy()

        if self.use_transforms:
            logger.info("transforming default SFH values into latent variables")
            if getattr(self, "sfh_bin_keys", None):
                sfh_values = self.to_latent([free_params[k][1] for k in self.sfh_bin_keys])
                for key, val in zip(self.sfh_bin_keys, sfh_values):
                    free_params[key] = [-5, val, 5]

        with open(ini_file, mode, encoding="utf-8") as file:
            file.write(f"; Default prior file for SFH model: {str(self.__class__)}\n")
            file.write(f"; use_transforms: {str(self.use_transforms)}\n")
            file.write(f"[{self.sect_name}]\n")
            for key, val in free_params.items():
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
        #TODO: fetch from self.model.parameters_recursive once val ranges
        # are handled
        db = DataBlock.from_dict({self.sect_name: free_params})
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
        """Set the parameter keys associated with the SFH bins."""
        self._sfh_bin_keys = value

    def get_sfh_parameters_array(self, datablock: DataBlock, dtype=float):
        """Get an array containing the values of each bin of the SFH.

        Parameters
        ----------
        datablock : DataBlock
            The datablock containing the values of each parameter of the SFH.
        """
        return np.array(
            [datablock[self.sect_name, key] for key in self.sfh_bin_keys], dtype=dtype
        )


class FixedTimeSFH(ZPowerLawMixin, SFHBase, PieceWiseSFHMixin):
    """A SFH model with fixed time bins.

    Description
    -----------
    The SFH of a galaxy is modelled as a stepwise function where the free
    parameters correspond to the mass fraction formed on each bin.

    The default starting point for uniform priors corresponds to the mass
    fraction formed assuming a constant star formation history.

    Attributes
    ----------
    lookback_time : astropy.units.Quantity
        Lookback time bin edges.
    """

    def __init__(self, lookback_time_bins, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.use_transforms:
            self.use_mass_normalization = False
        logger.info("Initialising FixedTimeSFH model")
        # From the begining of the Universe to the present date
        self.lookback_time = check_unit(
            np.sort(lookback_time_bins)[::-1], u.Gyr
        )
        use_mass_normalization = False
        # Add the present time as the last bin
        self.lookback_time = np.insert(
            self.lookback_time,
            self.lookback_time.size,
            0 << self.lookback_time.unit,
        )

        self.time = self.today - self.lookback_time
        if (self.time < 0).any():
            logger.warning("lookback time bin larger than the age of the Universe")

        logm_min = kwargs.get("logmass_min", 1.0)
        logm_max = kwargs.get("logmass_max", 12.0)
        logger.info("Setting up free parameters")
        logger.info("Minimum log(M/Msun)=%s", logm_min)
        logger.info("Maximum log(M/Msun)=%s", logm_max)
        self.sfh_bin_keys = []
        for lbt in self.lookback_time[:-1].to_value("Gyr"):
            # Initialise parameters assuming a constant star formation history
            k = f"logmass_at_{lbt:.3f}"
            self.sfh_bin_keys.append(k)
            self.free_params[k] = [logm_min,
                                #    self.today._to_value("Gyr") / lbt,
                                   6.0,
                                   logm_max]

        # Initialise PST
        self.model = cem.TabularCEM_ZPowerLaw(
            times=self.time,
            masses=np.ones(self.time.size) << u.Msun,
            today=self.today,
            mass_today=1 << u.Msun,
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02)
            << u.dimensionless_unscaled,
            alpha_powerlaw=kwargs.get("alpha_powerlaw", 0.0),
        )

    def parse_datablock(self, datablock: DataBlock):
        """Update the fixed-time SFH model from a CosmoSIS DataBlock."""
        sampled_masses = self.get_sfh_parameters_array(datablock)

        if self.use_transforms:
            # Fractions formed in each disjoint time bin.
            mass_per_bin = self.to_physical(sampled_masses)
        else:
            # Absolute mass formed in each disjoint time bin.
            mass_per_bin = 10.0**sampled_masses

        cumulative = np.insert(
            np.cumsum(mass_per_bin),
            0,
            0.0,
        )

        log_prior = self.evaluate_sfh_smoothness_prior(
            mass_per_bin=mass_per_bin,
            time_edges=self.time.to_value("Gyr"),
        )

        if not np.isfinite(log_prior):
            return 0, -1e20

        self.model.table_mass = cumulative << u.Msun
        self.model.alpha_powerlaw = datablock[
            self.sect_name,
            "alpha_powerlaw",
        ]
        self.model.ism_metallicity_today = (
            datablock[
                self.sect_name,
                "ism_metallicity_today",
            ]
            << u.dimensionless_unscaled
        )

        return 1, log_prior

    def to_latent(self, physical):
        """
        Inverse of the softmax branch (up to an additive constant).
        We center the log-fractions to have zero mean for determinism.
        """
        frac = np.asarray(physical, dtype=float)
        if frac.ndim != 1:
            raise ValueError("Expected 1D array of mass fractions.")
        if not np.isclose(frac.sum(), 1.0, atol=1e-6):
            raise ValueError("Mass fractions must sum to 1 to invert softmax.")
        log_frac = np.log(frac)
        return log_frac - log_frac.mean()

    def to_physical(self, latent):
        """Softmax-transform latent masses into fractions that sum to 1."""
        if self.use_transforms:
            return _softmax(latent)
        return np.asarray(latent, dtype=float)


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
        logger.info("Initialising FixedGrid-sSFR-SFH model")
        self.lookback_time = check_unit(np.sort(lookback_time)[::-1], u.Gyr)

        # Initialise the PST model
        self.model = cem.CC25TabularCEM(
            today=self.today,
            mass_today= 1 << u.Msun,
            tau_ssfr=self.lookback_time,
            ssfr=np.ones(self.lookback_time.size) << 1 / u.Gyr,
            ism_metallicity_today = 0.02 << u.dimensionless_unscaled,
            alpha_powerlaw = 1.0 << u.dimensionless_unscaled
        )
        self.model.tau_ssfr.fixed = True
        self.model.today.fixed = True
        self.model.mass_today.fixed = True

        self.sfh_bin_keys = []
        self.max_ssfr_logyr = np.zeros(self.lookback_time.size)
        for ith, lbt in enumerate(self.lookback_time.to_value("yr")):
            # This is the name of the value sampled by CosmoSIS
            k = f"logssfr_over_{np.log10(lbt):.2f}_logyr"
            self.sfh_bin_keys.append(k)
            # Maximum value of the sSFR
            max_logssfr = np.log10(1 / lbt)
            self.max_ssfr_logyr[ith] = max_logssfr
            if not self.use_transforms:
                self.free_params[k] = [
                    -14.0,
                    np.log10(1 / self.today.to_value("yr")),
                    max_logssfr,
                ]
            else:
                self.free_params[k] = [-10.0, 0.0, 10.0]

        # log(tau1 / tau2) where tau1 > tau2
        self.delta_logtau = - np.diff(np.log10(self.lookback_time.to_value("yr")))

    def parse_datablock(self, datablock: DataBlock):
        """Update the fixed-time sSFR model from a CosmoSIS DataBlock."""
        ssfr_over_last = self.get_sfh_parameters_array(datablock)

        if self.use_transforms:
            log_ssfr = self.to_physical(ssfr_over_last)
        else:
            if np.any(ssfr_over_last > self.max_ssfr_logyr):
                return 0, -1e20 #0**np.max(ssfr_over_last - self.max_ssfr_logyr)
            # log(ssfr2 / ssfr_1) < log(tau1 / tau2) for tau1 > tau2
            elif np.any(np.diff(ssfr_over_last) >= self.delta_logtau):
                return 0, -1e20

            log_ssfr = ssfr_over_last

        ssfr = 10.0**log_ssfr
        cumulative_mass = 1.0 - self.lookback_time.to_value("yr") * ssfr
        mass_edges = np.concatenate(([0.0], cumulative_mass, [1.0]))
        time_edges = np.concatenate(
            (
                [0.0],
                self.today.to_value("Gyr") - self.lookback_time.to_value("Gyr"),
                [self.today.to_value("Gyr")],
            )
        )
        log_prior = self.evaluate_sfh_smoothness_prior(
            mass_per_bin=np.diff(mass_edges),
            time_edges=time_edges,
        )
        if not np.isfinite(log_prior):
            return 0, -1e20

        self.model.ssfr = ssfr << 1 / u.yr
        # Update the chemical evolution parameters
        self.model.alpha_powerlaw.set(datablock[self.sect_name, "alpha_powerlaw"])
        self.model.ism_metallicity_today.set(
            datablock[self.sect_name, "ism_metallicity_today"])
        return 1, log_prior

    def to_physical(self, latent):
        """Return log10 sSFR over each timescale.

        If transforms are enabled, map latent vars -> softmax increments -> cumulative
        mass, then convert to average sSFR over each interval.
        """
        latent = np.asarray(latent, dtype=float)
        if self.use_transforms:
            # Map unconstrained latents to positive fractions that sum to 1
            increments = _softmax(latent)
            ssfr = (1 - np.cumsum(increments)) / self.lookback_time.to_value("yr")
            ssfr = np.clip(ssfr, 1e-20, None)
            return np.log10(ssfr)
        return latent

    def to_latent(self, physical):
        """Inverse mapping from log10 sSFR back to latent (softmax) space."""
        physical = np.asarray(physical, dtype=float)
        if not self.use_transforms:
            return physical
        lt_yr = self.lookback_time.to_value("yr")
        ssfr = 10**physical
        cumulative = 1 - lt_yr * ssfr
        cumulative = np.insert(cumulative, [0, cumulative.size], [0.0, 1.0])
        if (np.diff(cumulative) <= 0).any():
            raise ValueError("Provided sSFR yields non-monotonic cumulative mass.")
        # Rescale cumulative to [0, 1] and drop duplicate final edge
        cumulative = (cumulative - cumulative[0]) / (cumulative[-1] - cumulative[0])
        increments = np.diff(cumulative)[:-1]
        increments /= increments.sum()
        log_inc = np.log(increments)
        return log_inc - log_inc.mean()


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
        logger.info("Initialising FixedMassFracSFH model")

        mass_fraction = np.sort(mass_fraction)
        self.mass_fraction = mass_fraction.copy()

        self.sfh_bin_keys = []
        for frc in mass_fraction:
            k = f"t_at_frac_{frc:.4f}"
            self.sfh_bin_keys.append(k)
            if not self.use_transforms:
                self.free_params[k] = [
                    1e-3,
                    frc * self.today.to_value("Gyr"),
                    self.today.to_value("Gyr") * 0.999,
                ]
            else:
                self.free_params[k] = [-10.0, 0.0, 10.0]

        if self.use_transforms:
            logger.info("Using transforms to enforce monotonicity and bounds on time bins.")
            mass_fraction = mass_fraction[:-1]

        self.model = cem.TabularMassFracCEM(
            mass_frac=mass_fraction,
            times=np.ones(mass_fraction.size),
            today=self.today,
            mass_today=1 << u.Msun,
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02)
            << u.dimensionless_unscaled,
            alpha_powerlaw=kwargs.get("alpha_powerlaw", 0.0),
        )

    def get_prior_representation(self, times):
        """Return disjoint bin masses and time edges for the prior."""
        times = np.asarray(times, dtype=float)

        today = self.today.to_value("Gyr")
        # Fractions associated with the sampled time anchors.
        cumulative_mass = np.asarray(
            self.mass_fraction,
            dtype=float,
        )
        # Include the initial and final mass boundaries.
        mass_edges = np.concatenate(
            ([0.0], cumulative_mass, [1.0])
        )
        # Include the beginning of the Universe and observation time.
        time_edges = np.concatenate(
            ([0.0], times, [today])
        )
        mass_per_bin = np.diff(mass_edges)

        if time_edges.size != mass_per_bin.size + 1:
            raise ValueError(
                "The number of mass-fraction anchors does not match "
                "the number of time anchors."
            )

        return mass_per_bin, time_edges

    def parse_datablock(self, datablock: DataBlock):
        """Update the fixed-mass-fraction SFH model."""
        sampled_times = self.get_sfh_parameters_array(datablock)

        self.model.alpha_powerlaw = datablock[
            self.sect_name,
            "alpha_powerlaw",
        ]
        self.model.ism_metallicity_today = (
            datablock[
                self.sect_name,
                "ism_metallicity_today",
            ]
            << u.dimensionless_unscaled
        )

        if self.use_transforms:
            times = self.to_physical(sampled_times)
        else:
            times = np.asarray(sampled_times, dtype=float)

        today = self.today.to_value("Gyr")

        # Validate physical time anchors.
        complete_edges = np.concatenate(
            ([0.0], times, [today])
        )
        delta_t = np.diff(complete_edges)

        if np.any(delta_t <= 0):
            return 0, -1e20

        mass_per_bin, time_edges = self.get_prior_representation(
            times
        )

        log_prior = self.evaluate_sfh_smoothness_prior(
            mass_per_bin=mass_per_bin,
            time_edges=time_edges,
        )

        if not np.isfinite(log_prior):
            return 0, -1e20

        if self.use_transforms:
            # Check exactly which boundaries TabularMassFracCEM expects.
            model_times = np.insert(times, 0, 0.0)

            self.model._times = Parameter(
                model_times << u.Gyr,
                fixed=False,
                doc="Observing-time SFH anchors",
            )
        else:
            self.model.times = times << u.Gyr

        return 1, log_prior

    def to_physical(self, latent):
        """Map unconstrained latents to strictly increasing times (Gyr)."""
        if self.use_transforms:
            # Softmax transform
            time_frac = _softmax(latent)
            times = np.cumsum(time_frac) * self.today.to_value("Gyr")
            return times
        return np.asarray(latent, dtype=float)

    def to_latent(self, physical):
        """Inverse of the exp/cumsum mapping (up to normalisation)."""
        if self.use_transforms:
            times = np.asarray(physical, dtype=float)
            validate_monotonic(times, strict=True, name="times")
            deltas = np.diff(np.insert(times, 0, 0))
            return np.log(deltas)
        return np.asarray(physical, dtype=float)

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
        logger.info("Initialising ExponentialSFH model")
        self.time = kwargs.get("time")
        if self.time is None:
            self.time = self.today - np.geomspace(1e-5, 1, 200) * self.today
        self.time = np.sort(self.time)

        # Initialise the free parameter
        self.free_params["logtau"] = kwargs.get("logtau", [-1, 0.5, 1.7])

        self.model = cem.TabularCEM_ZPowerLaw(
            times=self.time,
            today=self.today,
            mass_today=1 << u.Msun,
            masses=np.ones(self.time.size) << u.Msun,
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02)
            << u.dimensionless_unscaled,
            alpha_powerlaw=kwargs.get("alpha_powerlaw", 0.0),
        )

    def parse_datablock(self, datablock: DataBlock):
        """Update the exponential SFH model from a CosmoSIS DataBlock."""
        tau = 10 ** datablock[self.sect_name, "logtau"]
        mass = 1 - np.exp(-self.time.to_value("Gyr") / tau)
        self.model.table_mass = mass / mass[-1] << u.Msun
        self.model.alpha_powerlaw = datablock[self.sect_name, "alpha_powerlaw"]
        self.model.ism_metallicity_today = (
            datablock[self.sect_name, "ism_metallicity_today"] << u.dimensionless_unscaled
        )
        return 1, 0.0


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
        logger.info("Initialising DelayedTauSFH model")
        # Initialise the free parameter
        self.free_params["logtau"] = kwargs.get("logtau", [-1, 0.5, 1.7])

        self.model = cem.ExponentialDelayedZPowerLawCEM(
            today=self.today,
            mass_today=1 << u.Msun,
            tau=1 << u.Gyr,
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02)
            << u.dimensionless_unscaled,
            alpha_powerlaw=kwargs.get("alpha_powerlaw", 0.0),
        )

    def parse_datablock(self, datablock: DataBlock):
        """Update the delayed-tau SFH model from a CosmoSIS DataBlock."""
        self.model = cem.ExponentialDelayedZPowerLawCEM(
            today=self.today,
            mass_today=1.0 << u.Msun,
            tau=10 ** datablock[self.sect_name, "logtau"],
            alpha_powerlaw=datablock[self.sect_name, "alpha_powerlaw"],
            ism_metallicity_today=datablock[self.sect_name, "ism_metallicity_today"]
            << u.dimensionless_unscaled,
        )
        return 1, 0.0


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
        logger.info("Initialising DelayedTauQuenchedSFH model")
        # Initialise the free parameter
        self.free_params["logtau"] = kwargs.get("logtau", [-1, 0.5, 1.7])
        self.free_params["quenching_time"] = kwargs.get(
            "quenching_time", [0, self.today / 2, self.today]
        )

        self.model = cem.ExponentialDelayedQuenchedCEM(
            today=self.today,
            mass_today=1 << u.Msun,
            tau=1 << u.Gyr,
            quenching_time=1 << u.Gyr,
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02)
            << u.dimensionless_unscaled,
            alpha_powerlaw=kwargs.get("alpha_powerlaw", 0.0),
        )

    def parse_datablock(self, datablock: DataBlock):
        """Update the quenched delayed-tau SFH model from a CosmoSIS DataBlock."""
        self.model = cem.ExponentialDelayedQuenchedCEM(
            today=self.today,
            mass_today=1.0 << u.Msun,
            tau=10 ** datablock[self.sect_name, "logtau"],
            quenching_time=datablock[self.sect_name, "quenching_time"],
            alpha_powerlaw=datablock[self.sect_name, "alpha_powerlaw"],
            ism_metallicity_today=datablock[self.sect_name, "ism_metallicity_today"]
            << u.dimensionless_unscaled,
        )
        return 1, 0.0


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
        logger.info("Initialising LogNormalSFH model")
        self.model = cem.LogNormalZPowerLawCEM(
            today=self.today,
            mass_today=1.0 << u.Msun,
            alpha_powerlaw=kwargs.get("alpha_powerlaw", 0.0),
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02)
            << u.dimensionless_unscaled,
            t0=1.0,
            scale=1.0,
        )

    def parse_datablock(self, datablock: DataBlock):
        """Update the log-normal SFH model from a CosmoSIS DataBlock."""
        self.model = cem.LogNormalZPowerLawCEM(
            today=self.today,
            mass_today=1.0 << u.Msun,
            alpha_powerlaw=datablock[self.sect_name, "alpha_powerlaw"],
            ism_metallicity_today=datablock[self.sect_name, "ism_metallicity_today"]
            << u.dimensionless_unscaled,
            t0=datablock[self.sect_name, "t0"] << u.Gyr,
            scale=datablock[self.sect_name, "scale"],
        )
        return 1, 0.0


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
        logger.info("Initialising LogNormalQuenched model")
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

        self.model = cem.LogNormalQuenchedCEM(
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
        """Update the quenched log-normal SFH model from a CosmoSIS DataBlock."""
        self.model = cem.LogNormalQuenchedCEM(
            today=self.today,
            mass_today=1.0 << u.Msun,
            alpha_powerlaw=datablock[self.sect_name, "alpha_powerlaw"],
            ism_metallicity_today=datablock[self.sect_name, "ism_metallicity_today"]
            << u.dimensionless_unscaled,
            t0=datablock[self.sect_name, "t0"] << u.Gyr,
            scale=datablock[self.sect_name, "scale"],
            quenching_time=datablock[self.sect_name, "quenching_time"] << u.Gyr,
        )
        return 1, 0.0

class BetaSFH(ZPowerLawMixin, SFHBase):
    """An analytical beta SFH model.

    Description
    -----------
    The SFH of a galaxy is modelled as a beta function.

    Attributes
    ----------
    time : astropy.units.Quantity
        Time bins to evaluate the SFH.
    lookback_time : astropy.units.Quantity

    """

    free_params = {
        "alpha_powerlaw": [0, 1, 10],
        "ism_metallicity_today": [0.005, 0.01, 0.08],
        "alpha": [0.1, 2.0, 10.0],
        "beta": [0.1, 2.0, 10.0],
        "t_start": [0.0, 0.5, 5.0],
        "t_end": [0.5, 5.0, 15.0],
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Initialising BetaSFH model")

        self.free_params["t_start"] = kwargs.get(
            "t_start", [0.0, min(0.1, 0.5 * self.today.to_value("Gyr")),
                        self.today.to_value("Gyr")]
        )
        self.free_params["t_end"] = kwargs.get(
            "t_end", [0.5 * self.today.to_value("Gyr"), self.today.to_value("Gyr"),
                      2 * self.today.to_value("Gyr")]
        )

        self.model = cem.BetaZPowerLawCEM(
            today=self.today,
            mass_today=1.0 << u.Msun,
            alpha_powerlaw=kwargs.get("alpha_powerlaw", 0.0),
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02)
            << u.dimensionless_unscaled,
            alpha=1.0,
            beta=1.0,
            t_start=0 << u.Gyr,
            t_end=self.today,
        )

    def parse_datablock(self, datablock: DataBlock):
        """Update the beta SFH model from a CosmoSIS DataBlock."""

        t_start = datablock[self.sect_name, "t_start"]
        t_end = datablock[self.sect_name, "t_end"]
        if t_start >= t_end:
            return 0, t_start - t_end
    
        self.model = cem.BetaZPowerLawCEM(
            today=self.today,
            mass_today=1.0 << u.Msun,
            alpha_powerlaw=datablock[self.sect_name, "alpha_powerlaw"],
            ism_metallicity_today=datablock[self.sect_name, "ism_metallicity_today"]
            << u.dimensionless_unscaled,
            alpha=datablock[self.sect_name, "alpha"], 
            beta=datablock[self.sect_name, "beta"],
            t_start=t_start << u.Gyr,
            t_end=t_end << u.Gyr,
        )
        return 1, 0.0

# Mr Krtxo \(ﾟ▽ﾟ)/
