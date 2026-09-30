"""
Star formation history fitting module
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import lgamma
import numpy as np
from numpy.polynomial import Polynomial
from astropy import units as u

from cosmosis import DataBlock
from pst import cem
from pst.utils import check_unit
from besta.config import cosmology
from besta.logging import get_logger

logger = get_logger(__name__)

# Latent variables and transformations

def _softmax(x):
    """Numerically stable softmax.
    
    The softmax is defined as:

    .. math::

        \mathrm{softmax}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}.

    """
    x = np.asarray(x, dtype=float)
    x_shift = x - np.max(x)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x)

def _sigmoid(x):
    """Simple sigmoid to clamp values into (0, 1).
    
    This function maps any real number to the interval (0, 1).
    The sigmoid is defined as:

    .. math::

        \mathrm{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.

    """
    return 1.0 / (1.0 + np.exp(-x))

def _logit(x):
    """Inverse of sigmoid on (0,1).
    
    The inverse sigmoid, or logit, is defined as:

    .. math::

        \mathrm{logit}(x) = \log\left(\frac{x}{1 - x}\right).

    """
    x = np.asarray(x, dtype=float)
    if np.any((x <= 0) | (x >= 1)):
        raise ValueError("Logit is only defined for values strictly within (0, 1).")
    return np.log(x) - np.log1p(-x)

def _log_sfr_jumps_to_time_fractions(jumps, delta_frac):
    r"""Map adjacent log-SFR jumps to ordered anchor positions in (0, 1).

    Parameters
    ----------
    jumps : array-like, shape (..., n)
        :math:`r_i = \log_{10}{\rm SFR}_i - \log_{10}{\rm SFR}_{i+1}`, where
        bin :math:`i` is the interval between anchors :math:`i - 1` and
        :math:`i` (bin 1 is the oldest). Positive values mean a declining SFH.
    delta_frac : array-like, shape (n + 1,)
        Mass fraction formed in each of the ``n + 1`` bins.

    Returns
    -------
    np.ndarray, shape (..., n)
        Cumulative time fractions of the ``n`` anchors. The bin widths are
        :math:`\Delta t_i \propto \Delta f_i / {\rm SFR}_i`, normalised to 1.
    """
    jumps = np.asarray(jumps, dtype=float)
    zeros = np.zeros(jumps.shape[:-1] + (1,))
    log_sfr = np.concatenate((zeros, -np.cumsum(jumps, axis=-1)), axis=-1)
    log_widths = np.log10(delta_frac) - log_sfr
    log_widths -= np.max(log_widths, axis=-1, keepdims=True)
    widths = 10.0 ** log_widths
    widths /= np.sum(widths, axis=-1, keepdims=True)
    return np.cumsum(widths[..., :-1], axis=-1)


def _time_fractions_to_log_sfr_jumps(time_frac, delta_frac):
    """Inverse of :func:`_log_sfr_jumps_to_time_fractions`."""
    time_frac = np.asarray(time_frac, dtype=float)
    zeros = np.zeros(time_frac.shape[:-1] + (1,))
    edges = np.concatenate((zeros, time_frac, zeros + 1.0), axis=-1)
    log_sfr = np.log10(delta_frac) - np.log10(np.diff(edges, axis=-1))
    return -np.diff(log_sfr, axis=-1)


def _validate_monotonic(array, *, strict=True, name="array"):
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


class _OrderedUniformTransform:
    r"""Map a unit hypercube to a uniform ordered region.

    The target region is

    .. math::

        0 \leq x_1 < x_2 < \cdots < x_n,
        \qquad x_i \leq w_i,

    for strictly increasing positive upper bounds ``w_i``.
    
    The mapping is the inverse Rosenblatt transform of the uniform density on
    this region. It generalises the equal-bound stick-breaking transform used by
    :class:`FixedMassFracSFH`.
    """

    def __init__(self, upper_bounds):

        # Set the upper bounds w_i
        upper_bounds = np.asarray(upper_bounds, dtype=float)
        if upper_bounds.ndim != 1 or upper_bounds.size == 0:
            raise ValueError("upper_bounds must be a non-empty 1D array.")
        if np.any(~np.isfinite(upper_bounds)) or np.any(upper_bounds <= 0.0):
            raise ValueError("upper_bounds must be finite and positive.")
        if np.any(np.diff(upper_bounds) <= 0.0):
            raise ValueError("upper_bounds must be strictly increasing.")

        self.upper_bounds = upper_bounds
        self._volume_pieces = [None]
        self._total_volumes = [1.0]

        # V_k(x) is the volume of 0 <= x_1 < ... < x_k <= x, subject
        # to x_i <= w_i.  Recursively,
        # V_k(x) = integral_0^min(x, w_k) V_{k-1}(t) dt.
        for order in range(1, upper_bounds.size + 1):
            boundaries = np.concatenate(([0.0], upper_bounds[:order]))
            pieces = []
            cumulative_volume = 0.0

            for interval, (left, right) in enumerate(
                zip(boundaries[:-1], boundaries[1:])
            ):
                if order == 1:
                    integrand = Polynomial([1.0])
                elif interval < order - 1:
                    integrand = self._volume_pieces[order - 1][interval]
                else:
                    integrand = Polynomial([self._total_volumes[order - 1]])

                antiderivative = integrand.integ()
                volume = antiderivative + (
                    cumulative_volume - antiderivative(left)
                )
                pieces.append(volume)
                cumulative_volume = float(volume(right))

            self._volume_pieces.append(tuple(pieces))
            self._total_volumes.append(cumulative_volume)

    def _volume(self, order, value):
        """Evaluate ``V_order(value)``."""
        value = float(value)
        if value <= 0.0:
            return 0.0
        if value >= self.upper_bounds[order - 1]:
            return self._total_volumes[order]

        interval = int(
            np.searchsorted(
                self.upper_bounds[:order],
                value,
                side="right",
            )
        )
        return float(self._volume_pieces[order][interval](value))

    def _inverse_volume(self, order, probability, conditional_upper):
        """Invert a conditional ordered-coordinate CDF by bisection."""
        support_upper = min(
            float(conditional_upper),
            self.upper_bounds[order - 1],
        )
        total_volume = self._volume(order, support_upper)
        target_volume = float(probability) * total_volume

        if target_volume <= 0.0:
            return 0.0
        if target_volume >= total_volume:
            return support_upper

        lower = 0.0
        upper = support_upper
        for _ in range(48):
            midpoint = 0.5 * (lower + upper)
            if self._volume(order, midpoint) < target_volume:
                lower = midpoint
            else:
                upper = midpoint
        return 0.5 * (lower + upper)

    def to_ordered(self, latent):
        """Map independent unit-uniform variables to the ordered region."""
        latent = np.asarray(latent, dtype=float)
        if latent.shape != self.upper_bounds.shape:
            raise ValueError(
                f"Expected {self.upper_bounds.size} latent variables."
            )
        if np.any(~np.isfinite(latent)) or np.any(
            (latent < 0.0) | (latent > 1.0)
        ):
            raise ValueError("Latent variables must lie in [0, 1].")

        ordered = np.empty_like(latent)
        conditional_upper = self.upper_bounds[-1]
        for order in range(latent.size, 0, -1):
            ordered[order - 1] = self._inverse_volume(
                order,
                latent[order - 1],
                conditional_upper,
            )
            conditional_upper = ordered[order - 1]
        return ordered

    def to_unit(self, ordered):
        """Map an interior point of the ordered region to the unit cube."""
        ordered = np.asarray(ordered, dtype=float)
        if ordered.shape != self.upper_bounds.shape:
            raise ValueError(
                f"Expected {self.upper_bounds.size} ordered variables."
            )
        if np.any(~np.isfinite(ordered)) or np.any(ordered < 0.0) or np.any(
            ordered > self.upper_bounds
        ):
            raise ValueError("Ordered variables lie outside their bounds.")
        _validate_monotonic(
            ordered,
            strict=True,
            name="ordered variables",
        )

        latent = np.empty_like(ordered)
        conditional_upper = self.upper_bounds[-1]
        for order in range(ordered.size, 0, -1):
            denominator = self._volume(order, conditional_upper)
            latent[order - 1] = (
                self._volume(order, ordered[order - 1]) / denominator
            )
            conditional_upper = ordered[order - 1]
        return latent

    # --- Vectorised versions for batch processing -----------------------------

    def _volume_batch(self, order, values):
        """Vectorised :meth:`_volume` (NaN in, NaN out)."""
        values = np.asarray(values, dtype=float)
        out = np.full(values.shape, np.nan)
        finite = np.isfinite(values)
        below = finite & (values <= 0.0)
        above = finite & (values >= self.upper_bounds[order - 1])
        inside = finite & ~(below | above)
        out[below] = 0.0
        out[above] = self._total_volumes[order]
        if np.any(inside):
            inner = values[inside]
            interval = np.searchsorted(
                self.upper_bounds[:order], inner, side="right")
            result = np.empty_like(inner)
            for index, piece in enumerate(self._volume_pieces[order]):
                selected = interval == index
                if np.any(selected):
                    result[selected] = piece(inner[selected])
            out[inside] = result
        return out

    def to_ordered_batch(self, latent):
        """Vectorised :meth:`to_ordered` for an ``(n_samples, n)`` array.

        Uses the same 48-step bisection as the scalar method, run on all
        samples at once. Rows with values outside ``[0, 1]`` or non-finite
        values are returned as NaN.
        """
        latent = np.asarray(latent, dtype=float)
        if latent.ndim != 2 or latent.shape[1] != self.upper_bounds.size:
            raise ValueError(
                f"Expected an array of shape (n_samples, {self.upper_bounds.size}).")
        ordered = np.full(latent.shape, np.nan)
        valid = np.all(
            np.isfinite(latent) & (latent >= 0.0) & (latent <= 1.0), axis=1)
        if not np.any(valid):
            return ordered

        values = latent[valid]
        result = np.empty_like(values)
        conditional_upper = np.full(values.shape[0], self.upper_bounds[-1])
        for order in range(values.shape[1], 0, -1):
            support_upper = np.minimum(
                conditional_upper, self.upper_bounds[order - 1])
            total_volume = self._volume_batch(order, support_upper)
            target_volume = values[:, order - 1] * total_volume

            lower = np.zeros_like(support_upper)
            upper = support_upper.copy()
            for _ in range(48):
                midpoint = 0.5 * (lower + upper)
                below = self._volume_batch(order, midpoint) < target_volume
                lower = np.where(below, midpoint, lower)
                upper = np.where(below, upper, midpoint)
            solution = 0.5 * (lower + upper)
            solution = np.where(target_volume <= 0.0, 0.0, solution)
            solution = np.where(
                target_volume >= total_volume, support_upper, solution)

            result[:, order - 1] = solution
            conditional_upper = solution
        ordered[valid] = result
        return ordered

    def to_unit_batch(self, ordered):
        """Vectorised :meth:`to_unit` for an ``(n_samples, n)`` array.

        Rows outside the bounds, non-finite or not strictly increasing are
        returned as NaN.
        """
        ordered = np.asarray(ordered, dtype=float)
        if ordered.ndim != 2 or ordered.shape[1] != self.upper_bounds.size:
            raise ValueError(
                f"Expected an array of shape (n_samples, {self.upper_bounds.size}).")
        latent = np.full(ordered.shape, np.nan)
        with np.errstate(invalid="ignore"):
            valid = np.all(
                np.isfinite(ordered) & (ordered >= 0.0)
                & (ordered <= self.upper_bounds), axis=1)
            valid &= np.all(np.diff(ordered, axis=1) > 0, axis=1)
        if not np.any(valid):
            return latent

        values = ordered[valid]
        result = np.empty_like(values)
        conditional_upper = np.full(values.shape[0], self.upper_bounds[-1])
        for order in range(values.shape[1], 0, -1):
            denominator = self._volume_batch(order, conditional_upper)
            result[:, order - 1] = (
                self._volume_batch(order, values[:, order - 1]) / denominator)
            conditional_upper = values[:, order - 1]
        latent[valid] = result
        return latent



# Non-parametric SFH smoothing priors

@dataclass(frozen=True)
class SFHSmoothnessPrior:
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

    - In the case of `dof=1`, the prior is equivalent to a Laplace distribution on the residuals.
    - In the case of `dof=2`, the prior is equivalent to a Cauchy distribution on the residuals.
    - In the case of `dof=3`, the prior is equivalent to a Student-t distribution with 3 degrees of freedom on the residuals.

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
    _defines_latent_free_params = False

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
            "smoothness",
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
            return SFHSmoothnessPrior(
                sigma_dex=sigma_dex,
                dof=dof,
                relative_sfr_floor=relative_floor,
            )

        raise ValueError(
            "Unknown sfh_smoothness_prior_type "
            f"{prior_type!r}; expected 'robust_time_curvature', "
            "'time_curvature', or 'smoothness'."
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

    @staticmethod
    def _as_batch(values):
        """Return ``values`` as a float ``(n_samples, n_parameters)`` array."""
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            values = values[np.newaxis, :]
        if values.ndim != 2:
            raise ValueError(
                "Expected a 1D or 2D array (n_samples, n_parameters).")
        return values

    def to_physical_batch(self, latent):
        """Map many latent samples to physical space.

        Parameters
        ----------
        latent : array_like, shape (n_samples, n_parameters) or (n_parameters,)
            Latent SFH parameters, columns ordered as ``sfh_bin_keys``.

        Returns
        -------
        np.ndarray, shape (n_samples, n_parameters)
            Physical parameters. Rows that cannot be mapped (outside the
            latent support, non-finite) are NaN.

        Notes
        -----
        Without ``use_transforms`` this is the identity. Otherwise this default
        loops over :meth:`to_physical`; models override it with a vectorised
        implementation.
        """
        latent = self._as_batch(latent)
        if not self.use_transforms:
            return latent.copy()
        physical = np.full(latent.shape, np.nan)
        for index, row in enumerate(latent):
            try:
                physical[index] = self.to_physical(row)
            except ValueError:
                pass
        return physical

    def to_latent_batch(self, physical):
        """Map many physical samples back to latent space.

        Inverse of :meth:`to_physical_batch`, with the same conventions
        (2D output, NaN for rows outside the physical support).
        """
        physical = self._as_batch(physical)
        if not self.use_transforms:
            return physical.copy()
        latent = np.full(physical.shape, np.nan)
        for index, row in enumerate(physical):
            try:
                latent[index] = self.to_latent(row)
            except ValueError:
                pass
        return latent

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

        if self.use_transforms and not self._defines_latent_free_params:
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
    parameters correspond to the mean SFR of each bin. The first bin ranges
    from the present time to the first lookback time, the second bin from the
    first to the second, and so on. The last bin ranges from the last lookback
    time to the beginning of the Universe.

    Attributes
    ----------
    lookback_time : astropy.units.Quantity
        Lookback time bin edges.
    """

    def __init__(self, lookback_time_bins, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.use_transforms:
            raise NotImplementedError(
                "FixedTimeSFH does not support latent variable transforms."
            )

        logger.info("Initialising FixedTimeSFH model")
        # From the begining of the Universe to the present date
        self.lookback_time = check_unit(
            np.sort(lookback_time_bins)[::-1], u.Gyr
        )
        self.use_mass_normalization = False
        # Add the present time as the last bin
        self.lookback_time = np.insert(
            self.lookback_time,
            self.lookback_time.size,
            0 << self.lookback_time.unit,
        )

        self.time = self.today - self.lookback_time
        if (self.time < 0).any():
            logger.warning("lookback time bin larger than the age of the Universe")
        self.delta_time = np.diff(self.time.to_value("yr"))

        logsfr_min = kwargs.get("logsfr_min", -5.0)
        logsfr_max = kwargs.get("logsfr_max", 3.0)
        logger.info("Setting up free parameters")
        logger.info("Minimum log(SFR)=%s", logsfr_min)
        logger.info("Maximum log(SFR)=%s", logsfr_max)
        self.sfh_bin_keys = []
        for lbt in self.lookback_time[:-1].to_value("Gyr"):
            # Initialise parameters assuming a constant star formation history
            k = f"logsfr_at_{lbt:.3f}"
            self.sfh_bin_keys.append(k)
            self.free_params[k] = [logsfr_min,
                                #    self.today._to_value("Gyr") / lbt,
                                   0.0,
                                   logsfr_max]

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
        self.model.times.fixed = True

    def parse_datablock(self, datablock: DataBlock):
        """Update the fixed-time SFH model from a CosmoSIS DataBlock."""
        sampled_logsfr = self.get_sfh_parameters_array(datablock)

        # Convert mean SFR per bin into cumulative mass formed in each bin
        mass_per_bin = 10.0**sampled_logsfr * self.delta_time
        cumulative = np.cumsum(mass_per_bin)
        cumulative = np.insert(cumulative, 0, 0.0) << u.Msun

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

    _defines_latent_free_params = True

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
        self.min_ssfr_logyr = np.full(self.lookback_time.size, -14.0)
        self.log_lookback_time_yr = np.log10(
            self.lookback_time.to_value("yr")
        )
        for ith, lbt in enumerate(self.lookback_time.to_value("yr")):
            # This is the name of the value sampled by CosmoSIS
            k = f"logssfr_over_{np.log10(lbt):.2f}_logyr"
            self.sfh_bin_keys.append(k)
            # Maximum value of the sSFR
            max_logssfr = np.log10(1 / lbt)
            self.max_ssfr_logyr[ith] = max_logssfr
            if not self.use_transforms:
                self.free_params[k] = [
                    self.min_ssfr_logyr[ith],
                    np.log10(1 / self.today.to_value("yr")),
                    max_logssfr,
                ]
            else:
                self.free_params[k] = [0.0, 0.5, 1.0]

        if self.use_transforms:
            # For z_i = -log10(tau_i * sSFR_i), the direct rectangular
            # log-sSFR prior conditioned on physical cumulative masses is
            # uniform over 0 <= z_1 < ... < z_n <= w_i.  The unequal w_i
            # arise from the shared lower log-sSFR bound.
            ordered_upper = -(
                self.min_ssfr_logyr + self.log_lookback_time_yr
            )
            self._latent_prior_transform = _OrderedUniformTransform(
                ordered_upper
            )
            logger.info(
                "Using a prior-preserving ordered transform for fixed-time "
                "sSFR bins."
            )

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
        """Map unit-cube latents to the conditioned direct log-sSFR prior."""
        latent = np.asarray(latent, dtype=float)
        if self.use_transforms:
            ordered_log_remaining_mass = (
                self._latent_prior_transform.to_ordered(latent)
            )
            return -(
                ordered_log_remaining_mass + self.log_lookback_time_yr
            )
        return latent

    def to_latent(self, physical):
        """Map a physical log-sSFR sequence back to the unit hypercube."""
        physical = np.asarray(physical, dtype=float)
        if not self.use_transforms:
            return physical
        if physical.shape != self.log_lookback_time_yr.shape:
            raise ValueError(
                f"Expected {self.lookback_time.size} log-sSFR values."
            )
        ordered_log_remaining_mass = -(
            physical + self.log_lookback_time_yr
        )
        return self._latent_prior_transform.to_unit(
            ordered_log_remaining_mass
        )

    def to_physical_batch(self, latent):
        """Vectorised :meth:`to_physical` (see :meth:`SFHBase.to_physical_batch`)."""
        latent = self._as_batch(latent)
        if not self.use_transforms:
            return latent.copy()
        ordered_log_remaining_mass = (
            self._latent_prior_transform.to_ordered_batch(latent))
        return -(ordered_log_remaining_mass + self.log_lookback_time_yr)

    def to_latent_batch(self, physical):
        """Vectorised :meth:`to_latent` (see :meth:`SFHBase.to_latent_batch`)."""
        physical = self._as_batch(physical)
        if not self.use_transforms:
            return physical.copy()
        ordered_log_remaining_mass = -(physical + self.log_lookback_time_yr)
        return self._latent_prior_transform.to_unit_batch(
            ordered_log_remaining_mass)


class FixedMassFracSFH(ZPowerLawMixin, SFHBase, PieceWiseSFHMixin):
    r"""A SFH model with fixed mass fraction bins.

    Description
    -----------
    The free parameters are the cosmic times at which given fractions of the
    stellar mass observed today had formed. PST interpolates the cumulative
    mass history through these anchors (plus ``M = 0`` at ``t = 0`` and
    ``M = 1`` today), and the SFR is its derivative.

    All time anchors share the range ``[min_time, today - min_last_interval]``.

    Latent spaces (``use_transforms = True``)
    -----------------------------------------
    The sampled coordinates set the base measure of the prior, since CosmoSIS
    applies a uniform prior over the ``values`` ranges:

    - ``latent_space = "stick_breaking"`` (default): unit-cube latents mapped
      to uniformly distributed ordered times. The prior is the same as
      sampling the times directly (``use_transforms = False``). With unequal
      mass-fraction steps this favours declining SFHs, because every time gap
      has the same expected length.
    - ``latent_space = "log_sfr_jumps"``: the latents are the jumps in
      log10 SFR across each anchor,
      :math:`r_i = \log_{10}{\rm SFR}_i - \log_{10}{\rm SFR}_{i+1}`, with
      uniform priors in ``[-max_dlogsfr, max_dlogsfr]``. The prior is flat in
      log-SFR ratios and centred on a constant SFH (``r = 0``), so it does not
      prefer declining or rising histories. Pair it with
      ``use_sfh_smoothness_prior`` for regularisation.

    Bin widths follow :math:`\Delta t_i \propto \Delta f_i / {\rm SFR}_i` and
    are normalised to ``[min_time, today - min_last_interval]``. For
    ``log_sfr_jumps`` the SFR ratios are therefore defined on that interval.
    They differ from the physical ones only by ``min_time`` in the oldest bin
    and ``min_last_interval`` in the youngest.

    Parameters
    ----------
    mass_fraction : array-like
        Cumulative mass fractions of the time anchors.
    min_last_interval : float, optional
        Minimum time, in Gyr, between the last anchor and the observation.
        It caps the average sSFR over that interval at
        ``(1 - mass_fraction[-1]) / min_last_interval``. Default 1e-4 Gyr
        (0.1 Myr).
    min_time : float, optional
        Earliest cosmic time allowed for the anchors, in Gyr. Default 1e-3.
    latent_space : {"stick_breaking", "log_sfr_jumps"}, optional
        Latent parametrisation used when ``use_transforms`` is ``True``.
        Ignored otherwise. Default ``"stick_breaking"``.
    max_dlogsfr : float, optional
        Half-width, in dex, of the uniform prior on each log-SFR jump
        (``latent_space = "log_sfr_jumps"`` only). Default 2.

    Attributes
    ----------
    mass_fraction : np.ndarray
        SFH mass fractions.
    time_bounds : tuple of float
        ``(min_time, today - min_last_interval)`` in Gyr.
    latent_space : str
        Latent parametrisation (only relevant with ``use_transforms``).
    """

    LATENT_SPACES = ("stick_breaking", "log_sfr_jumps")

    cem_model_class = cem.TabularMassFracCEM
    _defines_latent_free_params = True

    def __init__(self, mass_fraction, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info("Initialising FixedMassFracSFH model")

        mass_fraction = np.sort(np.asarray(mass_fraction, dtype=float))
        self.mass_fraction = mass_fraction.copy()
        self.time_bounds = self._time_bounds(
            kwargs.get("min_time", 1e-3),           # Gyr
            kwargs.get("min_last_interval", 1e-4),  # Gyr
        )
        t_min, t_max = self.time_bounds
        logger.info(
            "Time anchors within [%.4g, %.6g] Gyr; the average sSFR over the "
            "last interval is capped at %.3g / yr",
            t_min, t_max,
            (1.0 - mass_fraction[-1])
            / ((self.today.to_value("Gyr") - t_max) * 1e9),
        )

        # Mass fraction formed in each of the n + 1 bins
        self._delta_frac = np.diff(
            np.concatenate(([0.0], mass_fraction, [1.0])))

        self.latent_space = str(
            kwargs.get("latent_space", "stick_breaking")).strip().lower()
        if self.latent_space not in self.LATENT_SPACES:
            raise ValueError(
                f"Unknown latent_space {self.latent_space!r}; expected one of "
                f"{self.LATENT_SPACES}.")
        self.max_dlogsfr = float(kwargs.get("max_dlogsfr", 2.0))
        if self._uses_log_sfr_jumps:
            if not np.isfinite(self.max_dlogsfr) or self.max_dlogsfr <= 0:
                raise ValueError("max_dlogsfr must be finite and positive.")
            if np.any(self._delta_frac <= 0):
                raise ValueError(
                    "latent_space='log_sfr_jumps' requires mass fractions "
                    "strictly between 0 and 1 and strictly increasing.")

        self.sfh_bin_keys = []
        for frc in mass_fraction:
            k = f"t_at_frac_{frc:.4f}"
            self.sfh_bin_keys.append(k)
            if not self.use_transforms:
                # Start from a constant SFR, mapped strictly inside the bounds.
                self.free_params[k] = [
                    t_min,
                    t_min + frc * (t_max - t_min),
                    t_max,
                ]
            elif self._uses_log_sfr_jumps:
                # Jump across the anchor at this fraction; 0 = constant SFR.
                self.free_params[k] = [
                    -self.max_dlogsfr, 0.0, self.max_dlogsfr]
            else:
                self.free_params[k] = [0.0, 0.5, 1.0]

        if self._uses_log_sfr_jumps:
            logger.info(
                "Sampling the log10 SFR jumps across the anchors, uniform in "
                "[-%s, %s] dex.", self.max_dlogsfr, self.max_dlogsfr)
        elif self.use_transforms:
            logger.info(
                "Using a stick-breaking transform to enforce monotonicity "
                "and bounds on time bins."
            )

        self.model = self.cem_model_class(
            mass_frac=mass_fraction,
            times=np.ones(mass_fraction.size),
            today=self.today,
            mass_today=1 << u.Msun,
            ism_metallicity_today=kwargs.get("ism_metallicity_today", 0.02)
            << u.dimensionless_unscaled,
            alpha_powerlaw=kwargs.get("alpha_powerlaw", 0.0),
        )

    def _time_bounds(self, min_time, min_last_interval):
        """Validate and return ``(min_time, today - min_last_interval)`` in Gyr."""
        today = self.today.to_value("Gyr")
        min_time = float(min_time)
        min_last_interval = float(min_last_interval)
        if not np.isfinite(min_time) or min_time < 0:
            raise ValueError("min_time must be finite and non-negative.")
        if not np.isfinite(min_last_interval) or min_last_interval <= 0:
            raise ValueError("min_last_interval must be finite and positive.")
        t_max = today - min_last_interval
        if t_max <= min_time:
            raise ValueError(
                f"min_time ({min_time} Gyr) + min_last_interval "
                f"({min_last_interval} Gyr) must be smaller than today "
                f"({today:.4f} Gyr).")
        return min_time, t_max

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

        self.model.times = times << u.Gyr

        return 1, log_prior

    @property
    def _uses_log_sfr_jumps(self):
        """Whether the sampled (latent) parameters are log-SFR jumps."""
        return self.use_transforms and self.latent_space == "log_sfr_jumps"

    def _valid_jumps(self, jumps, rtol=0.0):
        """Row mask of jumps that are finite and within ``max_dlogsfr``.

        ``rtol`` widens the range slightly, to absorb round-off when mapping
        times back to jumps at the edge of the prior range.
        """
        limit = self.max_dlogsfr * (1.0 + rtol)
        return np.all(np.isfinite(jumps) & (np.abs(jumps) <= limit), axis=-1)

    def _jumps_to_times(self, jumps):
        t_min, t_max = self.time_bounds
        return t_min + _log_sfr_jumps_to_time_fractions(
            jumps, self._delta_frac) * (t_max - t_min)

    def _times_to_jumps(self, times):
        t_min, t_max = self.time_bounds
        return _time_fractions_to_log_sfr_jumps(
            (times - t_min) / (t_max - t_min), self._delta_frac)

    def to_physical(self, latent):
        """Map latent parameters to strictly increasing times (Gyr)."""
        if self._uses_log_sfr_jumps:
            jumps = np.asarray(latent, dtype=float)
            if not self._valid_jumps(jumps):
                raise ValueError(
                    "log-SFR jumps must be finite and lie in "
                    f"[-{self.max_dlogsfr}, {self.max_dlogsfr}].")
            return self._jumps_to_times(jumps)
        if self.use_transforms:
            if np.any(~np.isfinite(latent)) or np.any(
                (latent < 0.0) | (latent > 1.0)
            ):
                raise ValueError("Stick-breaking variables must lie in [0, 1].")

            # Uniform spacings on the ordered-time simplex are generated by
            # V_i ~ Beta(1, n - i), using the inverse CDF of a unit-uniform
            # latent variable. The unused remainder is the final interval
            # between the last time anchor and today.
            powers = np.arange(latent.size, 0, -1, dtype=float)
            breaks = 1.0 - (1.0 - latent) ** (1.0 / powers)
            remaining = np.concatenate(
                ([1.0], np.cumprod(1.0 - breaks)[:-1])
            )
            time_frac = np.cumsum(remaining * breaks)
            t_min, t_max = self.time_bounds
            return t_min + time_frac * (t_max - t_min)
        return latent

    def to_latent(self, physical):
        """Map strictly increasing times back to the latent variables."""
        if self.use_transforms:
            times = np.asarray(physical, dtype=float)
            _validate_monotonic(times, strict=True, name="times")
            t_min, t_max = self.time_bounds
            if np.any(~np.isfinite(times)) or np.any(times <= t_min) or np.any(
                times >= t_max
            ):
                raise ValueError(
                    f"Times must lie strictly between {t_min} and {t_max} Gyr.")

            if self._uses_log_sfr_jumps:
                jumps = self._times_to_jumps(times)
                if not self._valid_jumps(jumps, rtol=1e-9):
                    raise ValueError(
                        "The times imply log-SFR jumps outside "
                        f"[-{self.max_dlogsfr}, {self.max_dlogsfr}] dex.")
                return np.clip(jumps, -self.max_dlogsfr, self.max_dlogsfr)

            time_frac = (times - t_min) / (t_max - t_min)
            previous = np.concatenate(([0.0], time_frac[:-1]))
            breaks = (time_frac - previous) / (1.0 - previous)
            powers = np.arange(times.size, 0, -1, dtype=float)
            return 1.0 - (1.0 - breaks) ** powers
        return np.asarray(physical, dtype=float)

    def to_physical_batch(self, latent):
        """Vectorised :meth:`to_physical` (see :meth:`SFHBase.to_physical_batch`)."""
        latent = self._as_batch(latent)
        if not self.use_transforms:
            return latent.copy()
        physical = np.full(latent.shape, np.nan)
        if self._uses_log_sfr_jumps:
            valid = self._valid_jumps(latent)
            if np.any(valid):
                physical[valid] = self._jumps_to_times(latent[valid])
            return physical

        valid = np.all(
            np.isfinite(latent) & (latent >= 0.0) & (latent <= 1.0), axis=1)
        if not np.any(valid):
            return physical

        values = latent[valid]
        powers = np.arange(values.shape[1], 0, -1, dtype=float)
        breaks = 1.0 - (1.0 - values) ** (1.0 / powers)
        remaining = np.concatenate(
            (np.ones((values.shape[0], 1)),
             np.cumprod(1.0 - breaks, axis=1)[:, :-1]),
            axis=1)
        time_frac = np.cumsum(remaining * breaks, axis=1)
        t_min, t_max = self.time_bounds
        physical[valid] = t_min + time_frac * (t_max - t_min)
        return physical

    def to_latent_batch(self, physical):
        """Vectorised :meth:`to_latent` (see :meth:`SFHBase.to_latent_batch`)."""
        physical = self._as_batch(physical)
        if not self.use_transforms:
            return physical.copy()
        latent = np.full(physical.shape, np.nan)
        t_min, t_max = self.time_bounds
        with np.errstate(invalid="ignore"):
            valid = np.all(
                np.isfinite(physical) & (physical > t_min) & (physical < t_max),
                axis=1)
            valid &= np.all(np.diff(physical, axis=1) > 0, axis=1)
        if not np.any(valid):
            return latent

        if self._uses_log_sfr_jumps:
            jumps = self._times_to_jumps(physical[valid])
            jumps[~self._valid_jumps(jumps, rtol=1e-9)] = np.nan
            latent[valid] = np.clip(jumps, -self.max_dlogsfr, self.max_dlogsfr)
            return latent

        time_frac = (physical[valid] - t_min) / (t_max - t_min)
        previous = np.concatenate(
            (np.zeros((time_frac.shape[0], 1)), time_frac[:, :-1]), axis=1)
        breaks = (time_frac - previous) / (1.0 - previous)
        powers = np.arange(time_frac.shape[1], 0, -1, dtype=float)
        latent[valid] = 1.0 - (1.0 - breaks) ** powers
        return latent


class FixedMassFracSFH2D(FixedMassFracSFH):
    r"""Fixed-mass-fraction SFH with metallicity scatter at each formation time.

    This wrapper has the same SFH time anchors, transforms, and optional
    smoothness prior as :class:`FixedMassFracSFH`, but uses
    :class:`pst.cem.TabularMassFracCEM2D`.  Its mean enrichment history remains

    .. math::

        \overline{Z}(t) = Z_{\rm today}
        \left(\frac{M_\star(t)}{M_\star({\rm today})}\right)^{\alpha_Z},

    with a log-normal distribution of stellar metallicities around that mean.

    Parameters
    ----------
    mass_fraction : array-like
        Increasing cumulative mass fractions associated with the fitted cosmic
        formation times.
    sigma_log_metallicity : float, optional
        Initial scatter in log10 metallicity at fixed formation time, in dex.
        The default is 0.25 dex. During fitting this value is read from the
        ``stars.sfh`` section of the CosmoSIS DataBlock.
    """

    cem_model_class = cem.TabularMassFracCEM2D
    free_params = {
        **ZPowerLawMixin.free_params,
        "sigma_log_metallicity": [0.0, 0.25, 1.0],
    }

    def __init__(self, mass_fraction, *args, **kwargs):
        super().__init__(mass_fraction, *args, **kwargs)
        self.model.sigma_log_metallicity = kwargs.get(
            "sigma_log_metallicity", 0.25
        )

    def parse_datablock(self, datablock: DataBlock):
        """Update SFH, mean enrichment, and metallicity scatter parameters."""
        self.model.sigma_log_metallicity.set(
            datablock[self.sect_name, "sigma_log_metallicity"],
            validate=True,
        )
        return super().parse_datablock(datablock)


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
            return 0, -1e20

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


# Building SFH models from configuration options

# Options forwarded to the SFH constructor (name, type). Everything else comes
# from ``SFHArgs``.
_SFH_SMOOTHNESS_OPTIONS = (
    ("sfh_smoothness_prior_type", str),
    ("sfh_smoothness_sigma_dex", float),
    ("sfh_smoothness_dof", float),
    ("sfh_smoothness_relative_floor", float),
    ("sfh_smoothness_order", int),
    ("sfh_smoothness_min_sfr", float),
)

# Model options forwarded to the SFH constructor when present (name, type).
_SFH_MODEL_OPTIONS = (
    ("min_last_interval", float),
    ("min_time", float),
    ("latent_space", str),
    ("max_dlogsfr", float),
)


class _OptionReader:
    """Uniform access to CosmoSIS ``SectionOptions`` or plain (ini) dicts.

    Dict keys are matched case-insensitively, as CosmoSIS does.
    """

    def __init__(self, options):
        if hasattr(options, "has_value"):
            self._block = options
            self._dict = None
        else:
            self._block = None
            self._dict = {str(key).lower(): value for key, value in dict(options).items()}

    def has(self, name):
        if self._block is not None:
            return self._block.has_value(name)
        return name.lower() in self._dict

    def get(self, name, default=None):
        if not self.has(name):
            return default
        if self._block is not None:
            return self._block[name]
        return self._dict[name.lower()]


def _as_bool(value):
    """Interpret ini-style booleans (``T``/``F``, ``true``/``false``, 1/0)."""
    if isinstance(value, str):
        return value.strip().lower() in {"t", "true", "yes", "on", "1"}
    return bool(value)


def _parse_sfh_args(value):
    """Split the ``SFHArgs`` option into positional and keyword arguments."""
    from besta.io import string_to_func_args

    args, kwargs = [], {}
    if value is None:
        return args, kwargs
    if isinstance(value, str):
        return string_to_func_args(value)
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                item_args, item_kwargs = string_to_func_args(item)
                args.extend(item_args)
                kwargs.update(item_kwargs)
            elif isinstance(item, dict):
                kwargs.update(item)
            else:
                args.append(item)
        return args, kwargs
    args.append(value)
    return args, kwargs


def build_sfh_from_options(options, *, redshift=None, today=None):
    """Build an SFH model from module configuration options.

    Lightweight: only the SFH is built (no SSPs or data), so it can be used
    both by the pipeline modules and in post-processing.

    Parameters
    ----------
    options : cosmosis ``SectionOptions`` or dict
        Module options. Recognised keys (case-insensitive):

        - ``SFHModel`` (required): name of an :class:`SFHBase` subclass.
        - ``SFHArgs``: positional/keyword arguments of the model.
        - ``use_transforms``: sample latent variables (default ``False``).
        - ``use_sfh_smoothness_prior`` and ``sfh_smoothness_*`` settings.
        - ``min_last_interval`` and ``min_time`` (Gyr): time-anchor bounds of
          :class:`FixedMassFracSFH`.
        - ``latent_space`` and ``max_dlogsfr``: latent parametrisation of
          :class:`FixedMassFracSFH` with ``use_transforms``.
        - ``redshift``: used when the ``redshift`` argument is ``None``.
    redshift : float, optional
        Redshift of the source (sets the age of the Universe at observation).
        Defaults to the ``redshift`` option, or 0.
    today : astropy.units.Quantity, optional
        Age of the Universe at observation; overrides the redshift-based value.

    Returns
    -------
    SFHBase
        The configured SFH model.

    Notes
    -----
    Keyword arguments given in ``SFHArgs`` take precedence over the options
    above.
    """
    opts = _OptionReader(options)
    if not opts.has("SFHModel"):
        raise ValueError("Missing required option 'SFHModel'.")
    model_name = str(opts.get("SFHModel")).strip()
    model_class = globals().get(model_name)
    if not (isinstance(model_class, type) and issubclass(model_class, SFHBase)):
        raise ValueError(f"Unknown SFH model '{model_name}'.")

    args, sfh_kwargs = _parse_sfh_args(opts.get("SFHArgs"))

    if redshift is None:
        redshift = opts.get("redshift", 0.0)
    try:
        redshift = float(redshift)
    except (TypeError, ValueError):
        redshift = 0.0

    settings = {
        "redshift": redshift,
        "use_transforms": _as_bool(opts.get("use_transforms", False)),
        "use_sfh_smoothness_prior": _as_bool(
            opts.get("use_sfh_smoothness_prior", False)),
    }
    if today is not None:
        settings["today"] = today
    for name, cast in _SFH_SMOOTHNESS_OPTIONS + _SFH_MODEL_OPTIONS:
        if opts.has(name):
            settings[name] = cast(opts.get(name))

    overridden = sorted(set(settings) & set(sfh_kwargs))
    if overridden:
        logger.info("SFHArgs override the options: %s", ", ".join(overridden))
    settings.update(sfh_kwargs)

    logger.info("SFH model: %s", model_name)
    logger.info("SFH model arguments: %s", args)
    logger.info("SFH model keyword arguments: %s", settings)
    return model_class(*args, **settings)

# Mr Krtxo \(ﾟ▽ﾟ)/
