"""Kinematic convolution utilities for spectral modeling.

This module provides pixel-space LOSVD kernel classes and convolution helpers
used by BESTA spectral fitting modules. Kernels are designed to be reusable
across likelihood calls via lightweight in-memory caching.

Conventions
-----------
- Velocities and dispersions are provided in physical units and converted to
    pixel units using ``velocity_scale``.
- Convolutions act on the last axis of the input arrays.
- Kernels are normalized to unit sum before convolution.
"""
import numpy as np
import re
from scipy.signal import fftconvolve
from scipy.special import erf
from scipy import sparse

from astropy import units as u

from besta import config as CONFIG
from besta.logging import get_logger

logger = get_logger(__name__)

SQRT2 = np.sqrt(2)
SQRT2PI = np.sqrt(2 * np.pi)
CACHE_PIX_DECIMALS = 3
CACHE_NMODELS = 256
DELTA_KERNEL_ATOL = 1e-3

class LOSVDPixelKernel:
    """Line-of-sight velocity distribution kernel.

    Attributes
    ----------
    velocity_scale : float
        Velocity step represented by one pixel (same units as LOS velocities).
    kernel_weight : np.ndarray or None
        Normalized kernel weights sampled on the pixel grid.
    edge_pixels : int
        Number of edge pixels likely affected by convolution artifacts.
    skip_convolution : bool
        If ``True``, convolution is treated as identity (delta-like kernel).
    """

    @property
    def kernel_weight(self):
        """Kernel weights."""
        return self._kernel_weight
    
    @kernel_weight.setter
    def kernel_weight(self, value):
        if value is not None:
            value = np.asarray(value, dtype=float)
            norm = np.sum(value)

            if norm > 0:
                # Check if all the weight is on a single pixel (delta kernel)
                if np.isclose(norm, value.max(), atol=DELTA_KERNEL_ATOL):
                    self.skip_convolution = True
                else:
                    self.skip_convolution = False

                self._kernel_weight = value / norm

            else:
                logger.warning("Kernel weights sum to zero; using unnormalized values.")
                raise ValueError("Kernel weights sum to zero; cannot normalize.")
                # self.skip_convolution = False
                # self._kernel_weight = value

    @property
    def size(self):
        """Kernel size in pixels."""
        if self.kernel_weight is not None:
            return self.kernel_weight.size
        else:
            return 0

    def __init__(self, velocity_scale):
        """Initialize a pixel-space LOSVD kernel container.

        Parameters
        ----------
        velocity_scale : float
            Velocity step represented by one spectral pixel.
        """
        logger.debug("Initializing LOSVDPixelKernel with velocity_scale=%s", velocity_scale)
        self.velocity_scale = velocity_scale
        self._kernel_weight = None
        self.skip_convolution = False
        self.edge_pixels = 0
        self._cache = {}

    def _cached_kernel(self, key, build_kernel):
        """Populate ``kernel_weight`` from cache or from a builder callback.

        Parameters
        ----------
        key : hashable
            Cache key describing kernel parameters.
        build_kernel : Callable[[], np.ndarray]
            Callback used to build the kernel when ``key`` is absent.
        """
        kernel = self._cache.get(key)
        if kernel is None:
            kernel = build_kernel()
            self._cache[key] = kernel
            # Keep cache bounded to avoid unbounded memory growth in long chains.
            if len(self._cache) > CACHE_NMODELS:
                self._cache.pop(next(iter(self._cache)))
        self.kernel_weight = kernel

    def convolve(self, spectra):
        """Convolve the input spectra with the LOSVD kernel."""
        if self.kernel_weight is not None:
            if self.skip_convolution:
                return spectra
            if np.ndim(spectra) == 1:
                return fftconvolve(spectra, self.kernel_weight, mode="same")

            kernel = self.kernel_weight.reshape((1,) * (np.ndim(spectra) - 1) + (-1,))
            return fftconvolve(spectra, kernel, mode="same", axes=-1)
        else:
            raise ValueError("Kernel weights are not set.")

    def get_percentile_pixel(self, percentile):
        """Get a percentile location in pixel units relative to kernel center.

        Parameters
        ----------
        percentile : float
            Desired percentile (between 0 and 100).

        Returns
        -------
        pixel_offset : float
            Pixel offset relative to the central kernel pixel.
        """
        if self.kernel_weight is None:
            raise ValueError("Kernel weights are not set.")
        if not (0.0 <= percentile <= 100.0):
            raise ValueError("percentile must be in [0, 100].")
        
        cumulative = np.cumsum(self.kernel_weight)
        # Interpolate on bin edges so symmetric kernels yield zero-centered
        # median offsets instead of the half-pixel bias from center-grid CDF.
        cdf_edges = np.concatenate(([0.0], cumulative))
        pixel_edges = np.arange(len(self.kernel_weight) + 1, dtype=float) - 0.5
        pixel = np.interp(percentile / 100.0, cdf_edges, pixel_edges)
        center = 0.5 * (len(self.kernel_weight) - 1)
        return pixel - center

    def get_percentile_velocity(self, percentile):
        """Get a percentile location in velocity units relative to kernel center."""
        return self.get_percentile_pixel(percentile) * self.velocity_scale

    def parse_parameters(self, datablock):
        """Read kinematic parameters from a DataBlock and set kernel weights.

        Notes
        -----
        Subclasses must implement this method according to their parameterization.
        """
        raise NotImplementedError("This method should be implemented by subclasses to parse parameters from the kernel model.")

    @classmethod
    def make_ini(cls, ini_file: str) -> str:
        """Return default INI values for this kernel."""
        raise NotImplementedError("This method should be implemented by subclasses to provide default INI values.")

class GaussianPixelKernel(LOSVDPixelKernel):
    """Single-Gaussian LOSVD kernel in pixel space."""

    def __init__(self, velocity_scale, sigma_truncation=5.0):
        """Create a Gaussian LOSVD kernel.

        Parameters
        ----------
        velocity_scale : float
            Velocity step represented by one spectral pixel.
        sigma_truncation : float, optional
            Kernel half-width in units of sigma when building finite support.
        """
        super().__init__(velocity_scale)
        self.sigma_truncation = float(sigma_truncation)

    def parse_parameters(self, datablock):
        vel = datablock["kinematics", "los_vel"]
        sigma = datablock["kinematics", "los_sigma"]
        self.set_parameters(vel, sigma)

    def set_parameters(self, vel, sigma):
        """Set Gaussian LOSVD parameters and build/cache kernel weights.

        Parameters
        ----------
        vel : float
            Mean LOS velocity.
        sigma : float
            LOS velocity dispersion.
        """
        sigma_pixel = sigma / self.velocity_scale
        vel_pixel = vel / self.velocity_scale

        if sigma_pixel <= 0:
            self.edge_pixels = 0
            self.kernel_weight = np.array([1.0], dtype=float)
            return

        half_width = max(1, int(np.ceil(self.sigma_truncation * sigma_pixel + np.abs(vel_pixel))))
        self.edge_pixels = int(np.ceil(self.sigma_truncation * sigma_pixel))
        key = (round(vel_pixel, CACHE_PIX_DECIMALS), round(sigma_pixel, CACHE_PIX_DECIMALS), half_width)

        def build_kernel():
            x_edges = np.arange(-half_width - 0.5, half_width + 1.5, 1.0)
            cmf = self.__cumulative_distribution(x_edges, sigma_pixel, vel_pixel)
            return cmf[1:] - cmf[:-1]

        self._cached_kernel(key, build_kernel)

    def __cumulative_distribution(self, x, sigma_pixel, vel_pixel):
        return 0.5 * (1 + erf((x - vel_pixel) / (sigma_pixel * SQRT2)))

    @classmethod
    def make_ini(cls, ini_file: str) -> str:
        """Return default INI values for this kernel."""
        with open(ini_file, "a", encoding="utf-8") as file:
            file.write(f"; Default prior file for LOSVD kernel: {str(self.__class__)}\n")
            file.write(f"[kinematics]\n")
            file.write(f"los_vel = -500 0 500\n")
            file.write(f"los_sigma = 50 100 500\n")
        return ini_file

class SplitGaussianPixelKernel(LOSVDPixelKernel):
    """Split-Gaussian LOSVD kernel in pixel space.

    This kernel uses different velocity dispersions on blue and red sides
    around the LOS velocity centroid.
    """

    def __init__(self, velocity_scale, sigma_truncation=5.0):
        super().__init__(velocity_scale)
        self.sigma_truncation = float(sigma_truncation)

    def parse_parameters(self, datablock):
        vel = datablock["kinematics", "los_vel"]
        sigma_blue = datablock["kinematics", "los_sigma_blue"]
        sigma_red = datablock["kinematics", "los_sigma_red"]
        self.set_parameters(vel, sigma_blue, sigma_red)

    def set_parameters(self, vel, sigma_blue, sigma_red):
        """Set split-Gaussian LOSVD parameters and build/cache kernel weights.

        Parameters
        ----------
        vel : float
            Mean LOS velocity.
        sigma_blue : float
            Dispersion used for pixels blueward of the centroid.
        sigma_red : float
            Dispersion used for pixels redward of the centroid.
        """
        sigma_blue_px = sigma_blue / self.velocity_scale
        sigma_red_px = sigma_red / self.velocity_scale
        vel_pixel = vel / self.velocity_scale

        if sigma_blue_px <= 0 or sigma_red_px <= 0:
            self.edge_pixels = 0
            self.kernel_weight = np.array([1.0], dtype=float)
            return

        sigma_max = max(sigma_blue_px, sigma_red_px)
        half_width = max(1, int(np.ceil(self.sigma_truncation * sigma_max + np.abs(vel_pixel))))
        self.edge_pixels = int(np.ceil(self.sigma_truncation * sigma_max))
        key = (
            round(vel_pixel, CACHE_PIX_DECIMALS),
            round(sigma_blue_px, CACHE_PIX_DECIMALS),
            round(sigma_red_px, CACHE_PIX_DECIMALS),
            half_width,
        )

        def build_kernel():
            x = np.arange(-half_width, half_width + 1, dtype=float) - vel_pixel
            sigma = np.where(x < 0.0, sigma_blue_px, sigma_red_px)
            w = np.exp(-0.5 * (x / sigma) ** 2) / (sigma * SQRT2PI)
            return np.where(np.isfinite(w), w, 0.0)

        self._cached_kernel(key, build_kernel)

    @classmethod
    def make_ini(cls, ini_file: str) -> str:
        """Return default INI values for this kernel."""
        with open(ini_file, "a", encoding="utf-8") as file:
            file.write(f"; Default prior file for LOSVD kernel: {str(cls.__class__)}\n")
            file.write(f"[kinematics]\n")
            file.write(f"los_vel = -500 0 500\n")
            file.write(f"los_sigma_blue = 50 100 500\n")
            file.write(f"los_sigma_red = 50 100 500\n")
        return ini_file

class GaussHermitePixelKernel(LOSVDPixelKernel):
    """Gauss-Hermite LOSVD kernel in pixel space.

    The profile is controlled by mean velocity, velocity dispersion and
    optional third/fourth-order Hermite moments ``h3`` and ``h4``.
    """

    def __init__(self, velocity_scale, sigma_truncation=5.0):
        super().__init__(velocity_scale)
        self.sigma_truncation = float(sigma_truncation)

    def parse_parameters(self, datablock):
        self.set_parameters(
            float(datablock["kinematics", "los_vel"]),
            float(datablock["kinematics", "los_sigma"]),
            h3=float(datablock["kinematics", "los_h3"]),
            h4=float(datablock["kinematics", "los_h4"]),
        )

    def set_parameters(self, vel, sigma, h3=0.0, h4=0.0):
        """Set Gauss-Hermite LOSVD parameters and build/cache kernel weights.

        Parameters
        ----------
        vel : float
            Mean LOS velocity.
        sigma : float
            LOS velocity dispersion.
        h3 : float, optional
            Third-order Gauss-Hermite coefficient.
        h4 : float, optional
            Fourth-order Gauss-Hermite coefficient.
        """
        sigma_pixel = float(sigma) / self.velocity_scale
        vel_pixel = float(vel) / self.velocity_scale

        if sigma_pixel <= 0:
            self.edge_pixels = 0
            self.kernel_weight = np.array([1.0], dtype=float)
            return

        half_width = max(
            1,
            int(np.ceil(self.sigma_truncation * sigma_pixel + np.abs(vel_pixel))),
        )
        self.edge_pixels = int(np.ceil(self.sigma_truncation * sigma_pixel))
        key = (
            round(vel_pixel, CACHE_PIX_DECIMALS),
            round(sigma_pixel, CACHE_PIX_DECIMALS),
            round(float(h3), CACHE_PIX_DECIMALS),
            round(float(h4), CACHE_PIX_DECIMALS),
            half_width,
        )

        def build_kernel():
            x = np.arange(-half_width, half_width + 1, dtype=float) - vel_pixel
            w = self.__losvd(x, sigma_pixel=sigma_pixel, h3=float(h3), h4=float(h4))
            w = np.where(np.isfinite(w), w, 0.0)
            if np.sum(w) <= 0:
                w = np.exp(-0.5 * (x / sigma_pixel) ** 2) / (sigma_pixel * SQRT2PI)
            return w

        self._cached_kernel(key, build_kernel)

    def __losvd(self, vel_pixel, sigma_pixel, h3=0, h4=0):
        """Evaluate a Gauss-Hermite line-of-sight velocity distribution kernel."""

        y = vel_pixel / sigma_pixel

        g = (np.exp(-(y**2) / 2) / sigma_pixel / SQRT2PI
            * (
                1
                + h3 * (y * (2 * y**2 - 3) / np.sqrt(3))
                + h4 * ((4 * (y**2 - 3) * y**2 + 3) / np.sqrt(24))
            )
        )
        return g
    
    @classmethod
    def make_ini(cls, ini_file: str) -> str:
        """Return default INI values for this kernel."""
        with open(ini_file, "a", encoding="utf-8") as file:
            file.write(f"; Default prior file for LOSVD kernel: {str(cls.__class__)}\n")
            file.write(f"[kinematics]\n")
            file.write(f"los_vel = -500 0 500\n")
            file.write(f"los_sigma = 50 100 500\n")
            file.write(f"los_h3 = -0.1 0.0 0.1\n")
            file.write(f"los_h4 = -0.1 0.0 0.1\n")
        return ini_file


class PieceWisePixelKernel(LOSVDPixelKernel):
    """Piecewise-constant LOSVD kernel defined in velocity bins.

    The kernel is specified by sampled bin weights in velocity space and then
    interpolated onto the module pixel grid.
    """

    def __init__(self, velocity_scale, velocity_bin_size, velocity_min, velocity_max):
        """Create a piecewise LOSVD kernel parameterization.

        Parameters
        ----------
        velocity_scale : float
            Velocity step represented by one spectral pixel.
        velocity_bin_size : float
            Width of input velocity bins.
        velocity_min : float
            Lower bound of the piecewise velocity grid.
        velocity_max : float
            Upper bound of the piecewise velocity grid.
        """
        super().__init__(velocity_scale)
        self.velocity_bin_size = velocity_bin_size
        if velocity_min >= velocity_max:
            raise ValueError("velocity_min must be less than velocity_max.")
        self.velocity_bin_edges = np.arange(
            velocity_min, velocity_max + velocity_bin_size, velocity_bin_size)
        # Cache for querying the datablock
        self.bin_ids = np.arange(0, self.velocity_bin_edges.size - 1, 1)
        pixel_min = velocity_min / self.velocity_scale
        pixel_max = velocity_max / self.velocity_scale
        # Ensure kernel array is odd, symmetric and covers both edges
        edges = np.abs([pixel_min, pixel_max]).max()
        self.x_pixel_edges = np.arange(-edges - 0.5, edges + 1.5, 1.0)
        self.x_vel_edges = self.x_pixel_edges * self.velocity_scale

    def parse_parameters(self, datablock):
        weights = np.asarray([datablock["kinematics", f"vel_bin_{ith}"] for ith in self.bin_ids], dtype=float)
        # resample into the velocity scale pixel grid
        cum_kernel = np.cumsum(weights)
        cum_kernel = np.insert(cum_kernel, 0, 0.0)  # add zero at the beginning for interpolation
        cum_kernel = np.interp(
            self.x_vel_edges,
            self.velocity_bin_edges,
            cum_kernel,
            left=0.0,
            right=cum_kernel[-1],
        )
        self.edge_pixels = int(np.ceil(np.max(np.abs(self.x_pixel_edges))))
        self.kernel_weight = np.diff(cum_kernel)

    @classmethod
    def make_ini(cls, ini_file: str, velocity_min: float, velocity_max: float, velocity_bin_size: float) -> str:
        """Return default INI values for this kernel."""
        n_bins = int(np.ceil((velocity_max - velocity_min) / velocity_bin_size))
        with open(ini_file, "a", encoding="utf-8") as file:
            file.write(f"; Default prior file for LOSVD kernel: {str(cls.__class__)}\n")
            file.write(f"[kinematics]\n")
            for ith in range(n_bins):
                file.write(f"vel_bin_{ith} = 0.0 1.0 10.0\n")
        return ini_file


def normal_cdf(x, mu=0.0, sigma=1.0):
    """Normal cumulative density function.

    Parameters
    ----------
    x : array-like
        Evaluation points.
    mu : float, optional
        Mean of the normal distribution.
    sigma : float, optional
        Standard deviation of the normal distribution.

    Returns
    -------
    np.ndarray
        CDF values evaluated at ``x``.
    """
    return (1.0 + erf((x - mu) / sigma / np.sqrt(2.0))) / 2.0

def convolve_variable_gaussian_kernel(
    spectra,
    sigma_pixel,
    kappa_sigma_thresh=3.0,
    kappa_trunc=5.0,
):
    """
    Convolve an input spectra with a Gaussian kernel of varying width,
    using a sparse banded convolution matrix.

    Parameters
    ----------
    spectra : np.ndarray or astropy.units.Quantity
        N-dimensional spectra. The last dimension must correspond to the
        wavelength axis.
    sigma_pixel : np.ndarray
        Standard deviation of the Gaussian LSF for each spectral resolution
        element, expressed in pixel units (length = n_pix).
    kappa_sigma_thresh : float, optional
        Threshold in units of the Gaussian sigma. If a single pixel contains
        at least ``kappa_sigma_thresh`` sigmas of the LSF, that pixel is left
        untouched (i.e. the convolution kernel is clamped to a delta function
        at that pixel). Default is ``3.0``.
    kappa_trunc : float, optional
        Truncation radius of the Gaussian in units of sigma. Only pixels within
        ``±kappa_trunc * sigma`` of the central pixel contribute to the kernel.
        Default is ``5.0``.

    Returns
    -------
    convolved_spectra : np.ndarray or astropy.units.Quantity
        A convolved version of ``spectra``.
    """
    # Handle astropy units if present
    has_unit = hasattr(spectra, "unit")
    if has_unit:
        unit = spectra.unit
        spec_vals = np.asarray(spectra.value, dtype=float)
    else:
        unit = None
        spec_vals = np.asarray(spectra, dtype=float)

    sigma_pixel = np.asarray(sigma_pixel, dtype=float)
    n_pix = sigma_pixel.size

    if spec_vals.shape[-1] != n_pix:
        raise ValueError(
            "Last axis of 'spectra' must have the same length as 'sigma_pixel'."
        )

    # Build sparse convolution matrix W of shape (n_pix, n_pix)
    rows = []
    cols = []
    data = []

    for i, sigma in enumerate(sigma_pixel):
        # Clamp to delta kernel if LSF is effectively within one pixel
        # 0.5 / sigma >= kappa  ->  sigma <= 0.5 / kappa
        if sigma <= 0.0 or (0.5 / sigma) >= kappa_sigma_thresh:
            rows.append(i)
            cols.append(i)
            data.append(1.0)
            continue

        # Local window around pixel i
        half_width = int(np.ceil(kappa_trunc * sigma))
        j_min = max(0, i - half_width)
        j_max = min(n_pix - 1, i + half_width)

        # Pixel indices contributing to output pixel i
        j = np.arange(j_min, j_max + 1)

        # Pixel edges in "pixel number" coordinates, relative to the LSF centre at i
        # We define edges from (j_min - 0.5) to (j_max + 0.5), step 1
        edges = np.arange(j_min, j_max + 2) - i - 0.5  # length len(j) + 1

        # CDF at edges, sigma is already in pixel units
        cmf = normal_cdf(edges / sigma)
        w = cmf[1:] - cmf[:-1]  # length len(j)

        # Normalise row
        s = w.sum()
        if s > 0:
            w /= s
        else:
            # fallback: delta if something went very wrong numerically
            j = np.array([i])
            w = np.array([1.0])

        rows.extend([i] * len(j))
        cols.extend(j.tolist())
        data.extend(w.tolist())

    # Sparse matrix with non-zero entries only in the local Gaussian windows
    W = sparse.csr_matrix((data, (rows, cols)), shape=(n_pix, n_pix))

    # Flatten all non-spectral dimensions and apply the convolution
    orig_shape = spec_vals.shape
    spec_flat = spec_vals.reshape(-1, n_pix)           # (N_other, n_pix)
    # out = spec_flat @ W.T  (since rows of W map input->output)
    out_flat = spec_flat @ W.T
    out = out_flat.reshape(orig_shape)

    if has_unit:
        return out * unit
    return out
