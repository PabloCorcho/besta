"""Predefined CosmoSIS pipeline modules shipped with BESTA."""

from .base_module import BaseModule
from .full_spectral_fit import FullSpectralFitModule
from .galaxy_spectra import GalaxySpectraModule
from .galaxy_photometry import GalaxyPhotometryModule
from .spectra_redshift_fit import SpectraRedshiftFitModule
# from .sfh_photometry_grid import SFHPhotometryGridModule
# from .sfh_photometry_emu import SFHPhotometryEmulatorModule

__all__ = [
    "BaseModule",
    "FullSpectralFitModule",
    "GalaxySpectraModule",
    "GalaxyPhotometryModule",
    "SpectraRedshiftFitModule",
    # "SFHPhotometryGridModule",
    # "SFHPhotometryEmulatorModule",
]
