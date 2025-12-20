import numpy as np
from astropy import units as u
import pytest

from cosmosis import DataBlock

from besta.pipeline_modules.base_module import SpectraFitModule
from besta.pipeline_modules.kin_dust import KinDustModule
from besta.pipeline_modules.full_spectral_fit import FullSpectralFitModule
import importlib


def make_dummy_spectrum(tmp_path):
    wl = np.linspace(4000, 5000, 50)
    flux = np.ones_like(wl)
    err = np.ones_like(wl) * 0.1
    fname = tmp_path / "spec.dat"
    np.savetxt(fname, np.vstack([wl, flux, err]).T)
    return str(fname)


def test_prepare_observed_spectra_weights_guard(tmp_path):
    spec = make_dummy_spectrum(tmp_path)

    class Dummy(SpectraFitModule):
        name = "Dummy"

        def __init__(self, options):
            super().__init__(options)
            options = self.parse_options(options)
            self.prepare_observed_spectra(options)

        def make_observable(self, *args, **kwargs):
            pass

        def execute(self, *args, **kwargs):
            pass

        def plot_solution(self, *args, **kwargs):
            pass

    # Zero weights should trigger ValueError
    opts = {
        "Dummy": {
            "inputSpectrum": spec,
            "mask": spec,  # reuse shape; will all be >0
            "wlUnits": "Angstrom",
            "fluxUnits": "1e-16 erg / (s cm2 Angstrom)",
            "wlRange": [4010, 4990],
        }
    }
    # overwrite mask file with zeros
    np.savetxt(spec, np.vstack([np.linspace(4000, 5000, 50), np.ones(50), np.ones(50)]).T)
    np.savetxt(tmp_path / "mask.dat", np.zeros(50))
    opts["Dummy"]["mask"] = str(tmp_path / "mask.dat")
    with pytest.raises(ValueError):
        Dummy(opts)


def test_kindust_make_observable_shapes(tmp_path):
    spec = make_dummy_spectrum(tmp_path)
    block = DataBlock()
    for p in ["av", "los_vel", "los_sigma", "los_h3", "los_h4"]:
        block["parameters", p] = 0.0 if p != "los_sigma" else 100.0
    opts = {
        "KinDust": {
            "inputSpectrum": spec,
            "SSPModel": "PopStar",
            "SSPModelArgs": "cha",
            "SSPDir": "None",
            "wlUnits": "Angstrom",
            "fluxUnits": "1e-16 erg / (s cm2 Angstrom)",
            "wlRange": [4010, 4990],
            "velscale": 200.0,
            "ExtinctionLaw": "ccm89",
        }
    }
    mod = KinDustModule(opts)
    flux_model, weights = mod.make_observable(block)
    assert flux_model.shape == mod.config["flux"].shape
    assert weights.shape == mod.config["flux"].shape


def test_full_spectral_fit_make_observable(tmp_path):
    spec = make_dummy_spectrum(tmp_path)
    block = DataBlock()
    params = {
        "av": 0.0,
        "los_vel": 0.0,
        "los_sigma": 100.0,
        "los_h3": 0.0,
        "los_h4": 0.0,
        "logtau": 0.5,
        "alpha_powerlaw": 1.0,
        "ism_metallicity_today": 0.02,
    }
    for k, v in params.items():
        block["parameters", k] = v
    opts = {
        "FullSpectralFit": {
            "inputSpectrum": spec,
            "SSPModel": "PopStar",
            "SSPModelArgs": "cha",
            "SSPDir": "None",
            "wlUnits": "Angstrom",
            "fluxUnits": "1e-16 erg / (s cm2 Angstrom)",
            "wlRange": [4010, 4990],
            "velscale": 200.0,
            "ExtinctionLaw": "ccm89",
            "SFHModel": "ExponentialSFH",
        }
    }
    mod = FullSpectralFitModule(opts)
    flux_model, weights = mod.make_observable(block)
    assert flux_model.shape == mod.config["flux"].shape
    assert weights.shape == mod.config["flux"].shape


def test_sfh_spectra_module_removed():
    with pytest.raises(AttributeError):
        importlib.reload(importlib.import_module("besta.pipeline_modules")).SFHSpectraModule
