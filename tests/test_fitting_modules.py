import numpy as np
from astropy import units as u
import pytest

from cosmosis import DataBlock

from besta.pipeline_modules.base_module import SpectraFitModule
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


def test_full_spectral_fit_make_observable(tmp_path):
    spec = make_dummy_spectrum(tmp_path)
    block = DataBlock()
    all_params = {
        "dust.attenuation": {"a_v": 0.0},
        "kinematics":  {
        "los_vel": 0.0,
        "los_sigma": 100.0,
        "los_h3": 0.0,
        "los_h4": 0.0,
        },
        "stars.sfh": {
        "logtau": 0.5,
        "alpha_powerlaw": 1.0,
        "ism_metallicity_today": 0.02,
        }
    }
    for sect, params in all_params.items():
        for k, v in params.items():
            block[sect, k] = v

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
            "save_ssfr_over_tau": [0.1, 1.0],
        }
    }
    mod = FullSpectralFitModule(opts)
    flux_model, weights = mod.make_observable(block, parse=True)
    assert flux_model.shape == mod.config["flux"].shape
    assert weights.shape == mod.config["flux"].shape
    np.testing.assert_allclose(mod.config["ssfr_tau"], [0.1, 1.0])
    assert np.isfinite(block["extra", "ssfr_over_tau_0.1000"])
    assert np.isfinite(block["extra", "ssfr_over_tau_1.0000"])

if __name__ == "__main__":
    import sys
    import unittest

    unittest.main(argv=[sys.argv[0]])
