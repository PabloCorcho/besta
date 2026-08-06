import numpy as np
from astropy import units as u
import pytest

from cosmosis import DataBlock

from besta import sfh
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


def test_full_spectral_fit_rejects_zero_penalty_invalid_sample():
    class InvalidSFH:
        @staticmethod
        def parse_datablock(block):
            return 0, -1e20  # invalid sample with a prior penalty

    mod = FullSpectralFitModule.__new__(FullSpectralFitModule)
    mod.config = {"sfh_model": InvalidSFH()}
    mod.like_name = "full_spectral_fit_like"
    block = DataBlock()

    assert mod.execute(block) == 0
    assert block["likelihoods", mod.like_name] == -1e20
    assert np.isnan(block["extra", "stellar_mass"])


def test_prepare_sfh_model_forwards_smoothness_prior_options():
    class Dummy(SpectraFitModule):
        name = "Dummy"

        def make_observable(self, *args, **kwargs):
            pass

        def execute(self, *args, **kwargs):
            pass

        def plot_solution(self, *args, **kwargs):
            pass

    mod = Dummy.__new__(Dummy)
    mod.alias = "Dummy"
    mod.config = {}
    options = mod.parse_options(
        {
            "Dummy": {
                "SFHModel": "FixedTimeSFH",
                "SFHArgs": "[0.5, 1.0, 2.0, 5.0]",
                "use_sfh_smoothness_prior": True,
                "sfh_smoothness_prior_type": "legacy_index_gaussian",
                "sfh_smoothness_sigma_dex": 0.7,
                "sfh_smoothness_order": 1,
                "sfh_smoothness_min_sfr": 1e-10,
            }
        }
    )

    mod.prepare_sfh_model(options)

    prior = mod.config["sfh_model"].sfh_smoothness_prior
    assert isinstance(prior, sfh.SFHSmoothnessPrior)
    assert prior.sigma_dex == 0.7
    assert prior.order == 1
    assert prior.min_sfr == 1e-10


def test_prepare_sfh_model_builds_fixed_mass_frac_2d():
    class Dummy(SpectraFitModule):
        name = "Dummy"

        def make_observable(self, *args, **kwargs):
            pass

        def execute(self, *args, **kwargs):
            pass

        def plot_solution(self, *args, **kwargs):
            pass

    mod = Dummy.__new__(Dummy)
    mod.alias = "Dummy"
    mod.config = {}
    options = mod.parse_options(
        {
            "Dummy": {
                "SFHModel": "FixedMassFracSFH2D",
                "SFHArgs": "[0.2, 0.5, 0.8]",
            }
        }
    )

    mod.prepare_sfh_model(options)

    model = mod.config["sfh_model"]
    assert isinstance(model, sfh.FixedMassFracSFH2D)
    assert model.model.name == "tabular_mass_frac_cem_2d"
    assert "sigma_log_metallicity" in model.free_params


if __name__ == "__main__":
    import sys
    import unittest

    unittest.main(argv=[sys.argv[0]])
