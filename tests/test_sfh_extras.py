"""BaseModule.save_sfh_extras: saved t_frac / sSFR values match the per-value methods."""
import types
import unittest

import numpy as np
from astropy import units as u
from cosmosis import DataBlock

from besta import sfh
from besta.pipeline_modules.base_module import BaseModule


def make_module(sfh_model, fractions=None, taus=None):
    """Minimal stand-in exposing the BaseModule methods under test."""
    config = {"sfh_model": sfh_model}
    if fractions is not None:
        config.update(save_t_frac_at=True, t_frac_at=np.asarray(fractions, float))
    if taus is not None:
        config.update(save_ssfr_over_tau=True, ssfr_tau=np.asarray(taus, float))
    module = types.SimpleNamespace(
        config=config, T_FRAC_TIME_RESOLUTION=BaseModule.T_FRAC_TIME_RESOLUTION)
    for name in ("_prepare_sfh_extras", "save_sfh_extras",
                 "get_t_frac_at", "get_ssfr_over_tau"):
        setattr(module, name, types.MethodType(getattr(BaseModule, name), module))
    module._prepare_sfh_extras()
    return module


FRACTIONS = [0.1, 0.5, 0.9, 0.99]
TAUS = [0.01, 0.1, 1.0]


class TestSaveSFHExtras(unittest.TestCase):

    def assert_matches_reference(self, module, sfh_model):
        new, ref = DataBlock(), DataBlock()
        module.save_sfh_extras(new, sfh_model)
        for frac in module.config.get("t_frac_at", []):
            module.get_t_frac_at(ref, sfh_model, frac)
            key = f"t_frac_at_{frac:.4f}"
            np.testing.assert_allclose(new["extra", key], ref["extra", key],
                                       rtol=1e-12, err_msg=key)
        for tau in module.config.get("ssfr_tau", []):
            module.get_ssfr_over_tau(ref, sfh_model, tau)
            key = f"ssfr_over_tau_{tau:.4f}"
            np.testing.assert_allclose(new["extra", key], ref["extra", key],
                                       rtol=1e-12, atol=1e-12, err_msg=key)

    def test_fixed_mass_fraction_model(self):
        model = sfh.FixedMassFracSFH(
            [0.3, 0.5, 0.75, 0.9, 0.95, 0.99], ism_metallicity_today=0.02,
            use_transforms=True, latent_space="log_sfr_jumps", redshift=0.1)
        module = make_module(model, FRACTIONS, TAUS)
        rng = np.random.default_rng(0)
        for _ in range(10):
            params = dict(zip(model.sfh_bin_keys, rng.uniform(-2, 2, 6)))
            params.update(alpha_powerlaw=1.0, ism_metallicity_today=0.02)
            status, _ = model.parse_free_params(params)
            self.assertEqual(status, 1)
            self.assert_matches_reference(module, model)

    def test_exponential_model_and_partial_requests(self):
        model = sfh.ExponentialSFH()
        for logtau in (-0.5, 0.5, 1.5):
            model.parse_free_params({"logtau": logtau, "alpha_powerlaw": 1.0,
                                     "ism_metallicity_today": 0.02})
            for kwargs in ({"fractions": FRACTIONS}, {"taus": TAUS},
                           {"fractions": FRACTIONS, "taus": TAUS}):
                self.assert_matches_reference(make_module(model, **kwargs), model)

    def test_single_mass_history_evaluation(self):
        model = sfh.ExponentialSFH()
        model.parse_free_params({"logtau": 0.5, "alpha_powerlaw": 1.0,
                                 "ism_metallicity_today": 0.02})
        module = make_module(model, FRACTIONS, TAUS)
        calls = []
        original = model.model.stellar_mass_formed

        def counting(times):
            calls.append(times)
            return original(times)

        model.model.stellar_mass_formed = counting
        try:
            module.save_sfh_extras(DataBlock(), model)
        finally:
            model.model.stellar_mass_formed = original
        self.assertEqual(len(calls), 1)

    def test_nothing_requested(self):
        model = sfh.ExponentialSFH()
        module = make_module(model)
        self.assertIsNone(module.config["sfh_extras"])
        block = DataBlock()
        module.save_sfh_extras(block, model)
        self.assertEqual(len(list(block.keys())), 0)

    def test_invalid_timescales(self):
        model = sfh.ExponentialSFH()
        for taus in ([0.0], [-1.0], [100.0]):
            with self.subTest(taus=taus), self.assertRaises(ValueError):
                make_module(model, taus=taus)

    def test_times_follow_a_new_observing_time(self):
        model = sfh.ExponentialSFH()
        module = make_module(model, taus=TAUS)
        model.today = 10.0 << u.Gyr
        module.save_sfh_extras(DataBlock(), model)
        self.assertEqual(module.config["sfh_extras"]["today"], 10.0)
        np.testing.assert_allclose(
            module.config["sfh_extras"]["times"].to_value(u.Gyr),
            [10.0 - t for t in TAUS] + [10.0])


if __name__ == "__main__":
    unittest.main()
