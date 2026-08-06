import unittest
import numpy as np
from astropy import units as u

from cosmosis import DataBlock
from besta import sfh

import pst


class TestSFHSmoothnessPriors(unittest.TestCase):

    def setUp(self):
        self.time_edges = np.array([0.0, 1.0, 4.0, 10.0, 13.0])
        self.delta_t = np.diff(self.time_edges)
        self.time_centres = 0.5 * (
            self.time_edges[:-1] + self.time_edges[1:]
        )

    def test_robust_time_curvature_is_default(self):
        model = sfh.FixedTimeSFH(
            np.array([0.5, 1.0, 2.0, 5.0]) * u.Gyr,
            ism_metallicity_today=0.02,
            use_sfh_smoothness_prior=True,
        )
        self.assertIsInstance(
            model.sfh_smoothness_prior,
            sfh.SFHRobustTimeCurvaturePrior,
        )
        self.assertAlmostEqual(model.sfh_smoothness_prior.sigma_dex, 0.3)
        self.assertAlmostEqual(model.sfh_smoothness_prior.dof, 3.0)
        self.assertAlmostEqual(
            model.sfh_smoothness_prior.relative_sfr_floor,
            1e-4,
        )

    def test_legacy_index_gaussian_remains_available(self):
        model = sfh.FixedTimeSFH(
            np.array([0.5, 1.0, 2.0, 5.0]) * u.Gyr,
            ism_metallicity_today=0.02,
            use_sfh_smoothness_prior=True,
            sfh_smoothness_prior_type="legacy_index_gaussian",
        )
        self.assertIsInstance(model.sfh_smoothness_prior, sfh.SFHSmoothnessPrior)
        self.assertAlmostEqual(model.sfh_smoothness_prior.sigma_dex, 0.5)
        self.assertEqual(model.sfh_smoothness_prior.order, 2)

    def test_prior_is_invariant_to_mass_normalization(self):
        prior = sfh.SFHRobustTimeCurvaturePrior()
        masses = np.array([0.4, 0.3, 0.2, 0.1])
        value = prior(masses, self.time_edges)
        rescaled_value = prior(1e10 * masses, self.time_edges)
        self.assertAlmostEqual(value, rescaled_value, places=12)

    def test_physical_time_linear_history_is_preferred(self):
        prior = sfh.SFHRobustTimeCurvaturePrior(
            relative_sfr_floor=1e-12,
        )
        physical_log_sfr = 0.1 * self.time_centres
        physical_masses = 10**physical_log_sfr * self.delta_t

        index_log_sfr = np.linspace(
            physical_log_sfr[0],
            physical_log_sfr[-1],
            physical_log_sfr.size,
        )
        index_masses = 10**index_log_sfr * self.delta_t

        self.assertGreater(
            prior(physical_masses, self.time_edges),
            prior(index_masses, self.time_edges),
        )

    def test_burst_has_finite_heavy_tailed_penalty(self):
        prior = sfh.SFHRobustTimeCurvaturePrior()
        smooth_masses = self.delta_t.copy()
        burst_masses = smooth_masses.copy()
        burst_masses[1] *= 1e4

        smooth_value = prior(smooth_masses, self.time_edges)
        burst_value = prior(burst_masses, self.time_edges)
        self.assertTrue(np.isfinite(burst_value))
        self.assertLess(burst_value, smooth_value)
        self.assertGreater(burst_value, -100.0)

    def test_invalid_prior_type_is_rejected(self):
        with self.assertRaises(ValueError):
            sfh.FixedTimeSFH(
                np.array([0.5, 1.0, 2.0, 5.0]) * u.Gyr,
                ism_metallicity_today=0.02,
                use_sfh_smoothness_prior=True,
                sfh_smoothness_prior_type="unknown",
            )


class TestFixedTimeSFH(unittest.TestCase):

    def setUp(self):
        # Define example lookback time bins (in Gyr)
        self.lookback_bins = np.array([0.5, 1.0, 2.0, 5.0]) * u.Gyr
        self.model = sfh.FixedTimeSFH(self.lookback_bins, ism_metallicity_today=0.02)

    def _make_db(self, logmass_value=-6.0, alpha=1.0, ism_z=0.02):
        parameters = {key: logmass_value for key in self.model.sfh_bin_keys}
        parameters['alpha_powerlaw'] = alpha
        parameters['ism_metallicity_today'] = ism_z
        return DataBlock.from_dict({self.model.sect_name: parameters})

    def test_initialization(self):
        # Number of sfh_bin_keys must equal number of input bins
        self.assertEqual(len(self.model.sfh_bin_keys), len(self.lookback_bins))

        # Each key must be in free_params with valid [min, default, max]
        for key in self.model.sfh_bin_keys:
            self.assertIn(key, self.model.free_params)
            bounds = self.model.free_params[key]
            self.assertLess(bounds[0], bounds[1])
            self.assertLess(bounds[1], bounds[2])

    def test_lookback_time_sorted_descending(self):
        # Internal lookback_time should be sorted descending (oldest first),
        # with 0 appended as the last entry.
        lbt_values = self.model.lookback_time.to_value("Gyr")
        self.assertTrue(np.all(np.diff(lbt_values) <= 0))
        self.assertAlmostEqual(lbt_values[-1], 0.0)

    def test_time_array_size(self):
        # time array must have one more element than the input bins (0 appended)
        self.assertEqual(self.model.time.size, len(self.lookback_bins) + 1)

    def test_key_names_encode_lookback_time(self):
        # Each key should encode the lookback time in Gyr (sorted descending)
        expected_lbt = np.sort(self.lookback_bins.to_value("Gyr"))[::-1]
        for key, lbt in zip(self.model.sfh_bin_keys, expected_lbt):
            self.assertIn(f"{lbt:.3f}", key)

    def test_parse_datablock_valid(self):
        db = self._make_db(logmass_value=-6.0)
        status, info = self.model.parse_datablock(db)

        self.assertEqual(status, 1)
        self.assertTrue(info == 0.0)
        self.assertTrue(hasattr(self.model.model, 'table_mass'))

    def test_table_mass_size_matches_time(self):
        # table_mass must align with the internal time array
        db = self._make_db(logmass_value=-6.0)
        self.model.parse_datablock(db)
        self.assertEqual(
            len(self.model.model.table_mass),
            self.model.time.size,
        )

    def test_table_mass_starts_at_zero(self):
        # The first mass value (at the earliest time) should be 0
        db = self._make_db(logmass_value=-6.0)
        self.model.parse_datablock(db)
        self.assertAlmostEqual(
            self.model.model.table_mass[0].to_value(u.Msun), 0.0
        )

    def test_table_mass_monotonically_increasing(self):
        db = self._make_db(logmass_value=-6.0)
        self.model.parse_datablock(db)
        masses = self.model.model.table_mass.to_value(u.Msun)
        self.assertTrue(np.all(np.diff(masses) >= 0))

    def test_parse_datablock_updates_alpha(self):
        db = self._make_db(logmass_value=-6.0, alpha=2.5)
        self.model.parse_datablock(db)
        self.assertAlmostEqual(self.model.model.alpha_powerlaw, 2.5)

    def test_parse_datablock_updates_metallicity(self):
        db = self._make_db(logmass_value=-6.0, ism_z=0.03)
        self.model.parse_datablock(db)
        self.assertAlmostEqual(
            self.model.model.ism_metallicity_today.value, 0.03
        )

    def test_single_bin(self):
        # A model with a single bin should work without errors
        model = sfh.FixedTimeSFH(np.array([5.0]) * u.Gyr, ism_metallicity_today=0.02)
        self.assertEqual(len(model.sfh_bin_keys), 1)
        params = {model.sfh_bin_keys[0]: -3.0,
                  'alpha_powerlaw': 0.5,
                  'ism_metallicity_today': 0.02}
        db = DataBlock.from_dict({model.sect_name: params})
        status, info = model.parse_datablock(db)
        self.assertEqual(status, 1)
        self.assertTrue(info == 0.0)

    def test_parse_free_params(self):
        # parse_free_params is a convenience wrapper around parse_datablock
        params = {key: -6.0 for key in self.model.sfh_bin_keys}
        params['alpha_powerlaw'] = 1.0
        params['ism_metallicity_today'] = 0.02
        status, info = self.model.parse_free_params(params)
        self.assertEqual(status, 1)
        self.assertTrue(info == 0.0)

    def test_make_ini_creates_file(self):
        import tempfile, os, configparser
        with tempfile.NamedTemporaryFile(suffix=".ini", delete=False) as f:
            path = f.name
        try:
            self.model.make_ini(path)
            self.assertTrue(os.path.exists(path))
            with open(path) as f:
                content = f.read()
            # All bin keys must appear in the ini file
            for key in self.model.sfh_bin_keys:
                self.assertIn(key, content)
        finally:
            os.unlink(path)

    def test_use_transforms_mode(self):
        model = sfh.FixedTimeSFH(
            self.lookback_bins, ism_metallicity_today=0.02, use_transforms=True
        )
        # In transforms mode, latent values go through softmax → always valid
        params = {key: 0.0 for key in model.sfh_bin_keys}  # equal fractions
        params['alpha_powerlaw'] = 0.5
        params['ism_metallicity_today'] = 0.02
        db = DataBlock.from_dict({model.sect_name: params})
        status, info = model.parse_datablock(db)
        self.assertEqual(status, 1)
        self.assertTrue(info == 0.0)


class TestFixedTime_sSFR_SFH(unittest.TestCase):

    def setUp(self):
        # Simple decreasing lookback times
        self.lookback_bins = np.array([0.5, 1.0, 2.0]) * u.Gyr
        self.model = sfh.FixedTime_sSFR_SFH(self.lookback_bins, ism_metallicity_today=0.02)

    def test_initialization(self):
        # Check the number of time bins = len(lookback_bins)
        expected_keys = len(self.lookback_bins)
        self.assertEqual(len(self.model.sfh_bin_keys), expected_keys)

        # Check free parameter format and range
        for key in self.model.sfh_bin_keys:
            self.assertIn(key, self.model.free_params)
            bounds = self.model.free_params[key]
            self.assertTrue(bounds[0] < bounds[1] < bounds[2])

    def test_parse_datablock_valid(self):
        # Choose values to make mass fraction increase monotonically
        values = [-10.0, -10.0, -10.0]  # Safe logssfr values
        parameters = {
            key: val for key, val in zip(self.model.sfh_bin_keys, values)
        }
        parameters["alpha_powerlaw"] = 1.0
        parameters["ism_metallicity_today"] = 0.02

        db = DataBlock.from_dict({self.model.sect_name: parameters})
        status, info = self.model.parse_datablock(db)

        self.assertEqual(status, 1)
        self.assertEqual(info, 0.0)
        self.assertTrue(hasattr(self.model.model, "table_mass"))
        self.assertEqual(len(self.model.model.table_mass), len(self.model.lookback_time) + 2)

    def test_smoothness_prior_is_applied(self):
        model = sfh.FixedTime_sSFR_SFH(
            self.lookback_bins,
            ism_metallicity_today=0.02,
            use_sfh_smoothness_prior=True,
        )
        values = [-10.0, -10.0, -10.0]
        parameters = {
            key: val for key, val in zip(model.sfh_bin_keys, values)
        }
        parameters["alpha_powerlaw"] = 1.0
        parameters["ism_metallicity_today"] = 0.02

        status, log_prior = model.parse_datablock(
            DataBlock.from_dict({model.sect_name: parameters})
        )

        self.assertEqual(status, 1)
        self.assertTrue(np.isfinite(log_prior))
        self.assertIsInstance(
            model.sfh_smoothness_prior,
            sfh.SFHRobustTimeCurvaturePrior,
        )

    def test_parse_datablock_monotonicity_error(self):
        # Use large sSFR to force decreasing cumulative mass (non-monotonic)
        values = [0.0, 0.0, 0.0]  # logssfr = 0 => sSFR = 1 => mass_frac = 1 - lt_yr * 1 → < 0
        parameters = {
            key: val for key, val in zip(self.model.sfh_bin_keys, values)
        }
        parameters["alpha_powerlaw"] = 1.0
        parameters["ism_metallicity_today"] = 0.02

        db = DataBlock.from_dict({self.model.sect_name: parameters})
        status, prior_penalty = self.model.parse_datablock(db)

        self.assertEqual(status, 0)
        self.assertGreaterEqual(prior_penalty, -1e20)


class TestFixedMassFracSFH(unittest.TestCase):

    def setUp(self):
        # Define increasing mass fraction bins (excluding 0 and 1)
        self.mass_fractions = np.array([0.2, 0.5, 0.8])
        self.model = sfh.FixedMassFracSFH(self.mass_fractions, ism_metallicity_today=0.02)

    def test_initialization(self):
        # Check that the number of keys = number of intermediate mass fractions
        expected_keys = len(self.mass_fractions)
        self.assertEqual(len(self.model.sfh_bin_keys), expected_keys)

        # Validate key names and parameter bounds
        for i, key in enumerate(self.model.sfh_bin_keys):
            self.assertIn(key, self.model.free_params)
            bounds = self.model.free_params[key]
            self.assertTrue(0 <= bounds[0] < bounds[1] < bounds[2])

    def test_parse_datablock_valid(self):
        # Times must be in ascending order for a valid SFH
        today_gyr = self.model.today.to_value("Gyr")
        step = today_gyr / (len(self.mass_fractions) + 1)
        times = [(i + 1) * step for i in range(len(self.mass_fractions))]

        parameters = {
            key: val for key, val in zip(self.model.sfh_bin_keys, times)
        }
        parameters["alpha_powerlaw"] = 1.0
        parameters["ism_metallicity_today"] = 0.02

        db = DataBlock.from_dict({self.model.sect_name: parameters})
        status, prior_penalty = self.model.parse_datablock(db)

        self.assertEqual(status, 1)
        self.assertTrue(prior_penalty == 0.0)
        self.assertTrue(hasattr(self.model.model, "table_t"))
        self.assertEqual(len(self.model.model.table_t), len(self.mass_fractions) + 2)

    def test_parse_datablock_non_monotonic(self):
        # Intentionally provide times out of order to trigger error
        times = [5.0, 3.0, 2.0]  # Not strictly increasing

        parameters = {
            key: val for key, val in zip(self.model.sfh_bin_keys, times)
        }
        parameters["alpha_powerlaw"] = 1.0
        parameters["ism_metallicity_today"] = 0.02

        db = DataBlock.from_dict({self.model.sect_name: parameters})
        status, prior_penalty = self.model.parse_datablock(db)

        self.assertEqual(status, 0)
        self.assertGreaterEqual(prior_penalty, -1e20)


class TestFixedMassFracSFH2D(unittest.TestCase):

    def setUp(self):
        self.mass_fractions = np.array([0.2, 0.5, 0.8])
        self.today = 10.0 * u.Gyr
        self.model = sfh.FixedMassFracSFH2D(
            self.mass_fractions,
            today=self.today,
            ism_metallicity_today=0.02,
        )

    def _make_parameters(self, sigma=0.3):
        parameters = {
            key: value
            for key, value in zip(self.model.sfh_bin_keys, [2.0, 5.0, 8.0])
        }
        parameters.update(
            alpha_powerlaw=0.5,
            ism_metallicity_today=0.02,
            sigma_log_metallicity=sigma,
        )
        return parameters

    @staticmethod
    def _make_toy_ssp():
        ssp = pst.SSP.SSPBase()
        ssp.name = "toy_besta_cem_2d"
        ssp.ages = np.array([0.1, 1.0, 5.0, 10.0]) * u.Gyr
        ssp.metallicities = (
            np.array([0.002, 0.005, 0.01, 0.02, 0.05])
            << u.dimensionless_unscaled
        )
        return ssp

    def test_initialization_uses_pst_2d_model(self):
        self.assertIsInstance(self.model.model, pst.cem.TabularMassFracCEM2D)
        self.assertIsInstance(self.model.model, pst.cem.ChemicalEvolutionModel2D)
        self.assertIn("sigma_log_metallicity", self.model.free_params)
        self.assertTrue(
            np.isclose(self.model.model.sigma_log_metallicity.to_value(), 0.25)
        )

    def test_parse_datablock_updates_scatter(self):
        datablock = DataBlock.from_dict(
            {self.model.sect_name: self._make_parameters(sigma=0.35)}
        )

        status, log_prior = self.model.parse_datablock(datablock)

        self.assertEqual(status, 1)
        self.assertEqual(log_prior, 0.0)
        self.assertTrue(
            np.isclose(self.model.model.sigma_log_metallicity.to_value(), 0.35)
        )

    def test_zero_scatter_matches_fixed_mass_frac_sfh(self):
        deterministic = sfh.FixedMassFracSFH(
            self.mass_fractions,
            today=self.today,
            ism_metallicity_today=0.02,
        )
        parameters_2d = self._make_parameters(sigma=0.0)
        parameters_1d = parameters_2d.copy()
        parameters_1d.pop("sigma_log_metallicity")

        status_1d, _ = deterministic.parse_free_params(parameters_1d)
        status_2d, _ = self.model.parse_free_params(parameters_2d)
        ssp = self._make_toy_ssp()
        weights_1d = deterministic.model.interpolate_ssp_masses(
            ssp, self.today, oversample_factor=3
        )
        weights_2d = self.model.model.interpolate_ssp_masses(
            ssp, self.today, oversample_factor=3
        )

        self.assertEqual(status_1d, 1)
        self.assertEqual(status_2d, 1)
        self.assertTrue(
            u.allclose(weights_1d, weights_2d, rtol=0.0, atol=0.0 * u.Msun)
        )

    def test_finite_scatter_spreads_and_conserves_mass(self):
        status, _ = self.model.parse_free_params(self._make_parameters(sigma=0.3))
        weights = self.model.model.interpolate_ssp_masses(
            self._make_toy_ssp(), self.today, oversample_factor=3
        )

        self.assertEqual(status, 1)
        self.assertTrue(u.isclose(weights.sum(), 1.0 * u.Msun))
        self.assertGreater(np.count_nonzero(weights.sum(axis=1).value), 2)


class TestExponentialSFH(unittest.TestCase):

    def setUp(self):
        # Define mock time array for deterministic behavior
        self.mock_time = np.linspace(0.1, 13.5, 100) * u.Gyr
        self.model = sfh.ExponentialSFH(
            time=self.mock_time,
            ism_metallicity_today=0.02,
            alpha_powerlaw=1.0
        )

    def test_initialization(self):
        self.assertIn("logtau", self.model.free_params)
        bounds = self.model.free_params["logtau"]
        self.assertTrue(bounds[0] < bounds[1] < bounds[2])
        self.assertTrue((self.model.time == np.sort(self.mock_time)).all())

    def test_parse_datablock_valid(self):
        parameters = {
            "logtau": 0.5,  # tau ~ 3.16 Gyr
            "alpha_powerlaw": 1.0,
            "ism_metallicity_today": 0.02,
        }
        db = DataBlock.from_dict({self.model.sect_name: parameters})
        status, prior_penalty = self.model.parse_datablock(db)

        self.assertEqual(status, 1)
        self.assertTrue(prior_penalty == 0.0 or prior_penalty is None)

        # Ensure table_mass is normalized and positive
        mass = self.model.model.table_mass.to_value(u.Msun)
        self.assertTrue(np.all(mass >= 0))
        self.assertAlmostEqual(mass[-1], 1.0, places=6)

    def test_mass_monotonicity(self):
        # Check mass growth is monotonic
        parameters = {
            "logtau": 0.2,
            "alpha_powerlaw": 0.5,
            "ism_metallicity_today": 0.015,
        }
        db = DataBlock.from_dict({self.model.sect_name: parameters})
        self.model.parse_datablock(db)

        mass = self.model.model.table_mass.to_value(u.Msun)
        self.assertTrue(np.all(np.diff(mass) >= 0))


class TestBetaSFH(unittest.TestCase):

    def setUp(self):
        self.model = sfh.BetaSFH(
            ism_metallicity_today=0.02,
            alpha_powerlaw=1.0,
        )

    def _make_db(self, t_start=1.0, t_end=8.0, alpha=2.5, beta=4.0,
                 alpha_powerlaw=1.5, ism_z=0.03):
        parameters = {
            "alpha": alpha,
            "beta": beta,
            "t_start": t_start,
            "t_end": t_end,
            "alpha_powerlaw": alpha_powerlaw,
            "ism_metallicity_today": ism_z,
        }
        return DataBlock.from_dict({self.model.sect_name: parameters})

    def test_initialization(self):
        for key in ("alpha", "beta", "t_start", "t_end"):
            self.assertIn(key, self.model.free_params)
            bounds = self.model.free_params[key]
            self.assertLess(bounds[0], bounds[1])
            self.assertLess(bounds[1], bounds[2])

        self.assertAlmostEqual(
            self.model.model.t_end.value.to_value(u.Gyr),
            self.model.today.to_value(u.Gyr),
        )

    def test_parse_datablock_valid(self):
        status, prior_penalty = self.model.parse_datablock(self._make_db())

        self.assertEqual(status, 1)
        self.assertTrue(prior_penalty == 0.0 or prior_penalty is None)
        self.assertAlmostEqual(self.model.model.alpha, 2.5)
        self.assertAlmostEqual(self.model.model.beta, 4.0)
        self.assertAlmostEqual(
            self.model.model.t_start.value.to_value(u.Gyr), 1.0
        )
        self.assertAlmostEqual(
            self.model.model.t_end.value.to_value(u.Gyr), 8.0
        )
        self.assertAlmostEqual(self.model.model.alpha_powerlaw.value, 1.5)
        self.assertAlmostEqual(
            self.model.model.ism_metallicity_today.value, 0.03
        )

    def test_parse_datablock_rejects_invalid_time_bounds(self):
        status, prior_penalty = self.model.parse_datablock(
            self._make_db(t_start=8.0, t_end=1.0)
        )

        self.assertEqual(status, 0)
        self.assertGreaterEqual(prior_penalty, -1e20)


class TestTransforms(unittest.TestCase):

    def test_fixed_time_softmax_roundtrip(self):
        lookback_bins = np.array([0.5, 1.0, 2.0, 5.0]) * u.Gyr
        model = sfh.FixedTimeSFH(lookback_bins, ism_metallicity_today=0.02,
                                 use_transforms=True)
        latent = np.array([0.1, -0.2, 0.3, 0.0])
        physical = model.to_physical(latent)
        np.testing.assert_allclose(physical.sum(), 1.0, rtol=1e-6)
        # softmax inverse defined up to additive constant; center both
        inv = model.to_latent(physical)
        np.testing.assert_allclose(inv - inv.mean(), latent - latent.mean(),
                                   rtol=1e-6, atol=1e-8)
        params = {k: v for k, v in zip(model.sfh_bin_keys, latent)}
        params["alpha_powerlaw"] = 1.0
        params["ism_metallicity_today"] = 0.02
        status, prior_penalty = model.parse_datablock(DataBlock.from_dict({model.sect_name: params}))
        self.assertEqual(status, 1)
        self.assertTrue(prior_penalty == 0.0)
        self.assertAlmostEqual(model.model.table_mass.to_value(u.Msun)[-1], 1.0, places=6)

    def test_fixed_time_ssfr_softmax_roundtrip(self):
        lookback_bins = np.array([0.5, 1.0, 2.0]) * u.Gyr
        model = sfh.FixedTime_sSFR_SFH(lookback_bins, ism_metallicity_today=0.02,
                                       use_transforms=True)
        latent = np.array([0.0, 0.1, -0.1])
        physical_logssfr = model.to_physical(latent)
        inv = model.to_latent(physical_logssfr)
        np.testing.assert_allclose(inv - inv.mean(), latent - latent.mean(),
                                   rtol=1e-6, atol=1e-8)

    def test_fixed_mass_frac_time_roundtrip(self):
        mass_fractions = np.array([0.2, 0.5, 0.8])
        model = sfh.FixedMassFracSFH(mass_fractions, ism_metallicity_today=0.02,
                                     use_transforms=True)
        for key in model.sfh_bin_keys:
            self.assertEqual(model.free_params[key], [0.0, 0.5, 1.0])
        latent = np.array([0.2, 0.7, 0.4])
        physical = model.to_physical(latent)
        self.assertTrue(np.all(np.diff(physical) > 0))
        self.assertGreater(physical[0], 0.0)
        self.assertLess(physical[-1], model.today.to_value("Gyr"))
        inv = model.to_latent(physical)
        np.testing.assert_allclose(inv, latent, rtol=1e-6, atol=1e-12)

        params = {k: v for k, v in zip(model.sfh_bin_keys, latent)}
        params["alpha_powerlaw"] = 1.0
        params["ism_metallicity_today"] = 0.02
        status, prior_penalty = model.parse_datablock(
            DataBlock.from_dict({model.sect_name: params})
        )
        self.assertEqual(status, 1)
        self.assertEqual(prior_penalty, 0.0)
        self.assertEqual(model.model.times.size, len(mass_fractions) + 2)


if __name__ == "__main__":
    unittest.main()
