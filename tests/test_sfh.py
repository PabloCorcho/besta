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

    def test_smoothness_prior_is_default(self):
        model = sfh.FixedTimeSFH(
            np.array([0.5, 1.0, 2.0, 5.0]) * u.Gyr,
            ism_metallicity_today=0.02,
            use_sfh_smoothness_prior=True,
        )
        self.assertIsInstance(
            model.sfh_smoothness_prior,
            sfh.SFHSmoothnessPrior,
        )
        self.assertAlmostEqual(model.sfh_smoothness_prior.sigma_dex, 0.3)
        self.assertAlmostEqual(model.sfh_smoothness_prior.dof, 3.0)
        self.assertAlmostEqual(
            model.sfh_smoothness_prior.relative_sfr_floor,
            1e-4,
        )

    def test_legacy_prior_type_is_rejected(self):
        with self.assertRaises(ValueError):
            sfh.FixedTimeSFH(
                np.array([0.5, 1.0, 2.0, 5.0]) * u.Gyr,
                ism_metallicity_today=0.02,
                use_sfh_smoothness_prior=True,
                sfh_smoothness_prior_type="legacy_index_gaussian",
            )

    def test_prior_is_invariant_to_mass_normalization(self):
        prior = sfh.SFHSmoothnessPrior()
        masses = np.array([0.4, 0.3, 0.2, 0.1])
        value = prior(masses, self.time_edges)
        rescaled_value = prior(1e10 * masses, self.time_edges)
        self.assertAlmostEqual(value, rescaled_value, places=12)

    def test_physical_time_linear_history_is_preferred(self):
        prior = sfh.SFHSmoothnessPrior(
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
        prior = sfh.SFHSmoothnessPrior()
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

    def _make_db(self, logsfr_value=-6.0, alpha=1.0, ism_z=0.02):
        parameters = {key: logsfr_value for key in self.model.sfh_bin_keys}
        parameters['alpha_powerlaw'] = alpha
        parameters['ism_metallicity_today'] = ism_z
        return DataBlock.from_dict({self.model.sect_name: parameters})

    def test_initialization(self):
        # N lookback edges define N + 1 bins (the oldest one up to the Big Bang)
        self.assertEqual(len(self.model.sfh_bin_keys), len(self.lookback_bins) + 1)
        self.assertEqual(self.model.sfh_bin_keys[0], "logsfr_at_bigbang")

        # Each key must be in free_params with valid [min, default, max]
        for key in self.model.sfh_bin_keys:
            self.assertIn(key, self.model.free_params)
            bounds = self.model.free_params[key]
            self.assertLess(bounds[0], bounds[1])
            self.assertLess(bounds[1], bounds[2])

    def test_lookback_time_sorted_descending(self):
        # Internal lookback_time should be sorted descending (oldest first),
        # from the Big Bang (today) to 0.
        lbt_values = self.model.lookback_time.to_value("Gyr")
        self.assertTrue(np.all(np.diff(lbt_values) < 0))
        self.assertAlmostEqual(lbt_values[0], self.model.today.to_value("Gyr"))
        self.assertAlmostEqual(lbt_values[-1], 0.0)

    def test_time_array_size(self):
        # Big Bang + input edges + today
        self.assertEqual(self.model.time.size, len(self.lookback_bins) + 2)
        self.assertEqual(self.model.time[0].to_value("Gyr"), 0.0)
        self.assertAlmostEqual(self.model.time[-1].to_value("Gyr"),
                               self.model.today.to_value("Gyr"))

    def test_key_names_encode_lookback_time(self):
        # Each key should encode the lookback time in Gyr (sorted descending)
        expected_lbt = np.sort(self.lookback_bins.to_value("Gyr"))[::-1]
        for key, lbt in zip(self.model.sfh_bin_keys[1:], expected_lbt):
            self.assertEqual(key, f"logsfr_at_{lbt:.3f}")

    def test_parse_datablock_valid(self):
        db = self._make_db(logsfr_value=-6.0)
        status, info = self.model.parse_datablock(db)

        self.assertEqual(status, 1)
        self.assertTrue(info == 0.0)
        self.assertTrue(hasattr(self.model.model, 'table_mass'))

    def test_table_mass_size_matches_time(self):
        # table_mass must align with the internal time array
        db = self._make_db(logsfr_value=-6.0)
        self.model.parse_datablock(db)
        self.assertEqual(
            len(self.model.model.table_mass),
            self.model.time.size,
        )

    def test_table_mass_starts_at_zero(self):
        # The first mass value (at the earliest time) should be 0
        db = self._make_db(logsfr_value=-6.0)
        self.model.parse_datablock(db)
        self.assertAlmostEqual(
            self.model.model.table_mass[0].to_value(u.Msun), 0.0
        )

    def test_table_mass_monotonically_increasing(self):
        db = self._make_db(logsfr_value=-6.0)
        self.model.parse_datablock(db)
        masses = self.model.model.table_mass.to_value(u.Msun)
        self.assertTrue(np.all(np.diff(masses) >= 0))

    def test_log_sfr_sets_absolute_formed_mass(self):
        # log10(SFR / (Msun / yr)) = 0 since the Big Bang forms today * 1 Msun/yr.
        self.model.parse_datablock(self._make_db(logsfr_value=0.0))
        self.assertAlmostEqual(
            self.model.model.table_mass[-1].to_value(u.Msun)
            / self.model.today.to_value(u.yr),
            1.0,
        )
        self.assertFalse(self.model.use_mass_normalization)

    def test_stars_older_than_the_largest_lookback_time(self):
        # Regression: the oldest bin (largest lookback time to the Big Bang)
        # used to be missing, so no stars older than 5 Gyr could form.
        params = {key: -6.0 for key in self.model.sfh_bin_keys}
        params["logsfr_at_bigbang"] = 0.0
        params.update(alpha_powerlaw=1.0, ism_metallicity_today=0.02)
        status, _ = self.model.parse_free_params(params)
        self.assertEqual(status, 1)
        today = self.model.today.to_value("Gyr")
        cem_model = self.model.model
        old_edge = (today - 5.0) << u.Gyr
        expected = (today - 5.0) * 1e9
        self.assertAlmostEqual(
            cem_model.stellar_mass_formed(old_edge).to_value(u.Msun) / expected,
            1.0, places=6)
        # Stars form throughout the oldest bin: with the default linear mass
        # history, half of its mass halfway through it
        half = cem_model.stellar_mass_formed(0.5 * old_edge).to_value(u.Msun)
        self.assertAlmostEqual(half / expected, 0.5, places=6)

    def test_constant_sfr_is_reproduced(self):
        self.model.parse_datablock(self._make_db(logsfr_value=0.0))
        today = self.model.today
        times = np.linspace(0.05, 0.999, 30) * today
        mass = self.model.model.stellar_mass_formed(times).to_value(u.Msun)
        np.testing.assert_allclose(mass, times.to_value(u.yr), rtol=1e-8)

    def test_invalid_lookback_times(self):
        today = self.model.today.to_value("Gyr")
        for bins in ([0.0, 1.0], [1.0, 1.0, 2.0], [-1.0], [today], [1.0, today + 1]):
            with self.subTest(bins=bins), self.assertRaises(ValueError):
                sfh.FixedTimeSFH(np.array(bins) * u.Gyr, ism_metallicity_today=0.02)

    def test_parse_datablock_updates_alpha(self):
        db = self._make_db(logsfr_value=-6.0, alpha=2.5)
        self.model.parse_datablock(db)
        self.assertAlmostEqual(self.model.model.alpha_powerlaw, 2.5)

    def test_parse_datablock_updates_metallicity(self):
        db = self._make_db(logsfr_value=-6.0, ism_z=0.03)
        self.model.parse_datablock(db)
        self.assertAlmostEqual(
            self.model.model.ism_metallicity_today.value, 0.03
        )

    def test_single_bin(self):
        # A model with a single bin should work without errors
        model = sfh.FixedTimeSFH(np.array([5.0]) * u.Gyr, ism_metallicity_today=0.02)
        self.assertEqual(len(model.sfh_bin_keys), 2)  # plus the oldest bin
        params = {key: -3.0 for key in model.sfh_bin_keys}
        params.update(alpha_powerlaw=0.5, ism_metallicity_today=0.02)
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
            sfh.SFHSmoothnessPrior,
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


class TestFixedMassFracTimeBounds(unittest.TestCase):
    """Time-anchor bounds, start values and the transform range."""

    FRACTIONS = np.array([0.3, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999])

    def test_default_bounds_and_interior_start(self):
        model = sfh.FixedMassFracSFH(self.FRACTIONS, ism_metallicity_today=0.02)
        today = model.today.to_value("Gyr")
        t_min, t_max = model.time_bounds
        self.assertEqual(t_min, 1e-3)
        self.assertAlmostEqual(today - t_max, 1e-4)
        starts = []
        for key in model.sfh_bin_keys:
            low, start, high = model.free_params[key]
            self.assertEqual((low, high), (t_min, t_max))
            self.assertTrue(low < start < high)   # also for 0.999
            starts.append(start)
        self.assertTrue(np.all(np.diff(starts) > 0))
        # The start values describe a valid SFH
        params = dict(zip(model.sfh_bin_keys, starts))
        params["alpha_powerlaw"] = 1.0
        params["ism_metallicity_today"] = 0.02
        status, _ = model.parse_datablock(
            DataBlock.from_dict({model.sect_name: params}))
        self.assertEqual(status, 1)

    def test_custom_bounds(self):
        model = sfh.FixedMassFracSFH(
            self.FRACTIONS, ism_metallicity_today=0.02, today=10.0 * u.Gyr,
            min_last_interval=0.01, min_time=0.5)
        self.assertEqual(model.time_bounds, (0.5, 9.99))

    def test_invalid_bounds(self):
        for kwargs in ({"min_last_interval": 0.0}, {"min_last_interval": -1.0},
                       {"min_time": -1.0}, {"min_time": 9.0, "min_last_interval": 1.0}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                sfh.FixedMassFracSFH(self.FRACTIONS, ism_metallicity_today=0.02,
                                     today=10.0 * u.Gyr, **kwargs)

    def test_transform_spans_the_same_range(self):
        model = sfh.FixedMassFracSFH(
            self.FRACTIONS, ism_metallicity_today=0.02, use_transforms=True,
            today=10.0 * u.Gyr, min_last_interval=0.01, min_time=0.5)
        n = self.FRACTIONS.size
        np.testing.assert_allclose(model.to_physical(np.zeros(n)), 0.5)
        np.testing.assert_allclose(model.to_physical(np.ones(n))[0], 9.99)
        latent = np.random.default_rng(1).uniform(size=(500, n))
        times = model.to_physical_batch(latent)
        self.assertTrue(np.all((times > 0.5) & (times < 9.99)))
        np.testing.assert_allclose(model.to_latent_batch(times), latent,
                                   rtol=1e-6, atol=1e-8)
        with self.assertRaises(ValueError):
            model.to_latent(np.linspace(0.6, 9.995, n))   # last anchor past t_max

    def test_transform_prior_matches_the_box_prior(self):
        # Uniform latents give uniform ordered times on [t_min, t_max]: the
        # k-th of n ordered uniforms has mean t_min + k / (n + 1) * width.
        model = sfh.FixedMassFracSFH(
            self.FRACTIONS, ism_metallicity_today=0.02, use_transforms=True,
            today=10.0 * u.Gyr, min_last_interval=0.01, min_time=0.5)
        n = self.FRACTIONS.size
        times = model.to_physical_batch(
            np.random.default_rng(2).uniform(size=(200000, n)))
        expected = 0.5 + np.arange(1, n + 1) / (n + 1) * (9.99 - 0.5)
        np.testing.assert_allclose(times.mean(axis=0), expected, atol=0.02)

    def test_options_are_forwarded(self):
        model = sfh.build_sfh_from_options({
            "SFHModel": "FixedMassFracSFH",
            "SFHArgs": "[0.5, 0.9]",
            "min_last_interval": "0.002",
            "min_time": "0.1",
        })
        today = model.today.to_value("Gyr")
        self.assertAlmostEqual(model.time_bounds[0], 0.1)
        self.assertAlmostEqual(today - model.time_bounds[1], 0.002)


class TestFixedMassFracLogSFRJumps(unittest.TestCase):
    """``latent_space = "log_sfr_jumps"``: uniform prior on log-SFR jumps."""

    FRACTIONS = np.array([0.3, 0.5, 0.75, 0.9, 0.95, 0.99])

    def make_model(self, **kwargs):
        options = dict(ism_metallicity_today=0.02, use_transforms=True,
                       latent_space="log_sfr_jumps")
        options.update(kwargs)
        return sfh.FixedMassFracSFH(self.FRACTIONS, **options)

    def sfr_per_bin(self, model, times):
        """Mean SFR of each bin on the normalised [t_min, t_max] interval."""
        t_min, t_max = model.time_bounds
        edges = np.concatenate(([t_min], times, [t_max]))
        return np.diff(np.concatenate(([0.0], self.FRACTIONS, [1.0]))) / np.diff(edges)

    def test_free_params_and_ini(self):
        import os
        import tempfile

        model = self.make_model(max_dlogsfr=1.5)
        for key in model.sfh_bin_keys:
            self.assertEqual(model.free_params[key], [-1.5, 0.0, 1.5])
        with tempfile.NamedTemporaryFile(suffix=".ini", delete=False) as file:
            path = file.name
        try:
            model.make_ini(path, mode="w")
            with open(path, encoding="utf-8") as file:
                content = file.read()
            for key in model.sfh_bin_keys:
                self.assertIn(f"{key} = -1.5 0.0 1.5", content)
        finally:
            os.unlink(path)

    def test_zero_jumps_give_a_constant_sfr(self):
        model = self.make_model()
        times = model.to_physical(np.zeros(self.FRACTIONS.size))
        t_min, t_max = model.time_bounds
        np.testing.assert_allclose(
            times, t_min + self.FRACTIONS * (t_max - t_min), rtol=1e-12)
        sfr = self.sfr_per_bin(model, times)
        np.testing.assert_allclose(sfr, sfr[0], rtol=1e-10)

    def test_jumps_are_log_sfr_ratios(self):
        model = self.make_model()
        jumps = np.array([0.5, -0.3, 1.2, 0.0, -1.9, 0.7])
        times = model.to_physical(jumps)
        self.assertTrue(np.all(np.diff(times) > 0))
        log_sfr = np.log10(self.sfr_per_bin(model, times))
        np.testing.assert_allclose(-np.diff(log_sfr), jumps, atol=1e-10)

    def test_roundtrip_and_batch_match_single(self):
        model = self.make_model()
        n = self.FRACTIONS.size
        rng = np.random.default_rng(5)
        latent = rng.uniform(-2.0, 2.0, size=(300, n))
        latent[0] = 2.0
        latent[1] = -2.0
        latent[2, 0] = 2.5         # outside the prior range
        latent[3, -1] = np.nan     # non-finite
        times = model.to_physical_batch(latent)
        self.assertTrue(np.all(np.isnan(times[2:4])))
        valid = np.ones(len(latent), bool)
        valid[2:4] = False
        self.assertTrue(np.all(np.isfinite(times[valid])))
        t_min, t_max = model.time_bounds
        self.assertTrue(np.all((times[valid] > t_min) & (times[valid] < t_max)))
        for index in (0, 1, 4, 50, 299):
            np.testing.assert_allclose(
                times[index], model.to_physical(latent[index]), rtol=1e-12)
        # Round trip on random rows. (Rows 0 and 1 are the extreme corners,
        # 12 dex of monotonic change, where the youngest or oldest anchors
        # are less than a year apart and cannot be inverted to 1e-8 dex.)
        for index in (4, 50, 299):
            np.testing.assert_allclose(
                model.to_latent(times[index]), latent[index], atol=1e-6)
        np.testing.assert_allclose(
            model.to_latent_batch(times[4:]), latent[4:], atol=1e-6)
        with self.assertRaises(ValueError):
            model.to_physical(latent[2])

    def test_times_outside_the_prior_map_to_nan(self):
        model = self.make_model(max_dlogsfr=0.5)
        steep = model.to_physical(np.full(self.FRACTIONS.size, 0.45))
        model_wide = self.make_model(max_dlogsfr=2.0)
        too_steep = model_wide.to_physical(np.full(self.FRACTIONS.size, 1.0))
        np.testing.assert_allclose(
            model.to_latent(steep), 0.45, atol=1e-8)
        self.assertTrue(np.all(np.isnan(model.to_latent_batch(too_steep))))
        with self.assertRaises(ValueError):
            model.to_latent(too_steep)

    def test_parse_datablock_with_smoothness_prior(self):
        model = self.make_model(use_sfh_smoothness_prior=True)
        params = dict(zip(model.sfh_bin_keys, np.zeros(self.FRACTIONS.size)))
        params["alpha_powerlaw"] = 1.0
        params["ism_metallicity_today"] = 0.02
        status, log_prior = model.parse_datablock(
            DataBlock.from_dict({model.sect_name: params}))
        self.assertEqual(status, 1)
        self.assertTrue(np.isfinite(log_prior))
        np.testing.assert_allclose(
            model.model.times.to_value("Gyr")[1:-1],
            model.to_physical(np.zeros(6)))

        # A constant SFH has zero curvature: it maximises the smoothness prior
        params.update(dict(zip(model.sfh_bin_keys, [0.4, -0.4] * 3)))
        _, log_prior_wiggly = model.parse_datablock(
            DataBlock.from_dict({model.sect_name: params}))
        self.assertGreater(log_prior, log_prior_wiggly)

    def test_prior_does_not_prefer_declining_histories(self):
        # Uniform jumps: log SFR(youngest bin) - log SFR(oldest bin) is
        # symmetric around 0 (constant SFH). Stick-breaking (uniform ordered
        # times) instead puts most prior mass on declining histories.
        rng = np.random.default_rng(7)
        jumps = self.make_model()
        stick = sfh.FixedMassFracSFH(
            self.FRACTIONS, ism_metallicity_today=0.02, use_transforms=True)
        n = self.FRACTIONS.size
        medians = {}
        for name, model, latent in (
                ("jumps", jumps, rng.uniform(-2, 2, size=(50000, n))),
                ("stick", stick, rng.uniform(size=(50000, n)))):
            times = model.to_physical_batch(latent)
            t_min, t_max = model.time_bounds
            edges = np.column_stack(
                (np.full(len(times), t_min), times, np.full(len(times), t_max)))
            width = np.diff(edges, axis=1)
            # SFR_i = delta_f_i / width_i: youngest (0.01) vs oldest (0.3) bin
            log_ratio = (np.log10(0.01 / width[:, -1])
                         - np.log10(0.3 / width[:, 0]))
            medians[name] = np.median(log_ratio)
        self.assertLess(abs(medians["jumps"]), 0.1)
        self.assertLess(medians["stick"], -1.0)

    def test_other_transforms_are_unchanged(self):
        default = sfh.FixedMassFracSFH(
            self.FRACTIONS, ism_metallicity_today=0.02, use_transforms=True)
        self.assertEqual(default.latent_space, "stick_breaking")
        self.assertEqual(default.free_params[default.sfh_bin_keys[0]], [0.0, 0.5, 1.0])
        # Without transforms the option is ignored: times are sampled directly
        direct = sfh.FixedMassFracSFH(
            self.FRACTIONS, ism_metallicity_today=0.02,
            latent_space="log_sfr_jumps")
        low, _, high = direct.free_params[direct.sfh_bin_keys[0]]
        self.assertEqual((low, high), direct.time_bounds)
        times = np.linspace(1.0, 13.0, 6)
        np.testing.assert_array_equal(direct.to_physical(times), times)

    def test_invalid_options(self):
        with self.assertRaises(ValueError):
            self.make_model(latent_space="softmax")
        for value in (0.0, -1.0, np.inf):
            with self.subTest(max_dlogsfr=value), self.assertRaises(ValueError):
                self.make_model(max_dlogsfr=value)
        with self.assertRaises(ValueError):
            sfh.FixedMassFracSFH([0.5, 1.0], ism_metallicity_today=0.02,
                                 use_transforms=True, latent_space="log_sfr_jumps")

    def test_options_are_forwarded(self):
        model = sfh.build_sfh_from_options({
            "SFHModel": "FixedMassFracSFH2D",
            "SFHArgs": "[0.5, 0.9]",
            "use_transforms": "T",
            "latent_space": "log_sfr_jumps",
            "max_dlogsfr": "1.0",
        })
        self.assertEqual(model.latent_space, "log_sfr_jumps")
        self.assertEqual(model.max_dlogsfr, 1.0)
        self.assertEqual(model.free_params["t_at_frac_0.5000"], [-1.0, 0.0, 1.0])


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

    def test_fixed_time_ssfr_prior_preserving_roundtrip(self):
        lookback_bins = np.array([0.5, 1.0, 2.0]) * u.Gyr
        model = sfh.FixedTime_sSFR_SFH(lookback_bins, ism_metallicity_today=0.02,
                                       use_transforms=True)
        for key in model.sfh_bin_keys:
            self.assertEqual(model.free_params[key], [0.0, 0.5, 1.0])

        latent = np.array([0.2, 0.7, 0.4])
        physical_logssfr = model.to_physical(latent)
        inv = model.to_latent(physical_logssfr)
        np.testing.assert_allclose(inv, latent, rtol=1e-8, atol=1e-10)
        self.assertTrue(np.all(physical_logssfr >= model.min_ssfr_logyr))
        self.assertTrue(np.all(physical_logssfr <= model.max_ssfr_logyr))
        self.assertTrue(
            np.all(np.diff(physical_logssfr) < model.delta_logtau)
        )

        parameters = {
            key: value
            for key, value in zip(model.sfh_bin_keys, latent)
        }
        parameters["alpha_powerlaw"] = 1.0
        parameters["ism_metallicity_today"] = 0.02
        status, prior_penalty = model.parse_datablock(
            DataBlock.from_dict({model.sect_name: parameters})
        )
        self.assertEqual(status, 1)
        self.assertEqual(prior_penalty, 0.0)

        # The old n-logit softmax forced the latest of n+1 mass bins to zero.
        remaining_mass = (
            model.lookback_time.to_value("yr") * 10**physical_logssfr
        )
        self.assertGreater(remaining_mass[-1], 0.0)

    def test_fixed_time_ssfr_transform_preserves_uniform_prior_volume(self):
        lookback_bins = np.array([0.5, 1.0, 2.0]) * u.Gyr
        model = sfh.FixedTime_sSFR_SFH(
            lookback_bins,
            ism_metallicity_today=0.02,
            use_transforms=True,
        )

        def numerical_jacobian_determinant(latent):
            step = 1e-5
            columns = []
            for index in range(latent.size):
                offset = np.zeros_like(latent)
                offset[index] = step
                columns.append(
                    (
                        model.to_physical(latent + offset)
                        - model.to_physical(latent - offset)
                    )
                    / (2 * step)
                )
            return abs(np.linalg.det(np.column_stack(columns)))

        expected_volume = model._latent_prior_transform._total_volumes[-1]
        for latent in (
            np.array([0.2, 0.7, 0.4]),
            np.array([0.7, 0.3, 0.8]),
        ):
            self.assertAlmostEqual(
                numerical_jacobian_determinant(latent),
                expected_volume,
                places=5,
            )

    def test_fixed_time_ssfr_transformed_ini_uses_unit_priors(self):
        import os
        import tempfile

        model = sfh.FixedTime_sSFR_SFH(
            np.array([0.5, 1.0, 2.0]) * u.Gyr,
            ism_metallicity_today=0.02,
            use_transforms=True,
        )
        with tempfile.NamedTemporaryFile(suffix=".ini", delete=False) as file:
            path = file.name
        try:
            model.make_ini(path, mode="w")
            with open(path, encoding="utf-8") as file:
                content = file.read()
            for key in model.sfh_bin_keys:
                self.assertIn(f"{key} = 0.0 0.5 1.0", content)
        finally:
            os.unlink(path)

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

    # --- Batch (vectorised) transforms ---------------------------------

    @staticmethod
    def _latent_samples(n_params, n_samples=200, seed=3):
        """Random unit-cube samples plus edge rows and two invalid rows."""
        rng = np.random.default_rng(seed)
        latent = rng.uniform(size=(n_samples, n_params))
        latent[0] = 0.0
        latent[1] = 1.0
        latent[2, -1] = 1.5      # outside [0, 1]
        latent[3, 0] = np.nan    # non-finite
        return latent

    def _check_batch_matches_single(self, model, n_params):
        latent = self._latent_samples(n_params)
        physical = model.to_physical_batch(latent)
        self.assertEqual(physical.shape, latent.shape)

        # Invalid rows -> NaN, all others finite
        self.assertTrue(np.all(np.isnan(physical[2:4])))
        self.assertTrue(np.all(np.isfinite(np.delete(physical, [2, 3], axis=0))))

        # Same result as the per-sample method
        for index in (0, 1, *range(4, latent.shape[0])):
            np.testing.assert_allclose(
                physical[index], model.to_physical(latent[index]),
                rtol=1e-12, atol=1e-12)

        # Round trip on interior samples
        interior = physical[4:]
        latent_back = model.to_latent_batch(interior)
        for index in range(0, interior.shape[0], 20):
            np.testing.assert_allclose(
                latent_back[index], model.to_latent(interior[index]),
                rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(latent_back, latent[4:], rtol=1e-6, atol=1e-8)

        # A single sample is accepted and returned as a (1, n) array
        self.assertEqual(model.to_physical_batch(latent[5]).shape, (1, n_params))

    def test_fixed_time_ssfr_batch_transforms_match_single_sample(self):
        model = sfh.FixedTime_sSFR_SFH(
            np.array([0.1, 0.5, 1.0, 2.0, 5.0]) * u.Gyr,
            ism_metallicity_today=0.02,
            use_transforms=True,
        )
        self._check_batch_matches_single(model, 5)
        # Physically invalid rows (sSFR above 1 / tau) map back to NaN
        bad = np.full((1, 5), 0.0)
        self.assertTrue(np.all(np.isnan(model.to_latent_batch(bad))))

    def test_fixed_mass_frac_batch_transforms_match_single_sample(self):
        for model_class in (sfh.FixedMassFracSFH, sfh.FixedMassFracSFH2D):
            with self.subTest(model_class=model_class.__name__):
                model = model_class(
                    np.array([0.1, 0.3, 0.5, 0.8, 0.95]),
                    ism_metallicity_today=0.02,
                    use_transforms=True,
                )
                self._check_batch_matches_single(model, 5)
                today = model.today.to_value("Gyr")
                bad = np.array([
                    [2.0, 1.0, 3.0, 4.0, 5.0],               # not increasing
                    [1.0, 2.0, 3.0, 4.0, today + 1.0],       # after today
                ])
                self.assertTrue(np.all(np.isnan(model.to_latent_batch(bad))))

    def test_batch_transforms_are_identity_without_transforms(self):
        model = sfh.FixedTimeSFH(
            np.array([0.5, 1.0, 2.0, 5.0]) * u.Gyr, ism_metallicity_today=0.02)
        values = np.random.default_rng(0).normal(size=(10, 5))
        np.testing.assert_array_equal(model.to_physical_batch(values), values)
        np.testing.assert_array_equal(model.to_latent_batch(values), values)


class TestSFHInterpolationOption(unittest.TestCase):
    """``sfh_interpolation``: mass-history interpolation of piecewise models."""

    def piecewise_models(self, **kwargs):
        common = dict(ism_metallicity_today=0.02, **kwargs)
        return [
            sfh.FixedTimeSFH(np.array([0.1, 1.0, 5.0]) * u.Gyr, **common),
            sfh.FixedTime_sSFR_SFH(np.array([0.1, 1.0, 5.0]) * u.Gyr, **common),
            sfh.FixedMassFracSFH(np.array([0.5, 0.9, 0.99]), **common),
            sfh.FixedMassFracSFH2D(np.array([0.5, 0.9, 0.99]), **common),
        ]

    def test_default_is_linear(self):
        for model in self.piecewise_models():
            with self.subTest(model=type(model).__name__):
                self.assertEqual(model.sfh_interpolation, "linear")
                self.assertEqual(model.model.interpolation, "linear")

    def test_option_reaches_the_pst_model(self):
        for model in self.piecewise_models(sfh_interpolation="pchip"):
            with self.subTest(model=type(model).__name__):
                self.assertEqual(model.sfh_interpolation, "pchip")
                self.assertEqual(model.model.interpolation, "pchip")
        with self.assertRaises(ValueError):
            sfh.FixedMassFracSFH(np.array([0.5, 0.9]), sfh_interpolation="cubic")

    def test_exponential_model_keeps_the_cubic(self):
        model = sfh.ExponentialSFH(sfh_interpolation="linear")
        self.assertEqual(model.model.interpolation, "pchip")

    def test_linear_sfr_is_the_interval_mean(self):
        model = sfh.FixedMassFracSFH(np.array([0.5, 0.9, 0.99]),
                                     ism_metallicity_today=0.02)
        today = model.today.to_value("Gyr")
        times = np.array([4.0, 10.0, today - 0.5])
        params = {key: value for key, value in zip(model.sfh_bin_keys, times)}
        params.update(alpha_powerlaw=1.0, ism_metallicity_today=0.02)
        status, _ = model.parse_free_params(params)
        self.assertEqual(status, 1)
        # Long interval (0.9 -> 0.99) followed by a short one: flat SFR inside
        inside = np.linspace(10.05, today - 0.55, 50) << u.Gyr
        sfr = model.model.sfr(inside).to_value(u.Msun / u.Gyr)
        np.testing.assert_allclose(sfr, 0.09 / (today - 0.5 - 10.0), rtol=1e-10)

    def test_build_from_options(self):
        model = sfh.build_sfh_from_options({
            "SFHModel": "FixedMassFracSFH", "SFHArgs": "[0.5, 0.9]",
            "sfh_interpolation": "pchip"})
        self.assertEqual(model.model.interpolation, "pchip")
        model = sfh.build_sfh_from_options({
            "SFHModel": "FixedMassFracSFH", "SFHArgs": "[0.5, 0.9]"})
        self.assertEqual(model.model.interpolation, "linear")


class TestBuildSFHFromOptions(unittest.TestCase):
    """``sfh.build_sfh_from_options``: dict (ini) and CosmoSIS options."""

    def test_from_parsed_ini_dict(self):
        from besta import io
        from besta.config import cosmology
        # As stored by besta.io.Reader (values parsed from the ini strings)
        raw = {
            "SFHModel": "FixedMassFracSFH",
            "SFHArgs": "(0.1, 0.5, 0.9)",
            "use_transforms": "T",
            "redshift": "0.2",
            "use_sfh_smoothness_prior": "T",
            "sfh_smoothness_sigma_dex": "0.7",
            "file": "full_spectral_fit.py",   # unrelated options are ignored
        }
        options = {key: io._parse_value(value) for key, value in raw.items()}
        model = sfh.build_sfh_from_options(options)

        self.assertIsInstance(model, sfh.FixedMassFracSFH)
        self.assertTrue(model.use_transforms)
        np.testing.assert_allclose(model.mass_fraction, [0.1, 0.5, 0.9])
        self.assertAlmostEqual(model.redshift, 0.2)
        self.assertAlmostEqual(
            model.today.to_value("Gyr"), cosmology.age(0.2).to_value("Gyr"))
        self.assertIsInstance(model.sfh_smoothness_prior, sfh.SFHSmoothnessPrior)
        self.assertEqual(model.sfh_smoothness_prior.sigma_dex, 0.7)

    def test_keys_are_case_insensitive_and_booleans_parsed(self):
        model = sfh.build_sfh_from_options({
            "sfhmodel": "FixedMassFracSFH",
            "sfhargs": "[0.2, 0.8]",
            "USE_TRANSFORMS": "F",
        })
        self.assertFalse(model.use_transforms)
        self.assertIsNone(model.sfh_smoothness_prior)
        self.assertAlmostEqual(model.redshift, 0.0)

    def test_redshift_argument_and_sfhargs_precedence(self):
        options = {
            "SFHModel": "FixedTimeSFH",
            "SFHArgs": "[0.5, 1.0, 2.0], logsfr_min=-4.0",
            "redshift": 0.1,
        }
        # Explicit argument overrides the option
        model = sfh.build_sfh_from_options(options, redshift=0.3)
        self.assertAlmostEqual(model.redshift, 0.3)
        self.assertEqual(model.free_params[model.sfh_bin_keys[0]][0], -4.0)
        # Keywords in SFHArgs override everything (no duplicate-kwarg error)
        options["SFHArgs"] = "[0.5, 1.0, 2.0], redshift=0.5"
        model = sfh.build_sfh_from_options(options, redshift=0.3)
        self.assertAlmostEqual(model.redshift, 0.5)

    def test_invalid_model_names(self):
        with self.assertRaises(ValueError):
            sfh.build_sfh_from_options({"SFHArgs": "[0.5]"})
        for name in ("NotAModel", "cosmology", "SFHSmoothnessPrior"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                sfh.build_sfh_from_options({"SFHModel": name})


if __name__ == "__main__":
    unittest.main()
