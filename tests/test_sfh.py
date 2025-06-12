import unittest
import numpy as np
from astropy import units as u

from cosmosis import DataBlock
from besta import sfh

import pst

class TestFixedTimeSFH(unittest.TestCase):

    def setUp(self):
        # Define example lookback time bins (in Gyr)
        self.lookback_bins = np.array([0.5, 1.0, 2.0, 5.0]) * u.Gyr
        self.model = sfh.FixedTimeSFH(self.lookback_bins, ism_metallicity_today=0.02)

    def test_initialization(self):
        # Check if the number of sfh_bin_keys is correct (N_bins - 1)
        expected_bins = len(self.lookback_bins)
        self.assertEqual(len(self.model.sfh_bin_keys), expected_bins)

        # Check free parameters match bin keys
        for key in self.model.sfh_bin_keys:
            self.assertIn(key, self.model.free_params)

    def test_parse_datablock_valid(self):
        # Build valid parameters based on expected keys
        parameters = {
            key: -6.0 for key in self.model.sfh_bin_keys  # low mass bins
        }
        parameters['alpha_powerlaw'] = 1.0
        parameters['ism_metallicity_today'] = 0.02

        db = DataBlock.from_dict({"parameters": parameters})
        status, info = self.model.parse_datablock(db)

        self.assertEqual(status, 1)
        self.assertIsNone(info)
        self.assertTrue(hasattr(self.model.model, 'table_mass'))

    def test_parse_datablock_overflow(self):
        # Set high mass bins to force overflow
        parameters = {
            key: 0.5 for key in self.model.sfh_bin_keys  # logmass=0.5 => mass > 1 in total
        }
        parameters['alpha_powerlaw'] = 1.0
        parameters['ism_metallicity_today'] = 0.02

        db = DataBlock.from_dict({"parameters": parameters})
        status, overflow_value = self.model.parse_datablock(db)

        self.assertEqual(status, 0)
        self.assertGreater(overflow_value, 1.0)


class TestFixedCosmicTimeSFH(unittest.TestCase):

    def setUp(self):
        # Example lookback time bins (in Gyr)
        self.lookback_bins = np.array([1.0, 2.5, 5.0]) * u.Gyr
        self.model = sfh.FixedCosmicTimeSFH(
            self.lookback_bins, ism_metallicity_today=0.02)

    def test_initialization(self):
        # Check that number of keys matches number of inner bins
        expected_keys = len(self.lookback_bins)
        self.assertEqual(len(self.model.sfh_bin_keys), expected_keys)

        # Check keys are in free_params and within expected bounds
        for key in self.model.sfh_bin_keys:
            self.assertIn(key, self.model.free_params)
            bounds = self.model.free_params[key]
            self.assertEqual(bounds, [0.0, 0.5, 1.0])

    def test_parse_datablock_valid(self):
        # Simulate coefficient values that sum to < 1 through recursive scheme
        parameters = {key: 0.5 for key in self.model.sfh_bin_keys}
        parameters['alpha_powerlaw'] = 1.0
        parameters['ism_metallicity_today'] = 0.02
        db = DataBlock.from_dict({"parameters": parameters})
        status, _ = self.model.parse_datablock(db)
        self.assertEqual(status, 1)
        # Check mass table is updated
        self.assertTrue(hasattr(self.model.model, "table_mass"))
        self.assertEqual(len(self.model.model.table_mass), len(self.model.bin_masses) + 2)

    def test_update_mass_recursive(self):
        # Test how update_mass distributes values recursively
        self.model.bin_masses = np.zeros_like(self.model.bin_masses)
        self.model.update_mass(0, 0.5)
        self.model.update_mass(1, 0.5)
        self.model.update_mass(2, 0.5)

        expected = [
            0.5,
            0.5 * (1 - 0.5),            # 0.25
            0.5 * (1 - 0.5 - 0.25),     # 0.125
        ]
        np.testing.assert_allclose(self.model.bin_masses, expected, rtol=1e-6)


class TestFlexibleCosmicTimeSFH(unittest.TestCase):

    def setUp(self):
        self.n_bins = 4  # Choose a small number of bins for simplicity
        self.model = sfh.FlexibleCosmicTimeSFH(self.n_bins,
                                               ism_metallicity_today=0.02)

    def test_initialization(self):
        # One bin per n_bins, so expect n_bins sfh_bin_keys
        self.assertEqual(len(self.model.sfh_bin_keys), self.n_bins)

        # Ensure all keys are correctly registered
        for i in range(1, self.n_bins + 1):
            key = f"coeff_{i}"
            self.assertIn(key, self.model.free_params)
            self.assertEqual(self.model.free_params[key], [0.0, 0.5, 1.0])

        # Time should be monotonic
        self.assertTrue(np.all(np.diff(self.model.time.to_value(u.Gyr)) >= 0))

    def test_parse_datablock_valid(self):
        # Valid parameters that add up to less than 1 recursively
        parameters = {f"coeff_{i+1}": 0.5 for i in range(self.n_bins)}
        parameters["alpha_powerlaw"] = 1.0
        parameters["ism_metallicity_today"] = 0.02

        db = DataBlock.from_dict({"parameters": parameters})
        status, _ = self.model.parse_datablock(db)
        self.assertEqual(status, 1)

        self.assertTrue(hasattr(self.model.model, "table_mass"))
        self.assertEqual(len(self.model.model.table_mass), len(self.model.time))

    def test_update_mass_recursive(self):
        # Test recursive accumulation of mass
        self.model.bin_masses = np.zeros_like(self.model.bin_masses)
        self.model.update_mass(0, 0.5)
        self.model.update_mass(1, 0.5)
        self.model.update_mass(2, 0.5)

        expected = [
            0.5,
            0.5 * (1 - 0.5),            # 0.25
            0.5 * (1 - 0.5 - 0.25),     # 0.125
        ]
        np.testing.assert_allclose(self.model.bin_masses[:3], expected, rtol=1e-6)


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

        db = DataBlock.from_dict({"parameters": parameters})
        status, info = self.model.parse_datablock(db)

        self.assertEqual(status, 1)
        self.assertIsNone(info)
        self.assertTrue(hasattr(self.model.model, "table_mass"))
        self.assertEqual(len(self.model.model.table_mass), len(self.model.lookback_time))

    def test_parse_datablock_monotonicity_error(self):
        # Use large sSFR to force decreasing cumulative mass (non-monotonic)
        values = [0.0, 0.0, 0.0]  # logssfr = 0 => sSFR = 1 => mass_frac = 1 - lt_yr * 1 → < 0
        parameters = {
            key: val for key, val in zip(self.model.sfh_bin_keys, values)
        }
        parameters["alpha_powerlaw"] = 1.0
        parameters["ism_metallicity_today"] = 0.02

        db = DataBlock.from_dict({"parameters": parameters})
        status, overflow_val = self.model.parse_datablock(db)

        self.assertEqual(status, 0)
        self.assertGreater(overflow_val, 1.0)


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

        db = DataBlock.from_dict({"parameters": parameters})
        status, info = self.model.parse_datablock(db)

        self.assertEqual(status, 1)
        self.assertIsNone(info)
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

        db = DataBlock.from_dict({"parameters": parameters})
        status, overflow = self.model.parse_datablock(db)

        self.assertEqual(status, 0)
        self.assertGreater(overflow, 1.0)


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
        db = DataBlock.from_dict({"parameters": parameters})
        status, info = self.model.parse_datablock(db)

        self.assertEqual(status, 1)
        self.assertIsNone(info)

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
        db = DataBlock.from_dict({"parameters": parameters})
        self.model.parse_datablock(db)

        mass = self.model.model.table_mass.to_value(u.Msun)
        self.assertTrue(np.all(np.diff(mass) >= 0))


if __name__ == "__main__":
    unittest.main()