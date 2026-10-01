"""BaseModule.luminosity_values matches Quantity.to_value with a cached factor."""
import types
import unittest

import numpy as np
from astropy import units as u

from besta.pipeline_modules.base_module import BaseModule


def make_module():
    module = types.SimpleNamespace(
        _default_luminosity_units=BaseModule._default_luminosity_units)
    module.luminosity_values = types.MethodType(BaseModule.luminosity_values, module)
    return module


class TestLuminosityValues(unittest.TestCase):

    def test_matches_to_value(self):
        module = make_module()
        values = np.random.default_rng(0).uniform(0.1, 2.0, 50)
        # SED unit as produced by compute_SED (Msun x per-Msun SSP units), plus others
        units = [u.Msun * (u.Lsun / u.AA / u.Msun), u.Lsun / u.AA,
                 u.erg / u.s / u.AA, u.W / u.nm]
        for unit in units:
            for dtype in (np.float64, np.float32):
                with self.subTest(unit=unit, dtype=dtype):
                    lum = values.astype(dtype) << unit
                    new = module.luminosity_values(lum)
                    ref = lum.to_value(BaseModule._default_luminosity_units)
                    if dtype is np.float64:
                        self.assertEqual(new.dtype, np.float64)
                    np.testing.assert_allclose(new, ref, rtol=1e-6 if dtype is np.float32
                                               else 1e-14)

    def test_factor_is_cached_per_unit(self):
        module = make_module()
        unit = u.Msun * (u.Lsun / u.AA / u.Msun)
        module.luminosity_values(np.ones(3) << unit)
        module.luminosity_values(np.ones(3) << unit)
        module.luminosity_values(np.ones(3) << u.Lsun / u.AA)
        self.assertEqual(len(module._luminosity_factor_cache), 2)

    def test_incompatible_unit_raises(self):
        with self.assertRaises(u.UnitConversionError):
            make_module().luminosity_values(np.ones(3) << u.Jy)


if __name__ == "__main__":
    unittest.main()
