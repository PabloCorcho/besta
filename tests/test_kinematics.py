import unittest

import numpy as np
from astropy import units as u

from besta import kinematics


class TestKinematics(unittest.TestCase):
    def test_gaussian_kernel_normalization_and_centered_percentile(self):
        kernel = kinematics.GaussianPixelKernel(velocity_scale=100.0, sigma_truncation=5.0)
        kernel.set_parameters(vel=0.0, sigma=200.0)

        self.assertIsNotNone(kernel.kernel_weight)
        self.assertTrue(np.isclose(np.sum(kernel.kernel_weight), 1.0, atol=1e-12))
        self.assertTrue(kernel.size % 2 == 1)

        # Percentiles are now returned relative to the central pixel.
        self.assertTrue(np.isclose(kernel.get_percentile_pixel(50.0), 0.0, atol=1e-2))
        self.assertTrue(np.isclose(kernel.get_percentile_velocity(50.0), 0.0, atol=1.0))

    def test_gaussian_delta_kernel_skips_convolution(self):
        kernel = kinematics.GaussianPixelKernel(velocity_scale=100.0)
        kernel.set_parameters(vel=0.0, sigma=0.0)

        x = np.linspace(0.0, 1.0, 64)
        y = kernel.convolve(x)

        self.assertTrue(kernel.skip_convolution)
        self.assertTrue(np.allclose(y, x, atol=0.0, rtol=0.0))

    def test_gausshermite_parse_parameters_from_datablock(self):
        block = {
            ("kinematics", "los_vel"): 30.0,
            ("kinematics", "los_sigma"): 150.0,
            ("kinematics", "los_h3"): 0.05,
            ("kinematics", "los_h4"): -0.02,
        }

        kernel = kinematics.GaussHermitePixelKernel(velocity_scale=100.0)
        kernel.parse_parameters(block)

        self.assertIsNotNone(kernel.kernel_weight)
        self.assertTrue(np.isclose(np.sum(kernel.kernel_weight), 1.0, atol=1e-10))
        self.assertGreater(kernel.edge_pixels, 0)

        spec = np.sin(np.linspace(0.0, 2 * np.pi, 128))
        conv = kernel.convolve(spec)
        self.assertEqual(conv.shape, spec.shape)
        self.assertTrue(np.isfinite(conv).all())

    def test_piecewise_kernel_parse_and_centered_percentile(self):
        kernel = kinematics.PieceWisePixelKernel(
            velocity_scale=100.0,
            velocity_bin_size=100.0,
            velocity_min=-200.0,
            velocity_max=200.0,
        )

        block = {}
        # Symmetric piecewise weights around zero velocity.
        block["kinematics", "vel_bin_0"] = 0.1
        block["kinematics", "vel_bin_1"] = 0.4
        block["kinematics", "vel_bin_2"] = 0.4
        block["kinematics", "vel_bin_3"] = 0.1

        kernel.parse_parameters(block)

        self.assertTrue(np.isclose(np.sum(kernel.kernel_weight), 1.0, atol=1e-10))
        self.assertTrue(np.isclose(kernel.get_percentile_velocity(50.0), 0.0, atol=100.0))

    def test_convolve_variable_gaussian_kernel_identity_limit(self):
        rng = np.random.default_rng(42)
        spec = rng.normal(size=(3, 128))

        # Tiny sigma means all rows are clamped to delta kernels.
        sigma_pixel = np.full(spec.shape[-1], 0.01)
        out = kinematics.convolve_variable_gaussian_kernel(spec, sigma_pixel)

        self.assertEqual(out.shape, spec.shape)
        self.assertTrue(np.allclose(out, spec, atol=1e-12, rtol=0.0))

    def test_convolve_variable_gaussian_kernel_preserves_units(self):
        spec = np.ones((2, 64)) * u.Unit("erg / (s cm2 Angstrom)")
        sigma_pixel = np.full(64, 0.8)

        out = kinematics.convolve_variable_gaussian_kernel(spec, sigma_pixel)

        self.assertTrue(hasattr(out, "unit"))
        self.assertEqual(out.unit, spec.unit)
        self.assertEqual(out.shape, spec.shape)


if __name__ == "__main__":
    unittest.main()
