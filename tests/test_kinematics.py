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

    def test_convolve_edge_padding(self):
        """``pad_mode='edge'`` removes the zero-padding flux drop at the edges
        and matches the default convolution away from them."""
        kernel = kinematics.GaussianPixelKernel(velocity_scale=50.0, sigma_truncation=5.0)
        kernel.set_parameters(vel=100.0, sigma=300.0)
        half_width = kernel.size // 2

        const = np.ones(200)
        padded = kernel.convolve(const, pad_mode="edge")
        default = kernel.convolve(const)
        self.assertEqual(padded.shape, const.shape)
        self.assertTrue(np.allclose(padded, 1.0))
        self.assertLess(default[0], 0.9)  # zero padding loses flux at the edge

        rng = np.random.default_rng(0)
        spec = rng.normal(size=300)
        self.assertTrue(np.allclose(
            kernel.convolve(spec, pad_mode="edge")[half_width:-half_width],
            kernel.convolve(spec)[half_width:-half_width]))

        spec_2d = np.tile(spec, (3, 1))
        conv_2d = kernel.convolve(spec_2d, pad_mode="edge")
        self.assertEqual(conv_2d.shape, spec_2d.shape)
        self.assertTrue(np.allclose(conv_2d[1], kernel.convolve(spec, pad_mode="edge")))

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

    def test_piecewise_kernel_is_pixel_aligned_for_non_integer_bounds(self):
        """Velocity bounds that are not a whole number of pixels
        (500 / 70 km/s) must still give an odd kernel centred on v = 0.

        Previously the bin edges were offset from the pixel grid (even kernel,
        -25 km/s spurious median for a symmetric LOSVD).
        """
        velscale = 70.0
        kernel = kinematics.PieceWisePixelKernel(
            velocity_scale=velscale,
            velocity_bin_size=100.0,
            velocity_min=-500.0,
            velocity_max=500.0,
        )
        n_bins = kernel.bin_ids.size
        self.assertEqual(kernel.half_width, 8)  # ceil(500 / 70)

        # Symmetric LOSVD about zero velocity
        weights = np.exp(-0.5 * ((np.arange(n_bins) - (n_bins - 1) / 2) / 1.5) ** 2)
        kernel.parse_parameters(
            {("kinematics", f"vel_bin_{i}"): w for i, w in enumerate(weights)})
        self.assertEqual(kernel.size % 2, 1)
        self.assertTrue(np.allclose(kernel.kernel_weight, kernel.kernel_weight[::-1]))
        self.assertTrue(np.isclose(kernel.get_percentile_velocity(50.0), 0.0, atol=1e-6))

        # All weight in the [100, 200] km/s bin shifts a line by ~ +150 km/s
        # (up to the pixel discretisation of the top-hat bin).
        shifted = np.zeros(n_bins)
        shifted[6] = 1.0
        kernel.parse_parameters(
            {("kinematics", f"vel_bin_{i}"): w for i, w in enumerate(shifted)})
        x = np.arange(401)
        line = np.exp(-0.5 * ((x - 200) / 3.0) ** 2)
        out = kernel.convolve(line)
        shift = (np.sum(x * out) / np.sum(out) - 200) * velscale
        self.assertLess(abs(shift - 150.0), 0.1 * velscale)

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
