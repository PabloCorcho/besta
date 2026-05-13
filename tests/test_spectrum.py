import os
import unittest
import numpy as np
from astropy import units as u

from besta import spectrum


class TestPipelineModule(unittest.TestCase):
    def test_mask_telluric_regions(self):
        wl = np.linspace(6000, 8000, 10)
        flux = np.ones_like(wl)
        err = np.ones_like(wl) * 0.1
        weights = np.ones_like(wl)
        new_w, mask, bands = spectrum.mask_telluric_regions(
            wl, weight=weights, return_mask=True, pad=0.0
        )
        # Ensure some mask applied within known band (~6860-6950)
        self.assertTrue(mask.any())
        self.assertTrue(np.all(new_w[mask] == 0.0))

    def test_mask_strong_emission_lines(self):
        wl = np.linspace(4990, 5010, 50)
        flux = np.zeros_like(wl)
        err = np.ones_like(wl) * 0.1
        weights = np.ones_like(wl)
        # Inject a "line" at center
        flux[25] = 10.0
        new_w, mask, lines = spectrum.mask_strong_emission_lines(
            wl,
            flux,
            err,
            weights,
            redshift=0.0,
            return_mask=True,
            return_lines_masked=True,
            pad=0.0,
        )

        self.assertTrue(mask.any())
        self.assertTrue(np.all(new_w[mask] == 0.0))

    def test_emission_line_list(self):
        # Create a file containing some emission lines

        with open("test_emission_lines.dat", "w") as f:
            f.write("# name rest_wavelength[AA] default_half_width[AA]\n")
            f.write("Halpha 6562.8 10.0\n")
            f.write("Hbeta 4861.3 10.0\n")

        eline_list = spectrum.EmissionLineList.from_file(
            "test_emission_lines.dat", format="ascii"
        )

        self.assertTrue(eline_list.names == ["Halpha", "Hbeta"])
        self.assertTrue(
            np.array_equal(eline_list.rest_wavelengths, np.array([6562.8, 4861.3]))
        )
        self.assertTrue(
            np.array_equal(eline_list.default_half_widths, np.array([10.0, 10.0]))
        )

        eline_list.to_table(filename="test_emission_lines_out.dat", format="ascii")
        # Read back the file and check it matches
        eline_list2 = spectrum.EmissionLineList.from_file(
            "test_emission_lines_out.dat", format="ascii"
        )
        self.assertTrue(eline_list2.names == eline_list.names)
        self.assertTrue(
            np.array_equal(eline_list2.rest_wavelengths, eline_list.rest_wavelengths)
        )
        self.assertTrue(
            np.array_equal(
                eline_list2.default_half_widths, eline_list.default_half_widths
            )
        )

        # Test methods
        redshifted_lines = eline_list.get_observed_wavelengths(redshift=0.1)
        self.assertTrue(
            np.array_equal(
                redshifted_lines,
                eline_list.rest_wavelengths * (1 + 0.1),
            )
        )

        line = eline_list.get_line_by_observed_wavelength(
            6562.8 * (1 + 0.1), redshift=0.1, tol=5.0
        )
        self.assertIsNotNone(line)
        self.assertEqual(line.name, "Halpha")

        line = eline_list.get_line_by_observed_wavelength(
            7000.8 * (1 + 0.1), redshift=0.1, tol=5.0
        )
        self.assertIsNone(line)

        dummy_wl = np.arange(6500, 6700, 1)
        mask = eline_list.to_mask(dummy_wl, redshift=0)
        self.assertTrue(mask.any())
        # Remove test files
        os.remove("test_emission_lines.dat")
        os.remove("test_emission_lines_out.dat")

    def test_emission_line_list_preserves_measurements(self):
        table = spectrum.Table(
            {
                "name": ["Ha"],
                "rest_wavelength": [6562.8],
                "default_half_width": [10.0],
                "flux": [1.5],
                "flux_error": [0.2],
                "rest_wavelength_error": [0.05],
                "flag": [2],
            }
        )

        eline_list = spectrum.EmissionLineList.from_table(table)

        self.assertEqual(len(eline_list), 1)
        self.assertEqual(eline_list[0].name, "Ha")
        self.assertAlmostEqual(eline_list[0].flux, 1.5)
        self.assertAlmostEqual(eline_list[0].flux_error, 0.2)
        self.assertAlmostEqual(eline_list[0].rest_wavelength_error, 0.05)
        self.assertEqual(eline_list[0].flag, 2)

    def test_estimate_continuum(self):
        rng = np.random.default_rng(1234)
        wl = np.linspace(4000, 5000, 500)
        flux = 2.0 + 0.001 * (wl - 4500)
        flux_with_noise = flux + rng.normal(0, 0.1, size=wl.size)
        err = np.ones_like(wl) * 0.1
        continuum, continuum_error = spectrum.estimate_continuum(
            wl, flux_with_noise, err, knot_spacing=100, use_log=False
        )
        self.assertTrue(np.isfinite(continuum).all())
        self.assertTrue(np.isfinite(continuum_error).all())

        median_diff = np.median(np.abs((flux - continuum) / err))
        self.assertTrue(median_diff < 0.5)

    def test_find_emission_lines(self):
        rng = np.random.default_rng(5678)
        delta_wl = 0.5
        wl = np.arange(4000, 5000, delta_wl)
        flux = np.ones_like(wl)
        flux += 0.001 * (wl - 4500)  # add a slight slope
        err = np.ones_like(wl) * 0.1
        flux += rng.normal(0, 0.1, size=wl.size)  # add noise

        # Inject some lines
        flux += spectrum._gaussian(wl, line_flux=10, center=4500, sigma=delta_wl * 4)
        flux += spectrum._gaussian(wl, line_flux=5, center=4900, sigma=delta_wl * 3)

        weights = np.ones_like(wl)
        lines_table, line_segm_map = spectrum.find_emission_lines(
            wl, flux, err, weights, redshift=0.0, lines=None, snr_threshold=5.0
        )

        self.assertTrue(lines_table is not None)
        self.assertEqual(len(lines_table), 2)
        self.assertTrue(np.isclose(lines_table["center"][0], 4500, atol=delta_wl / 2))
        self.assertTrue(np.isclose(lines_table["center"][1], 4900, atol=delta_wl / 2))
        
        self.assertTrue(np.isclose(lines_table["line_flux"][0], 10, rtol=0.2))
        self.assertTrue(np.isclose(lines_table["line_flux"][1], 5, rtol=0.2))

    def test_find_emission_lines_with_line_list_updates_names(self):
        delta_wl = 0.5
        wl = np.arange(4400, 5000, delta_wl)
        flux = np.ones_like(wl)
        err = np.ones_like(wl) * 0.1
        continuum = np.ones_like(wl)
        continuum_error = np.ones_like(wl) * 0.01

        flux += spectrum._gaussian(wl, line_flux=10, center=4500, sigma=delta_wl * 4)
        flux += spectrum._gaussian(wl, line_flux=5, center=4900, sigma=delta_wl * 3)

        lines = spectrum.EmissionLineList([
            spectrum.EmissionLine("LineA", 4500.0, 8.0),
            spectrum.EmissionLine("LineB", 4900.0, 8.0),
        ])

        lines_table, line_segm_map = spectrum.find_emission_lines(
            wl,
            flux,
            err,
            lines=lines,
            continuum=continuum,
            continuum_error=continuum_error,
            redshift=0.0,
            snr_threshold=5.0,
        )

        self.assertEqual(list(lines_table["line_name"]), ["LineA", "LineB"])
        self.assertEqual([line.name for line in line_segm_map.lines], ["LineA", "LineB"])


    # ------------------------------------------------------------------
    # Watershed deblending tests
    # ------------------------------------------------------------------

    def test_watershed_1d_splits_two_peaks(self):
        """Two peaks separated by a trough must be assigned to different labels."""
        # Build a simple SNR profile: peak at index 2 (label 1) and peak at index 7 (label 2)
        signal = np.array([0, 3, 10, 5, 1, 5, 10, 3, 0], dtype=float)
        markers = np.zeros(len(signal), dtype=int)
        markers[2] = 1   # seed for left peak
        markers[7] = 2   # seed for right peak
        mask = signal > 0

        labels = spectrum._watershed_1d(signal, markers, mask)

        # Every bright pixel should be labeled
        self.assertTrue(np.all(labels[mask] > 0))
        # Left half should belong to label 1, right half to label 2
        self.assertTrue(np.all(labels[:4][mask[:4]] == 1))
        self.assertTrue(np.all(labels[5:][mask[5:]] == 2))
        # Boundary pixel (index 4) is the trough; assigned to either side is acceptable
        self.assertIn(labels[4], [0, 1, 2])

    def test_watershed_1d_single_peak(self):
        """A single seed should flood the entire masked region."""
        signal = np.array([0, 2, 5, 8, 5, 2, 0], dtype=float)
        markers = np.zeros(len(signal), dtype=int)
        markers[3] = 1
        mask = signal > 0

        labels = spectrum._watershed_1d(signal, markers, mask)

        self.assertTrue(np.all(labels[mask] == 1))
        self.assertTrue(np.all(labels[~mask] == 0))

    def test_watershed_1d_respects_mask(self):
        """Pixels outside the mask must never be labeled."""
        signal = np.array([10, 8, 5, 3, 1], dtype=float)
        markers = np.array([1, 0, 0, 0, 0], dtype=int)
        mask = np.array([True, True, False, False, False])

        labels = spectrum._watershed_1d(signal, markers, mask)

        self.assertEqual(labels[0], 1)
        self.assertEqual(labels[1], 1)
        self.assertEqual(labels[2], 0)
        self.assertEqual(labels[3], 0)
        self.assertEqual(labels[4], 0)

    def test_find_emission_lines_deblend_blended_doublet(self):
        """Two close lines that merge into one bright region must be split by watershed."""
        delta_wl = 0.5
        wl = np.arange(4480, 4540, delta_wl)
        continuum = np.ones_like(wl)
        continuum_error = np.ones_like(wl) * 0.01

        # Two broad lines separated by 12 Å. They produce two local peaks but
        # remain one contiguous bright region at low SNR threshold.
        line_sep = 12.0
        sigma = 3.5
        flux = continuum.copy()
        flux += spectrum._gaussian(wl, line_flux=15, center=4500.0,            sigma=sigma)
        flux += spectrum._gaussian(wl, line_flux=14, center=4500.0 + line_sep, sigma=sigma)
        err = np.ones_like(wl) * 0.1

        # Without deblending, the two lines fuse into one detection
        table_no_deblend, _ = spectrum.find_emission_lines(
            wl, flux, err,
            continuum=continuum, continuum_error=continuum_error,
            redshift=0.0, snr_threshold=1.5,
            deblend_lines=False,
        )
        self.assertEqual(len(table_no_deblend), 1,
                         "Expected a single fused detection without deblending")

        # With deblending, the two lines must be resolved
        table_deblend, _ = spectrum.find_emission_lines(
            wl, flux, err,
            continuum=continuum, continuum_error=continuum_error,
            redshift=0.0, snr_threshold=1.5,
            deblend_lines=True,
        )
        self.assertEqual(len(table_deblend), 2,
                         "Expected two separate detections after watershed deblending")

        centers = sorted(table_deblend["center"])
        self.assertAlmostEqual(centers[0], 4500.0,            delta=2.0)
        self.assertAlmostEqual(centers[1], 4500.0 + line_sep, delta=2.0)

    def test_find_emission_lines_deblend_disabled(self):
        """deblend_lines=False must leave well-separated lines unchanged."""
        rng = np.random.default_rng(99)
        delta_wl = 0.5
        wl = np.arange(4000, 5000, delta_wl)
        continuum = np.ones_like(wl)
        continuum_error = np.ones_like(wl) * 0.01
        flux = continuum.copy()
        flux += spectrum._gaussian(wl, line_flux=10, center=4300, sigma=delta_wl * 4)
        flux += spectrum._gaussian(wl, line_flux=8,  center=4800, sigma=delta_wl * 4)
        flux += rng.normal(0, 0.05, size=wl.size)
        err = np.ones_like(wl) * 0.1

        table, _ = spectrum.find_emission_lines(
            wl, flux, err,
            continuum=continuum, continuum_error=continuum_error,
            redshift=0.0, snr_threshold=3.0,
            deblend_lines=False,
        )
        self.assertEqual(len(table), 2)

    def test_find_emission_lines_deblend_no_extra_splits(self):
        """Watershed must not introduce spurious extra lines for clean isolated detections."""
        delta_wl = 0.5
        wl = np.arange(4000, 5000, delta_wl)
        continuum = np.ones_like(wl)
        continuum_error = np.ones_like(wl) * 0.01
        flux = continuum.copy()
        flux += spectrum._gaussian(wl, line_flux=10, center=4300, sigma=delta_wl * 4)
        flux += spectrum._gaussian(wl, line_flux=8,  center=4800, sigma=delta_wl * 4)
        err = np.ones_like(wl) * 0.1

        table, _ = spectrum.find_emission_lines(
            wl, flux, err,
            continuum=continuum, continuum_error=continuum_error,
            redshift=0.0, snr_threshold=3.0,
            deblend_lines=True,
        )
        self.assertEqual(len(table), 2,
                         "Watershed must not split single-peak detections")


if __name__ == "__main__":
    from besta.logging import setup_logging
    setup_logging(level="DEBUG")
    unittest.main()
