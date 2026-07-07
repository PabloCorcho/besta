import os
import json
import unittest
import tempfile

import numpy as np
from astropy.table import Table
from astropy import units as u

from besta.postprocess import (
    summarize_results,
    ResultsSummary,
)
from besta import io

class TestPostprocessingUtils(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.dirname(__file__)
    
    @classmethod
    def tearDownClass(cls):
        pass

    def test_as_float_array_func(self):
        from besta.postprocess import _as_float_array

        # Test with a list of floats
        arr = [1.0, 2.0, 3.0]
        result = _as_float_array(arr)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, float)
        np.testing.assert_array_equal(result, np.array(arr))

        # Test with a astropy Quantity
        arr = np.array([4.0, 5.0, 6.0]) << u.m
        result = _as_float_array(arr)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, float)
        np.testing.assert_array_equal(result, arr.value)

    def test_normalize_weights(self):
        from besta.postprocess import normalize_weights

        # Test with uniform weights
        weights = np.array([1.0, 1.0, 1.0])
        normalized = normalize_weights(weights)
        np.testing.assert_array_almost_equal(normalized, np.array([1/3, 1/3, 1/3]))

        # Test with non-uniform weights
        weights = np.array([0.5, 1.5, 2.0])
        normalized = normalize_weights(weights)
        expected = weights / np.sum(weights)
        np.testing.assert_array_almost_equal(normalized, expected)

    def test_check_multimodal_pdf(self):
        from besta.postprocess import check_multimodal_pdf

        # Create a simple bimodal distribution
        x = np.linspace(-5, 5, 200)
        delta_x = x[1] - x[0]
        f = np.exp(-0.5 * ((x + 2) / 0.5) ** 2) + np.exp(-0.5 * ((x - 2) / 0.5) ** 2)

        n_maxima, maxima_x, maxima_val = check_multimodal_pdf(x, f)

        print(n_maxima, maxima_x, maxima_val, delta_x)
        self.assertEqual(n_maxima, 2)
        self.assertTrue(np.allclose(maxima_x, [2, -2], atol=delta_x / 3))
        self.assertTrue(np.allclose(maxima_val, [1.0, 1.0], rtol=0.1))

class TestPostprocessing(unittest.TestCase):
    """
    Unit tests for the BESTA post-processing module.

    This test suite checks:
    - Reading CosmoSIS-style text results into an Astropy Table
    - Summarisation outputs (mean/MAP/cov/corr, percentiles, PDFs)
    - FITS + JSON export
    - Evidence estimation given (post, prior) columns
    """

    @classmethod
    def setUpClass(cls):
        cls.test_dir = os.path.dirname(__file__)
        cls.data_file = os.path.join(cls.test_dir, "test_data", "sfh.txt")

        # If the repository does not ship the legacy test_data/sfh.txt,
        # the test will create a temporary results file for I/O tests.
        cls._synthetic_results_path = None

    @classmethod
    def tearDownClass(cls):
        if cls._synthetic_results_path is not None and os.path.isfile(cls._synthetic_results_path):
            try:
                os.remove(cls._synthetic_results_path)
            except Exception as e:
                print(f"Warning: Failed to remove synthetic results file: {e}")

    def _make_synthetic_results_file(self) -> str:
        """
        Create a CosmoSIS-like results text file with:
        - Header starting with '#'
        - Tab-separated columns
        - Columns: post, prior, sfh--tau, sfh--age, dust--av
        """
        rng = np.random.default_rng(12345)
        n = 2000

        tau = rng.uniform(0.1, 10.0, size=n)
        age = rng.uniform(1.0, 13.5, size=n)
        av = rng.uniform(0.0, 2.0, size=n)

        # Simple synthetic log-likelihood around some "truth"
        tau0, age0, av0 = 3.0, 8.0, 0.4
        # Gaussian-ish loglike (up to constant)
        loglike = -0.5 * (
            ((tau - tau0) / 0.8) ** 2
            + ((age - age0) / 1.2) ** 2
            + ((av - av0) / 0.2) ** 2
        )

        # Weak log-prior (uniform-like, still finite): constant in region
        logprior = np.zeros_like(loglike)

        logpost = loglike + logprior

        fd, path = tempfile.mkstemp(suffix=".txt", prefix="besta_test_results_")
        os.close(fd)

        with open(path, "w", encoding="utf-8") as f:
            f.write("#post\tprior\tsfh--tau\tsfh--age\tdust--av\n")
            for i in range(n):
                f.write(f"{logpost[i]:.10e}\t{logprior[i]:.10e}\t{tau[i]:.10e}\t{age[i]:.10e}\t{av[i]:.10e}\n")

        self.__class__._synthetic_results_path = path
        return path

    def _get_results_path(self) -> str:
        if os.path.isfile(self.data_file):
            return self.data_file
        return self._make_synthetic_results_file()

    def test_summarize_results_and_exports(self):
        path = self._get_results_path()
        table = io.read_results_file(path)

        with tempfile.TemporaryDirectory() as tmp:
            out_fits = os.path.join(tmp, "besta_summary.fits")
            out_json = os.path.join(tmp, "besta_summary.json")
            out_corner = os.path.join(tmp, "corner.png")

            # Evidence estimation is only possible if a prior column exists.
            estimate_evidence = "prior" in table.colnames

            summary = summarize_results(
                table,
                output_fits=out_fits,
                output_json=out_json,
                posterior_key="post",
                parameter_prefix="--",
                compute_1d=True,
                compute_2d=True,
                # Pick 2D pairs from existing parameter keys if possible
                parameter_key_pairs=[
                    (k0, k1)
                    for (k0, k1) in zip(
                        [c for c in table.colnames if "--" in c][:-1],
                        [c for c in table.colnames if "--" in c][1:],
                    )
                ][:1],  # one pair is enough for a unit test
                kde_1d=False,  # keep test fast and deterministic
                kde_2d=False,
                # optional evidence hook (if you integrated it into summarize_results)
                estimate_evidence=estimate_evidence,
                evidence_method="laplace",
                logprior_key="prior",
            )

            self.assertIsInstance(summary, ResultsSummary)
            self.assertGreater(summary.n_samples, 0)
            self.assertEqual(summary.samples.shape[0], len(summary.parameter_keys))
            self.assertEqual(summary.samples.shape[1], summary.n_samples)

            # Check summary arrays
            self.assertEqual(summary.mean.shape[0], len(summary.parameter_keys))
            self.assertEqual(summary.map.shape[0], len(summary.parameter_keys))
            self.assertEqual(summary.covariance.shape, (len(summary.parameter_keys), len(summary.parameter_keys)))
            self.assertEqual(summary.correlation.shape, (len(summary.parameter_keys), len(summary.parameter_keys)))

            # Correlation diagonal should be ~ 1
            self.assertTrue(np.allclose(np.diag(summary.correlation), 1.0, atol=1e-6))

            # Percentiles shape
            self.assertEqual(summary.percentiles_values.shape, (len(summary.parameter_keys), len(summary.percentiles)))

            # 1D PDF presence (by short names)
            for nm in summary.parameter_names:
                self.assertIn(nm, summary.pdf_1d)
                self.assertIn("grid", summary.pdf_1d[nm])
                self.assertIn("hist_pdf", summary.pdf_1d[nm])
                self.assertIn("n_maxima", summary.pdf_1d[nm])
                self.assertIn("map", summary.pdf_1d[nm])

                # New summary products: fixed-mass HDIs and 1D mode locations
                self.assertIn(nm, summary.hdi_intervals_68)
                self.assertIn(nm, summary.hdi_intervals_95)
                self.assertIn(nm, summary.map_1d)

                self.assertIsInstance(summary.hdi_intervals_68[nm], list)
                self.assertIsInstance(summary.hdi_intervals_95[nm], list)
                self.assertGreaterEqual(len(summary.hdi_intervals_68[nm]), 1)
                self.assertGreaterEqual(len(summary.hdi_intervals_95[nm]), 1)

                self.assertEqual(
                    np.asarray(summary.map_1d[nm]).shape,
                    np.asarray(summary.pdf_1d[nm]["map"]).shape,
                )

            # FITS and JSON outputs exist
            self.assertTrue(os.path.isfile(out_fits), "FITS summary not written")
            self.assertTrue(os.path.isfile(out_json), "JSON summary not written")

            # JSON parses and has key fields
            with open(out_json, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.assertIn("parameter_keys", payload)
            self.assertIn("mean", payload)
            self.assertIn("covariance", payload)
            self.assertIn("pdf_1d", payload)
            self.assertIn("hdi_intervals_68", payload)
            self.assertIn("hdi_intervals_95", payload)
            self.assertIn("map_1d", payload)
            self.assertNotIn("hdi_intervals", payload)
            self.assertNotIn("hdi_mass", payload)

            for nm in summary.parameter_names:
                self.assertIn(nm, payload["hdi_intervals_68"])
                self.assertIn(nm, payload["hdi_intervals_95"])
                self.assertIn(nm, payload["map_1d"])

            # Evidence: only validate if prior exists and estimation was enabled
            if estimate_evidence:
                self.assertIn("evidence", payload)
                # evidence may be None if estimation failed; check method/logz if present
                if payload["evidence"] is not None:
                    self.assertIn("logz", payload["evidence"])
                    self.assertIn("method", payload["evidence"])

            # Corner plot (smoke test)
            summary.corner_plot(out_corner, max_points=2000, bins=40, show=False)
            self.assertTrue(os.path.isfile(out_corner), "Corner plot not written")


if __name__ == "__main__":
    unittest.main()
