import os
import json
import unittest
import tempfile

import numpy as np
from astropy.table import Table

# New API (adjust import path if your module name differs)
from besta.postprocess import (
    read_results_file,
    summarize_results,
    ResultsSummary,
)


class TestPostprocessingNew(unittest.TestCase):
    """
    Unit tests for the refactored BESTA post-processing module.

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
            except Exception:
                pass

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

    def test_read_results_file(self):
        path = self._get_results_path()
        table = read_results_file(path)

        self.assertIsInstance(table, Table)
        self.assertGreater(len(table), 0, "Results table is empty")

        # Basic column expectations
        self.assertIn("post", table.colnames, "Missing 'post' column")
        # The synthetic file has prior; a real sfh.txt might not.
        # So only assert prior if present in the file.
        # Parameter columns: at least one 'section--param' should exist
        self.assertTrue(any("--" in c for c in table.colnames), "No parameter columns found (expected '--' delimiter).")

    def test_summarize_results_and_exports(self):
        path = self._get_results_path()
        table = read_results_file(path)

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
