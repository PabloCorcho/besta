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

            # 1D PDF presence (by section-qualified names)
            qualified_names = [
                ".".join((section, name))
                for section, name in zip(
                    summary.parameter_sections, summary.parameter_names
                )
            ]
            for nm in qualified_names:
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

            for nm in qualified_names:
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


class TestPhysicalSFHPostprocessing(unittest.TestCase):
    """Latent -> physical conversion of SFH parameters in results tables."""

    mass_fractions = (0.1, 0.5, 0.9)

    @staticmethod
    def _ini(use_transforms=True, redshift=0.1):
        # Values parsed as in ``besta.io.Reader`` (strings from the ini file)
        section = {
            "file": "full_spectral_fit.py",
            "SFHModel": "FixedMassFracSFH",
            "SFHArgs": "(0.1, 0.5, 0.9)",
            "use_transforms": "T" if use_transforms else "F",
            "redshift": str(redshift),
        }
        return {
            "pipeline": {"modules": "FullSpectralFit"},
            "FullSpectralFit": {k: io._parse_value(v) for k, v in section.items()},
        }

    def setUp(self):
        from besta import sfh
        self.model = sfh.FixedMassFracSFH(
            np.array(self.mass_fractions), ism_metallicity_today=0.02,
            use_transforms=True)
        rng = np.random.default_rng(7)
        n = 400
        self.latent = rng.uniform(size=(n, 3))
        self.columns = [f"stars.sfh--{key}" for key in self.model.sfh_bin_keys]
        table = Table()
        for index, col in enumerate(self.columns):
            table[col] = self.latent[:, index]
        table["stars.sfh--alpha_powerlaw"] = rng.uniform(0, 3, n)
        table["extra--stellar_mass"] = rng.normal(10, 0.1, n)
        table["post"] = rng.normal(-100, 1, n)
        self.table = table

    def test_latent_to_physical_table(self):
        from besta.postprocess import latent_to_physical_table
        original = self.table.copy()
        physical = latent_to_physical_table(self.table, self.model)

        expected = self.model.to_physical_batch(self.latent)
        for index, col in enumerate(self.columns):
            np.testing.assert_allclose(physical[col], expected[:, index])
        # Other columns untouched, input table not modified
        for col in ("stars.sfh--alpha_powerlaw", "extra--stellar_mass", "post"):
            np.testing.assert_array_equal(physical[col], self.table[col])
        for col in self.table.colnames:
            np.testing.assert_array_equal(self.table[col], original[col])
        self.assertEqual(physical.meta["besta_sfh_space"], "physical")

        # Converting twice is a no-op
        again = latent_to_physical_table(physical, self.model)
        for col in self.columns:
            np.testing.assert_array_equal(again[col], physical[col])

        # Optional copies of the latent columns
        with_latent = latent_to_physical_table(self.table, self.model, keep_latent=True)
        for col in self.columns:
            np.testing.assert_array_equal(with_latent["latent_" + col], self.table[col])

    def test_unmapped_samples_are_dropped_by_summary(self):
        from besta.postprocess import latent_to_physical_table
        table = self.table.copy()
        table[self.columns[0]][0] = 1.5  # outside the latent support
        physical = latent_to_physical_table(table, self.model)
        self.assertTrue(np.isnan(physical[self.columns[0]][0]))
        summary = summarize_results(physical, compute_1d=False)
        self.assertEqual(summary.n_samples, len(table) - 1)

    def test_summarize_results_in_physical_space(self):
        from besta.postprocess import latent_to_physical_table
        direct = summarize_results(self.table, sfh_model=self.model, compute_1d=False)
        converted = summarize_results(
            latent_to_physical_table(self.table, self.model), compute_1d=False)
        np.testing.assert_allclose(direct.percentiles_values, converted.percentiles_values)
        self.assertEqual(direct.extra_info["sfhspace"], "physical")

        # Physical times (Gyr) differ from the latent unit-cube values
        latent = summarize_results(self.table, compute_1d=False)
        index = direct.parameter_keys.index(self.columns[-1])
        self.assertGreater(direct.percentiles_values[index, 0], 1.0)
        self.assertLessEqual(latent.percentiles_values[index, -1], 1.0)
        # The MAP sample is the same (constant Jacobian)
        self.assertEqual(direct.map_index, latent.map_index)

    def test_build_sfh_model_from_ini(self):
        from besta.config import cosmology
        from besta.postprocess import build_sfh_model, find_sfh_modules
        ini = self._ini(use_transforms=True, redshift=0.1)
        self.assertEqual(find_sfh_modules(ini), ["FullSpectralFit"])

        model = build_sfh_model(ini)
        self.assertEqual(type(model).__name__, "FixedMassFracSFH")
        self.assertTrue(model.use_transforms)
        self.assertEqual(model.sfh_bin_keys, self.model.sfh_bin_keys)
        self.assertAlmostEqual(
            model.today.to_value("Gyr"), cosmology.age(0.1).to_value("Gyr"))

        self.assertFalse(build_sfh_model(self._ini(use_transforms=False)).use_transforms)

        with self.assertRaises(ValueError):
            build_sfh_model({"pipeline": {"modules": "Other"}, "Other": {"x": 1}})

    def test_to_physical_table_from_reader(self):
        from besta.postprocess import to_physical_table
        reader = io.Reader.__new__(io.Reader)
        reader.ini = self._ini(use_transforms=True, redshift=0.0)
        reader.results_table = self.table
        physical = to_physical_table(reader)
        expected = self.model.to_physical_batch(self.latent)
        np.testing.assert_allclose(physical[self.columns[1]], expected[:, 1])



def _write_cosmosis_results(path, latent, extra, use_transforms=True):
    """Minimal CosmoSIS text results file for a FixedMassFracSFH run."""
    keys = ["t_at_frac_0.1000", "t_at_frac_0.5000", "t_at_frac_0.9000"]
    columns = [f"stars.sfh--{k}" for k in keys] + ["extra--stellar_mass", "prior", "post"]
    values = "0.0 0.5 1.0" if use_transforms else "0.001 5.0 13.0"
    lines = ["#" + "\t".join(columns) + "\n",
             "#sampler=emcee\n", "#n_varied=3\n",
             "## START_OF_PARAMS_INI\n",
             "## [runtime]\n", "## sampler = emcee\n", "## \n",
             "## [output]\n", f"## filename = {path}\n", "## format = text\n", "## \n",
             "## [pipeline]\n", "## modules = FullSpectralFit\n",
             "## values = values.ini\n", "## \n",
             "## [FullSpectralFit]\n", "## file = full_spectral_fit.py\n",
             "## redshift = 0.0\n", "## sfhmodel = FixedMassFracSFH\n",
             "## sfhargs = (0.1, 0.5, 0.9)\n",
             f"## use_transforms = {'T' if use_transforms else 'F'}\n", "## \n",
             "## END_OF_PARAMS_INI\n",
             "## START_OF_VALUES_INI\n", "## [stars.sfh]\n"]
    lines += [f"## {k} = {values}\n" for k in keys]
    lines += ["## alpha_powerlaw = 1.0\n", "## \n", "## END_OF_VALUES_INI\n",
              "## START_OF_PRIORS_INI\n", "## END_OF_PRIORS_INI\n"]
    data = np.column_stack([latent, extra])
    with open(path, "w") as file:
        file.writelines(lines)
        np.savetxt(file, data, delimiter="\t", fmt="%.17g")
        file.write("#evaluations=100\n#complete=1\n")
    return columns


class TestPhysicalResultsFile(unittest.TestCase):
    """CosmoSIS text results written in physical SFH space."""

    def setUp(self):
        from besta import sfh
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "results.txt")
        rng = np.random.default_rng(11)
        self.latent = rng.uniform(size=(200, 3))
        self.extra = np.column_stack([rng.normal(10, 0.1, 200),
                                      np.zeros(200), rng.normal(-50, 1, 200)])
        self.columns = _write_cosmosis_results(self.path, self.latent, self.extra)
        self.model = sfh.FixedMassFracSFH(
            np.array([0.1, 0.5, 0.9]), ism_metallicity_today=0.02,
            use_transforms=True, redshift=0.0)

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _values_block(path):
        with open(path) as file:
            lines = file.readlines()
        start = next(i for i, l in enumerate(lines) if "START_OF_VALUES_INI" in l)
        end = next(i for i, l in enumerate(lines) if "END_OF_VALUES_INI" in l)
        block = {}
        for line in lines[start + 1:end]:
            body = line[2:].strip()
            if "=" in body:
                key, value = body.split("=", 1)
                block[key.strip()] = value.strip()
        return block

    def test_convert_results_file(self):
        from besta.postprocess import (
            build_sfh_model, convert_results_file, physical_results_path)
        output = convert_results_file(self.path)
        self.assertEqual(output, physical_results_path(self.path))
        self.assertTrue(output.endswith("results_physical.txt"))

        table = io.read_results_file(output)
        original = io.read_results_file(self.path)
        self.assertEqual(table.colnames, original.colnames)
        expected = self.model.to_physical_batch(self.latent)
        for index, col in enumerate(self.columns[:3]):
            np.testing.assert_allclose(table[col], expected[:, index], rtol=1e-15)
        for col in self.columns[3:]:
            np.testing.assert_array_equal(table[col], original[col])
        self.assertEqual(table.meta["besta_sfh_space"], "physical")
        self.assertEqual(table.meta["complete"], 1)

        # The embedded configuration describes the physical table
        ini = io.Reader.read_ini_file_from_results(output)
        self.assertFalse(ini["FullSpectralFit"]["use_transforms"])
        self.assertEqual(ini["output"]["filename"], output)
        self.assertFalse(build_sfh_model(ini).use_transforms)
        values = self._values_block(output)
        t_min, t_max = self.model.time_bounds
        for key in self.model.sfh_bin_keys:
            low, start, high = map(float, values[key].split())
            self.assertEqual((low, high), (t_min, t_max))
            self.assertTrue(low < start < high)
        self.assertEqual(values["alpha_powerlaw"], "1.0")

        # A physical table is not converted again
        self.assertIsNone(convert_results_file(output, output + ".again"))

    def test_nothing_to_convert_without_transforms(self):
        from besta.postprocess import convert_results_file
        path = os.path.join(self.tmp.name, "plain.txt")
        _write_cosmosis_results(path, self.latent * 10, self.extra,
                                use_transforms=False)
        self.assertIsNone(convert_results_file(path))
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "plain_physical.txt")))

    def test_output_checks(self):
        from besta.postprocess import convert_results_file
        with self.assertRaises(ValueError):
            convert_results_file(self.path, self.path)
        output = convert_results_file(self.path)
        with self.assertRaises(FileExistsError):
            convert_results_file(self.path, output, overwrite=False)

    def test_load_and_summarize_in_physical_space_by_default(self):
        from besta.postprocess import load_physical_results, summarize_results_file
        table = load_physical_results(self.path)
        expected = self.model.to_physical_batch(self.latent)
        np.testing.assert_allclose(table[self.columns[2]], expected[:, 2])
        self.assertEqual(table.meta["besta_sfh_space"], "physical")

        summary = summarize_results_file(self.path, compute_1d=False)
        self.assertEqual(summary.extra_info["sfhspace"], "physical")
        latent = summarize_results_file(self.path, physical=False, compute_1d=False)
        self.assertNotIn("sfhspace", latent.extra_info)

    def test_command_line(self):
        from besta.cli.besta_to_physical import main
        output = os.path.join(self.tmp.name, "converted.txt")
        self.assertEqual(main([self.path, "-o", output, "--quiet"]), 0)
        self.assertTrue(os.path.exists(output))
        self.assertEqual(main([self.path, "-o", output, "--no-overwrite", "--quiet"]), 1)
        self.assertEqual(main([self.path, self.path, "-o", output]), 2)

if __name__ == "__main__":
    unittest.main()
