import os
import shutil
import tempfile
import unittest

import numpy as np

from cosmosis import DataBlock

from besta.io import Reader
from besta.pipeline import MainPipeline
from besta.pipeline_modules.spectra_redshift_fit import SpectraRedshiftFitModule
from besta.sfh import DelayedTauSFH
from pst.SSP import PopStar


class TestSpectraRedshiftFitModule(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="besta-zfit-full-")
        cls.spectrum_path = os.path.join(cls.temp_dir, "test_redshift_full_sampling.dat")
        cls.true_redshift = 0.05
        cls.sfh_params = {
            "logtau": 0.5,
            "alpha_powerlaw": 0.2,
            "ism_metallicity_today": 0.02,
        }

        rng = np.random.default_rng(42)
        ssp = PopStar(IMF="cha")
        sfh_model = DelayedTauSFH()
        sfh_model.parse_free_params(cls.sfh_params)
        sed = sfh_model.model.compute_SED(ssp, t_obs=sfh_model.today).to_value(
            "1e-16 erg / (s Angstrom)"
        )

        wavelength_obs = ssp.wavelength.to_value("Angstrom") * (1.0 + cls.true_redshift)
        keep = (wavelength_obs >= 3500.0) & (wavelength_obs <= 7000.0)
        flux = sed[keep]
        error = np.maximum(np.abs(flux) / 50.0, np.nanmax(np.abs(flux)) * 1e-6)
        noisy_flux = flux + rng.normal(scale=error)

        cls.wavelength_range = [float(wavelength_obs[keep][0]), float(wavelength_obs[keep][-1])]
        np.savetxt(
            cls.spectrum_path,
            np.column_stack([wavelength_obs[keep], noisy_flux, error]),
            header="wavelength[AA] flux error",
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_recovers_redshift_from_delayed_tau_spectrum(self):
        config = {
            "SpectraRedshiftFit": {
                "file": SpectraRedshiftFitModule.get_path(),
                "redshift": 0.0,
                "inputSpectrum": self.spectrum_path,
                "SSPModel": "PopStar",
                "SSPModelArgs": "cha",
                "SSPDir": "None",
                "wlRange": self.wavelength_range,
                "SFHModel": "DelayedTauSFH",
            }
        }

        block = DataBlock()
        block["stars.sfh", "logtau"] = self.sfh_params["logtau"]
        block["stars.sfh", "alpha_powerlaw"] = self.sfh_params["alpha_powerlaw"]
        block["stars.sfh", "ism_metallicity_today"] = self.sfh_params["ism_metallicity_today"]

        module = SpectraRedshiftFitModule(config)
        self.assertEqual(module.execute(block), 0)

        recovered_redshift = block["redshift", "redshift"]
        self.assertTrue(np.isfinite(recovered_redshift))
        self.assertAlmostEqual(recovered_redshift, self.true_redshift, delta=0.01)

        flux_model, weights = module.make_observable(block)
        self.assertTrue(np.isfinite(flux_model[weights > 0]).all())
        self.assertTrue(np.isfinite(block["likelihoods", module.like_name]))

    def test_rejects_nonzero_input_redshift(self):
        config = {
            "SpectraRedshiftFit": {
                "file": SpectraRedshiftFitModule.get_path(),
                "redshift": 0.1,
                "inputSpectrum": self.spectrum_path,
                "SSPModel": "PopStar",
                "SSPModelArgs": "cha",
                "SSPDir": "None",
                "wlRange": self.wavelength_range,
                "SFHModel": "DelayedTauSFH",
            }
        }

        with self.assertRaisesRegex(ValueError, "redshift = 0.0"):
            SpectraRedshiftFitModule(config)

    def test_pipeline_recovers_redshift_from_synthetic_spectrum(self):
        output_root = os.path.join(self.temp_dir, "spectra_redshift_fit_result")
        values_path = os.path.join(self.temp_dir, "values.ini")
        with open(values_path, "w", encoding="utf-8") as handle:
            handle.write("[stars.sfh]\n")
            handle.write(
                f"logtau = {self.sfh_params['logtau'] - 0.05} {self.sfh_params['logtau']} {self.sfh_params['logtau'] + 0.05}\n"
            )
            handle.write(f"alpha_powerlaw = {self.sfh_params['alpha_powerlaw']}\n")
            handle.write(
                f"ism_metallicity_today = {self.sfh_params['ism_metallicity_today']}\n"
            )

        configuration = {
            "runtime": {
                "sampler": "maxlike",
            },
            "maxlike": {
                "method": "Nelder-Mead",
                "tolerance": 1e-3,
                "maxiter": 50,
            },
            "output": {
                "filename": output_root,
                "format": "text",
            },
            "pipeline": {
                "modules": "SpectraRedshiftFit",
                "values": values_path,
                "likelihoods": "SpectraRedshiftFit",
                "quiet": "T",
                "timing": "F",
                "debug": "F",
                "extra_output": "redshift/redshift",
            },
            "SpectraRedshiftFit": {
                "file": SpectraRedshiftFitModule.get_path(),
                "redshift": 0.0,
                "inputSpectrum": self.spectrum_path,
                "SSPModel": "PopStar",
                "SSPModelArgs": "cha",
                "SSPDir": "None",
                "wlRange": self.wavelength_range,
                "SFHModel": "DelayedTauSFH",
            },
        }

        pipeline = MainPipeline([configuration], n_cores_list=[1])
        self.assertEqual(pipeline.execute_all(plot_result=False), 0)

        results_path = output_root + ".txt"
        reader = Reader.from_results_file(results_path)
        reader.load_results()
        solution = reader.get_maxlike_solution()
        best_fit_block = reader.solution_to_datablock(solution)
        module = reader.last_module

        self.assertEqual(module.execute(best_fit_block), 0)
        self.assertAlmostEqual(
            best_fit_block["redshift", "redshift"], self.true_redshift, delta=0.01
        )


if __name__ == "__main__":
    from besta.logging import setup_logging
    setup_logging()
    unittest.main()