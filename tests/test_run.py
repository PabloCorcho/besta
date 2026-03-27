import unittest
import os
import numpy as np
from time import time

from besta.sfh import ExponentialSFH
from besta.pipeline import MainPipeline
from besta.pipeline_modules.full_spectral_fit import FullSpectralFitModule
from besta.io import Reader
from besta.postprocess import summarize_results
from pst.SSP import PopStar

class TestPipelineManagerFit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup stuff

        print("Creating test spectra using an exponential SFH")
        # Use the default SSP from PST
        ssp = PopStar(IMF="cha")
        # Create a SFH model and generate a synthetic spectra
        params = {}
        params['logtau'] = 1.0
        params['alpha_powerlaw'] = 1
        params['ism_metallicity_today'] = 0.02

        sfh = ExponentialSFH()
        sfh.parse_free_params(params)
        sed = sfh.model.compute_SED(ssp, t_obs=sfh.today)

        np.savetxt("./test_spectra_exp_sfh.dat", np.array(
            [ssp.wavelength, np.random.normal(sed, sed * 0.01), sed * 0.01]).T)
    
        # Create values file
        text = """[dust.extinction]
        a_v = 0 0 1
        [stars.sfh]
        alpha_powerlaw = 0 1 10
        ism_metallicity_today = 0.005 0.01 0.08
        logtau = -1 0.5 1.7
        [kinematics]
        los_vel = -500 0 500
        los_sigma = 50 100 500
        los_h3 = 0
        los_h4 = 0
        """
        with open("values.ini", "w") as file:
            file.write(text)

    @classmethod
    def tearDownClass(cls):
        print("Removing files")
        if os.path.exists("./test_spectra_exp_sfh.dat"):
            os.remove("./test_spectra_exp_sfh.dat")
        if os.path.exists("./values.ini"):
            os.remove("./values.ini")
        if os.path.exists("./FullSpectralFit_auto.ini"):
            os.remove("./FullSpectralFit_auto.ini")
        if os.path.exists("./full_fit_exponential_sfh.txt"):
            os.remove("./full_fit_exponential_sfh.txt")
        if os.path.exists("./full_fit_exponential_sfh.maxlike.txt"):
            os.remove("./full_fit_exponential_sfh.maxlike.txt")
        if os.path.exists("./full_fit_exponential_sfh_FullSpectralFit_best_fit_spectra.png"):
            os.remove("./full_fit_exponential_sfh_FullSpectralFit_best_fit_spectra.png")

    def test_fit(self):
        configuration = {
        
        "runtime": {
            "sampler": "maxlike emcee"
        },

        "maxlike": {
            "method": "Nelder-Mead",
            "tolerance": 1e-3,
            "maxiter": 3000,
        },

        "emcee": {
            "walkers": 32,
            "samples": 100,
            "nsteps": 100,
        },

        "output": {
            "filename": "./full_fit_exponential_sfh",
            "format": "text"
        },

        "pipeline": {
            "modules": "FullSpectralFit",
    #        "values": "./full_fit_values.ini",
            "values": "./values.ini",
            "likelihoods": "FullSpectralFit",
            "quiet": "F",
            "timing": "T",
            "debug": "T",
            "extra_output": "extra/stellar_mass"
        },

        "FullSpectralFit": {
                "file": FullSpectralFitModule.get_path(),
                "redshift": 0.0,
                "inputSpectrum": "./test_spectra_exp_sfh.dat",
                #"mask": "./a2744_65_mask.txt",
                "SSPModel": "PopStar",
                "SSPModelArgs": "cha",
                "SSPDir": "None",
                "wlRange": [3500.0, 9000.0],
                "SFHModel": "ExponentialSFH",
                "velscale": 50.0,
                "ExtinctionLaw": "ccm89",
                }}

        t0 = time()
        main_pipe = MainPipeline([configuration], n_cores_list=[1])
        main_pipe.execute_all(plot_result=True)
        tend = time()
        print("TOTAL ELAPSED TIME (min): ", (tend - t0) / 60)

    def test_postprocess(self):
        results = Reader.from_results_file("./full_fit_exponential_sfh.txt")
        # Load the results
        results.load_results()
        maxlike_sol = results.get_maxlike_solution()
        print("Maximum likelihood solution: ", maxlike_sol)

        # self.assertTrue(np.isclose(maxlike_sol["av"], 0, atol=0.15),
        #                "Wrong extinction")
        # self.assertTrue(np.isclose(maxlike_sol["logtau"], 1, atol=0.2),
        #                "Wrong exponential SFH tau")
        # self.assertTrue(np.isclose(maxlike_sol["ism_metallicity_today"], 0.02,
        #                           atol=0.01),
        #                "Wrong present-day metallicity")
        # self.assertTrue(np.isclose(maxlike_sol["stellar_mass"], 1.0,
        #                           atol=0.1),
        #                "Wrong exponential SFH tau")        

        results = summarize_results(results.results_table)

if __name__ == "__main__":
    unittest.main()