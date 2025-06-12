import unittest

import os
import numpy as np

from cosmosis import DataBlock

from besta.pipeline_modules.kin_dust import KinDustModule
from besta.pipeline_modules.full_spectral_fit import FullSpectralFitModule
from besta.sfh import ExponentialSFH
from pst.SSP import PopStar

class TestPipelineModule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Setup stuff

        print("Creating test spectra using an exponential SFH")
        # Use the default SSP from PST
        ssp = PopStar(IMF="cha")
        # Create a SFH model and generate a synthetic spectra
        params = {}
        params['logtau'] = 0.5
        params['alpha_powerlaw'] = 1
        params['ism_metallicity_today'] = 0.02

        sfh = ExponentialSFH()
        sfh.parse_free_params(params)
        sed = sfh.model.compute_SED(ssp, t_obs=sfh.today)

        np.savetxt("./test_spectra_exp_sfh.dat", np.array([ssp.wavelength,
                                                           sed, sed * 0.1]).T)

    @classmethod
    def tearDownClass(cls):
        print("Removing test spectra")
        os.remove("./test_spectra_exp_sfh.dat") 

    def test_kin_dust(self):
        print("#" * 23 + "\nTesting KinDust module\n" + "#" * 23)
        kin_configuration = {
            "KinDust": {
                "file": KinDustModule.get_path(),
                "redshift": 0.0,
                "inputSpectrum": "./test_spectra_exp_sfh.dat",
                "SSPModel": "PopStar",
                "SSPModelArgs": "cha",
                "SSPDir": "None",
                "wlRange": [3700.0, 8000.0],
                "velscale": 200.0,
                "ExtinctionLaw": "ccm89",
            }
        }

        block = DataBlock()
        block['parameters', 'av'] = 0
        block['parameters', 'los_vel'] = 0
        block['parameters', 'los_sigma'] = 100.
        block['parameters', 'los_h3'] = 0
        block['parameters', 'los_h4'] = 0

        kindust_module = KinDustModule(kin_configuration)
        kindust_module.execute(block)
        print("Module successfully executed")

    def test_full_spectral_fit(self):
        print("#" * 30 + "\nTesting FullSpectralFit module\n" + "#" * 30)

        config = {"FullSpectralFit": {
                "file": FullSpectralFitModule.get_path(),
                "redshift": 0.0,
                "inputSpectrum": "./test_spectra_exp_sfh.dat",
                "SSPModel": "PopStar",
                "SSPModelArgs": "cha",
                "SSPDir": "None",
                "wlRange": [3700.0, 6000.0],
                "SFHModel": "ExponentialSFH",
                "velscale": 200.0,
                "ExtinctionLaw": "ccm89",
            }}

        block = DataBlock()
        block['parameters', 'av'] = 0
        block['parameters', 'los_vel'] = 0
        block['parameters', 'los_sigma'] = 100.
        block['parameters', 'los_h3'] = 0
        block['parameters', 'los_h4'] = 0
        block['parameters', 'logtau'] = 1
        block['parameters', 'alpha_powerlaw'] = 1
        block['parameters', 'ism_metallicity_today'] = 0.02

        module = FullSpectralFitModule(config)
        self.assertFalse(module.execute(block))
        print("Module successfully executed")

if __name__ == "__main__":
    unittest.main()