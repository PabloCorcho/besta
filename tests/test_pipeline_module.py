import unittest

import os
import numpy as np

from requests.exceptions import ConnectTimeout
from cosmosis import DataBlock

from besta.pipeline_modules import FullSpectralFitModule, GalaxySpectraModule, GalaxyPhotometryModule

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

        np.savetxt("./test_spectra_exp_sfh.dat", np.array([ssp.wavelength.to_value("nm"),
                                                           sed, sed * 0.1]).T)
        # Fake photometry
        np.savetxt("./test_photo_exp_sfh.dat", np.array([
            ["SLOAN_SDSS.r"], np.ones(1), np.ones(1)]).T, fmt="%s")

    @classmethod
    def tearDownClass(cls):
        print("Removing test spectra")
        os.remove("./test_spectra_exp_sfh.dat")
        os.remove("./test_photo_exp_sfh.dat")

    def test_full_spectral_fit(self):
        print("#" * 30 + "\nTesting FullSpectralFit module\n" + "#" * 30)
        module = FullSpectralFitModule
        config = {"FullSpectralFit": {
                "file": module.get_path(),
                "redshift": 0.0,
                "inputSpectrum": "./test_spectra_exp_sfh.dat",
                "SSPModel": "PopStar",
                "SSPModelArgs": "cha",
                "SSPDir": "None",
                "wlRange": [3700.0, 6000.0],
                "wlUnits": "nm",
                "SFHModel": "ExponentialSFH",
                "velscale": 200.0,
                "ExtinctionLaw": "ccm89",
                "save_chi2": True 
            }}

        block = DataBlock()
        block['dust.attenuation', 'a_v'] = 0
        block['kinematics', 'los_vel'] = 0
        block['kinematics', 'los_sigma'] = 100.
        block['kinematics', 'los_h3'] = 0
        block['kinematics', 'los_h4'] = 0
        block['stars.sfh', 'logtau'] = 1
        block['stars.sfh', 'alpha_powerlaw'] = 1
        block['stars.sfh', 'ism_metallicity_today'] = 0.02

        module = module(config)
        self.assertFalse(module.execute(block))
        self.assertTrue(block["extra", module.like_name + "_chi2"])
        print("Module successfully executed")

    def test_galaxy_spectra(self):
        print("#" * 30 + "\nTesting GalaxySpectra module\n" + "#" * 30)
        module = GalaxySpectraModule
        config = {"GalaxySpectra": {
                "file": module.get_path(),
                "redshift": 0.0,
                "inputSpectrum": "./test_spectra_exp_sfh.dat",
                "SSPModel": "PopStar",
                "SSPModelArgs": "cha",
                "SSPDir": "None",
                "wlRange": [3700.0, 6000.0],
                "wlUnits": "nm",
                "SFHModel": "ExponentialSFH",
                "velscale": 200.0,
                "DustAttenuation": True,
                "ExtinctionLaw": "ccm89",
                "save_ssfr_over_tau": [0.1, 1.0],
            }}

        block = DataBlock()
        block['dust.attenuation', 'a_v'] = 0
        block['kinematics', 'los_vel'] = 0
        block['kinematics', 'los_sigma'] = 100.
        block['kinematics', 'los_h3'] = 0
        block['kinematics', 'los_h4'] = 0
        block['stars.sfh', 'logtau'] = 1
        block['stars.sfh', 'alpha_powerlaw'] = 1
        block['stars.sfh', 'ism_metallicity_today'] = 0.02

        module = module(config)
        self.assertFalse(module.execute(block))
        flux_model, normalization = module.make_observable(block, parse=True)
        self.assertTrue(np.isfinite(flux_model).all())
        self.assertTrue(np.isfinite(normalization).all())
        self.assertTrue(np.isfinite(block["extra", "ssfr_over_tau_0.1000"]))
        self.assertTrue(np.isfinite(block["extra", "ssfr_over_tau_1.0000"]))
        print("Module successfully executed")

    def test_galaxy_photometry(self):
        print("#" * 30 + "\nTesting GalaxyPhotometry module\n" + "#" * 30)
        module = GalaxyPhotometryModule
        config = {"GalaxyPhotometry": {
                "file": module.get_path(),
                "redshift": 0.0,
                "inputPhotometry": "./test_photo_exp_sfh.dat",
                "SSPModel": "PopStar",
                "SSPModelArgs": "cha",
                "SSPDir": "None",
                "SFHModel": "ExponentialSFH",
                "DustAttenuation": True,
                "ExtinctionLaw": "ccm89",
            }}

        block = DataBlock()
        block['dust_attenuation', 'a_v'] = 0
        block['kinematics', 'los_vel'] = 0
        block['kinematics', 'los_sigma'] = 100.
        block['kinematics', 'los_h3'] = 0
        block['kinematics', 'los_h4'] = 0
        block['stars.sfh', 'logtau'] = 1
        block['stars.sfh', 'alpha_powerlaw'] = 1
        block['stars.sfh', 'ism_metallicity_today'] = 0.02

        try:
            module = module(config)
        except ConnectTimeout as exc:
            print("SVO filter query probably failed: skipping")
            return
        self.assertFalse(module.execute(block))
        flux_model = module.make_observable(block, parse=True)
        self.assertTrue(np.isfinite(flux_model).all())
        print("Module successfully executed")

if __name__ == "__main__":
    unittest.main()
