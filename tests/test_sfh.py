import unittest

import numpy as np
from astropy import units as u

from cosmosis import DataBlock

from besta import sfh

import pst


class TestSFH(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.ssp = pst.SSP.PopStar(IMF='cha')
        self.metallicity_params = {
            "alpha_powerlaw" : 1.0,
            "ism_metallicity_today": 0.02}

    def test_fixedtimesfh(self):
        lookback_time_bins = np.array([1e-3, 1e-2, 1e-1, 1, 10])
        model_fitter = sfh.FixedTimeSFH(lookback_time_bins)
        
        model_parameters = {}
        for k in model_fitter.sfh_bin_keys:
            model_parameters[k] = np.log10(0.999 / lookback_time_bins.size)
        model_parameters = {**model_parameters, **self.metallicity_params}
        value, penalty = model_fitter.parse_free_params(model_parameters)
        self.assertEqual(value, 1)

    def test_fixedtime_ssfr_sfh(self):
        lookback_time_bins = np.array([1e-3, 1e-2, 1e-1, 1, 10])
        model_fitter = sfh.FixedTime_sSFR_SFH(lookback_time_bins)
        
        model_parameters = {}
        for k in model_fitter.sfh_bin_keys:
            model_parameters[k] = -11.0

        model_parameters = {**model_parameters, **self.metallicity_params}
        value, penalty = model_fitter.parse_free_params(model_parameters)
        self.assertEqual(value, 1)

    def test_fixedmass_sfh(self):
        mass_frac_bins = np.array([1e-3, 1e-2, 1e-1, 0.5, 0.9])
        model_fitter = sfh.FixedMassFracSFH(mass_fraction=mass_frac_bins)

        model_parameters = {}
        for k in model_fitter.sfh_bin_keys:
            model_parameters[k] = 1.0

        model_parameters = {**model_parameters, **self.metallicity_params}
        value, penalty = model_fitter.parse_free_params(model_parameters)
        self.assertEqual(value, 1)

    def test_exponential_sfh(self):
        model_fitter = sfh.ExponentialSFH()
        model_parameters = {"logtau": 0.5}
        model_parameters = {**model_parameters, **self.metallicity_params}
        value, penalty = model_fitter.parse_free_params(model_parameters)
        self.assertEqual(value, 1)
    
    def test_delayed_tau_sfh(self):
        model_fitter = sfh.DelayedTauSFH()
        model_parameters = {"logtau": 0.5}
        model_parameters = {**model_parameters, **self.metallicity_params}
        value, penalty = model_fitter.parse_free_params(model_parameters)
        self.assertEqual(value, 1)
    
    def test_delayed_tau_quenched_sfh(self):
        model_fitter = sfh.DelayedTauQuenchedSFH()
        model_parameters = {"logtau": 0.5, "quenching_time" : 1.0}
        model_parameters = {**model_parameters, **self.metallicity_params}
        value, penalty = model_fitter.parse_free_params(model_parameters)
        self.assertEqual(value, 1)

    def test_lognormal_sfh(self):
        model_fitter = sfh.LogNormalSFH()
        model_parameters = {"scale": 0.5, "t0": 10.0}
        model_parameters = {**model_parameters, **self.metallicity_params}
        value, penalty = model_fitter.parse_free_params(model_parameters)
        self.assertEqual(value, 1)
    
    def test_lognormal_quenched_sfh(self):
        model_fitter = sfh.LogNormalQuenchedSFH()
        model_parameters = {"scale": 0.5, "t0": 10.0, "quenching_time" : 10.0}
        model_parameters = {**model_parameters, **self.metallicity_params}
        value, penalty = model_fitter.parse_free_params(model_parameters)
        self.assertEqual(value, 1)


if __name__ == "__main__":
    unittest.main()