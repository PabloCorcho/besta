import unittest

from cosmosis import DataBlock

from hbsps.pipeline_modules.kin_dust import KinDustModule
from hbsps.pipeline_modules.sfh_spectra import SFHSpectraModule

class TestPipelineModule(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        # Setup stuff
        pass

    def test_kin_dust(self):
        
        kin_configuration = {
            "KinDust": {
                "save_ssp": "/path/to/ssp_output.txt",
                "file": "/home/pcorchoc/Develop/HBSPS/KinDust.py",
                "redshift": 0.0,
                "inputSpectrum": "/home/pcorchoc/Develop/tutorial-pst-hbsps/generate_mock_data/exponential/input_spectra.dat",
                "SSPModel": "PyPopStar",
                "SSPModelArgs": "KRO",
                "SSPDir": "None",
                # "SSP-NMF": True, 
                # "SSP-NMF-N": 10,
                "SSPSave": "F",
                "wlRange": [3700.0, 8000.0],
                "wlNormRange": [5000.0, 5500.0],
                "velscale": 70.0,
                "oversampling": 2,
                "polOrder": 10,
                "ExtinctionLaw": "ccm89",
            }
        }

        block = DataBlock()
        block['parameters', 'av'] = 0
        block['parameters', 'los_vel'] = 0
        block['parameters', 'los_sigma'] = 100.
        block['parameters', 'los_h3'] = 0
        block['parameters', 'los_h4'] = 0
        
        print("Path to module ", KinDustModule.get_path())
        kindust_module = KinDustModule(kin_configuration)
        kindust_module.execute(block)

    def test_sfh_spectra(self):
        config = {"SFH_spectra": {
                "file": "/home/pcorchoc/Develop/HBSPS/SFH_stellar_mass.py",
                "redshift": 0.0,
                "inputSpectrum": "/home/pcorchoc/Develop/tutorial-pst-hbsps/generate_mock_data/exponential/input_spectra.dat",
                "SSPModel": "PyPopStar",
                "SSPModelArgs": "KRO",
                "SSPDir": "None",
                # "SSP-NMF": "F",
                # "SSP-NMF-N": None,
                #"SSPSave": "F",
                "wlRange": [3700.0, 6000.0],
                "wlNormRange": [5000.0, 5500.0],
                # "SFHModel": "LogNormalQuenchedSFH",
                # "SFHModel": "LogNormalSFH",
                "SFHModel": "FixedMassFracSFH",
                "SFHArgs1": [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
                "velscale": 70.0,
                "oversampling": 2,
                "polOrder": 10,
                "los_vel": 0.0,
                "los_sigma": 100.0,
                "av": 0.0,
                "ExtinctionLaw": "ccm89",
            }}

        block = DataBlock()
        block['parameters', 'av'] = 0
        block['parameters', 'los_vel'] = 0
        block['parameters', 'los_sigma'] = 100.
        block['parameters', 'los_h3'] = 0
        block['parameters', 'los_h4'] = 0

        sfh_module = SFHSpectraModule(config)

        print("Path to module ", sfh_module.get_path())

if __name__ == "__main__":
    unittest.main()