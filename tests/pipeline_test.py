from hbsps.pipeline import MainPipeline
from time import time

kin_configuration = {
    
    "runtime": {
        "sampler": "maxlike"
    },

    "maxlike": {
        "method": "Nelder-Mead",
        "tolerance": 1e-3,
        "maxiter": 3000,
    },

    "emcee": {
        "walkers": 16,
        "samples": 1000,
        "nsteps": 500,
    },

    "output": {
        "filename": "/home/pcorchoc/Develop/HBSPS/output/lognormal/kinematics",
        "format": "text"
    },

    "pipeline": {
        "modules": "KinDust",
        "values": "/home/pcorchoc/Develop/HBSPS/output/lognormal/values_KinDust.ini",
        "likelihoods": "KinDust",
        "quiet": "T",
        "timing": "T",
        "debug": "F",
        #"extra_output": "parameters/ssp1 parameters/ssp2"
    },

    "KinDust": {
        "file": "/home/pcorchoc/Develop/HBSPS/KinDust.py",
        "redshift": 0.0,
        "inputSpectrum": "test/lognormal/input_spectra.dat",
        "SSPModel": "PyPopStar",
        "SSPModelArgs": "KRO",
        "SSPDir": None,
        "SSP-NMF": "F", 
        "SSP-NMF-N": 10,
        "SSPSave": "T",
        "wlRange": "3700.0 8000.0",
        "wlNormRange": "5000.0 5500.0",
        "velscale": 70.0,
        "oversampling": 2,
        "polOrder": 10,
        "ExtinctionLaw": "ccm89",
    },

    "Values": {
        "los_vel": "-500 0 500",
        "los_sigma": "50 100 400",
        "av": "0 0.1 3"
    }
}

sfh_configuration = {
    
    "runtime": {
        "sampler": "emcee"
    },

    "maxlike": {
        "method": "Nelder-Mead",
        "tolerance": 1e-5,
        "maxiter": 10000,
    },

    "emcee": {
        "walkers": 16,
        "samples": 1000,
        "nsteps": 100,
    },

    "multinest":{
        "max_iterations": 50000,
        "live_points": 700,
        "feedback": True,
        "update_interval": 2000,
        "log_zero": -1e14,
        "multinest_outfile_root": "/home/pcorchoc/Develop/HBSPS/output/lognormal/sampling/"
    },

    "output": {
        "filename": "/home/pcorchoc/Develop/HBSPS/output/lognormal/SFH_results",
        "format": "text"
    },

    "pipeline": {
        "modules": "SFH_stellar_mass",
        "values": "/home/pcorchoc/Develop/HBSPS/output/lognormal/values_SFH.ini",
        "likelihoods": "SFH_stellar_mass",
        "quiet": "T",
        "timing": "T",
        "debug": "T",
        "extra_output": "parameters/normalization"
    },

    "SFH_stellar_mass": {
        "file": "/home/pcorchoc/Develop/HBSPS/SFH_stellar_mass.py",
        "redshift": 0.0,
        "inputSpectrum": "/home/pcorchoc/Develop/HBSPS/test/lognormal/input_spectra.dat",
        "SSPModel": "PyPopStar",
        "SSPModelArgs": "KRO",
        "SSPDir": None,
        # "SSP-NMF": "F",
        # "SSP-NMF-N": None,
        #"SSPSave": "F",
        "wlRange": "3700.0 6000.0",
        "wlNormRange": "5000.0 5500.0",
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
    },

    "Values": {
        # "alpha": "0.0 1.0 3.0",
        # "z_today": "0.005 0.01 0.08",
        # "scale": "0.1 3.0 50",
        # "lnt0": "-2.3025850929940455 1.9070938758868938 5.2004821128936785",
        #"lnt_quench": "-1.2039728043259361 2.6002410564468392 3.2933882370067846",
        #"lntau_quench": "-2.3025850929940455 -0.6931471805599453 2.6002410564468392"
    }
}

t0 = time()
main_pipe = MainPipeline([
                          kin_configuration,
                          sfh_configuration
                          ],
                         n_cores_list=[
                             1, 1
                             #           4
                                       ])
main_pipe.execute_all()
tend = time()
print("TOTAL ELAPSED TIME (min): ", (tend - t0) / 60)