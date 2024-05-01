from hbsps.pipeline import MainPipeline

kin_configuration = {
    
    "runtime": {
        "sampler": "maxlike emcee"
    },

    "maxlike": {
        "method": "Nelder-Mead",
        "tolerance": 1e-3,
        "maxiter": 3000,
    },

    "emcee": {
        "walkers": 16,
        "samples": 500,
        "nsteps": 500,
    },

    "output": {
        "filename": "/home/pcorchoc/Develop/HBSPS/output/KinDust/tng_kin",
        "format": "text"
    },

    "pipeline": {
        "modules": "KinDust",
        "values": "/home/pcorchoc/Develop/HBSPS/output/KinDust/values_KinDust.ini",
        "likelihoods": "KinDust",
        "quiet": "T",
        "timing": "F",
        "debug": "F",
        #"extra_output": "parameters/ssp1 parameters/ssp2"
    },

    "KinDust": {
        "file": "/home/pcorchoc/Develop/HBSPS/KinDust.py",
        "redshift": 0.0,
        "inputSpectrum": "/home/pcorchoc/Develop/HBSPS/test/tng/tng_sub_id_588577.dat",
        "SSPModel": "PyPopStar",
        "SSPModelArgs": "KRO",
        "SSPDir": None,
        "SSP-NMF": "T",
        "SSP-NMF-N": 10,
        "SSPSave": "T",
        "wlRange": "3700.0 8900.0",
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
        "walkers": 256,
        "samples": 5000,
        "nsteps": 1000,
    },

    "multinest":{
        "max_iterations": 50000,
        "live_points": 700,
        "feedback": True,
        "update_interval": 2000,
        "log_zero": -1e14,
        "multinest_outfile_root": "/home/pcorchoc/Develop/HBSPS/output/KinDust/sampling/"
    },

    "output": {
        "filename": "/home/pcorchoc/Develop/HBSPS/output/KinDust/tng_sfh",
        "format": "text"
    },

    "pipeline": {
        "modules": "SFH",
        "values": "/home/pcorchoc/Develop/HBSPS/output/KinDust/values_SFH.ini",
        "likelihoods": "SFH",
        "quiet": "T",
        "timing": "F",
        "debug": "F",
    },

    "SFH": {
        "file": "/home/pcorchoc/Develop/HBSPS/SFH.py",
        "redshift": 0.0,
        "inputSpectrum": "/home/pcorchoc/Develop/HBSPS/test/tng/tng_sub_id_588577.dat",
        "SSPModel": "PyPopStar",
        "SSPModelArgs": "KRO",
        "SSPDir": None,
        # "SSP-NMF": "F",
        # "SSP-NMF-N": None,
        "SSPSave": "T",
        "wlRange": "3700.0 8900.0",
        "wlNormRange": "5000.0 5500.0",
        "ageRange": [5.0, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0],
        "metRange": [-3., -2.5, -2.0, -1.5, -1.0],
        "velscale": 70.0,
        "oversampling": 2,
        "polOrder": 10,
        "los_vel": 100.0,
        "los_sigma": 100.0,
        "av": 1.0,
        "ExtinctionLaw": "ccm89",
    },

    "Values": {
        "ssp1": "-6 -0.3 0",
        "ssp2": "-6 -0.3 0",
        "ssp3": "-6 -0.3 0",
        "ssp4": "-6 -0.3 0",
        "ssp5": "-6 -0.3 0",
        "ssp6": "-6 -0.3 0",
        "ssp7": "-6 -0.3 0",
        "ssp8": "-6 -0.3 0",
        "ssp9": "-6 -0.3 0",
    }
}

main_pipe = MainPipeline([kin_configuration,
                          sfh_configuration
                          ],
                         n_cores_list=[1,
                                        4
                                       ])
main_pipe.execute_all()
