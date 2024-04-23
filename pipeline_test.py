from hbsps.pipeline import MainPipeline

configuration = {
    
    "runtime": {
        "sampler": "maxlike"
    },

    "maxlike": {
        "method": "Nelder-Mead",
        "tolerance": 1e-3,
        "maxiter": 3000,
    },

    # "runtime": {
    #     "sampler": "emcee"
    # },

    # "emcee": {
    #     "walkers": 16,
    #     "samples": 500,
    #     "nsteps": 100,
    # },

    "output": {
        "filename": "/home/pcorchoc/Develop/HBSPS/output/KinDust/tng_sfh",
        "format": "text"
    },

    "pipeline": {
        "modules": "KinDust",
        "values": "/home/pcorchoc/Develop/HBSPS/values_KinDust.ini",
        "likelihoods": "KinDust",
        "quiet": "T",
        "timing": "F",
        "debug": "F"
    },

    "KinDust": {
        "file": "/home/pcorchoc/Develop/HBSPS/KinDust.py",
        "values": "/home/pcorchoc/Develop/HBSPS/values_KinDust.ini",
        "redshift": 0.0,
        "values": "/home/pcorchoc/Develop/HBSPS/values_KinDust.ini",
        "inputSpectrum": "/home/pcorchoc/Develop/HBSPS/test/tng_sub_id_69535.dat",
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
    }
}

main_pipe = MainPipeline([configuration], n_cores_list=[1])
main_pipe.execute_all()
