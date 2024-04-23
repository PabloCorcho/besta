import cosmosis
from KinDust import kindust_module
from cosmosis.runtime.mpi_pool import MPIPool
from cosmosis.runtime.utils import ParseExtraParameters, stdout_redirected, import_by_path, under_over_line, underline, overline
from cosmosis import LikelihoodPipeline
from cosmosis.main import sampler_main_loop, run_cosmosis

import os

# Configuration options for all our modules
params = {
    "KinDust": {
        "redshift":0.0,
        "values": "/home/pcorchoc/Develop/HBSPS/values_KinDust.ini",
        "inputSpectrum": "/home/pcorchoc/Develop/HBSPS/test/tng_sub_id_69535.dat",
        "SSPModel": "PyPopStar",
        "SSPModelArgs": "KRO",
        "SSPDir": None,
        "wlRange": "3700.0 8900.0",
        "wlNormRange": "5000.0 5500.0",
        "velscale": 70.0,
        "oversampling": 2,
        "polOrder": 10,
        "ExtinctionLaw": "ccm89",
    }
}
# The values can either contain individual values for parameters that are fixed,
# or ranges as strings: "min_value   start_value   max_value".
values = {
    "parameters":
    {"los_vel": "-300 0.0 300",
     "sigma": "10 155.0 300",
     "av": "0.0 1.5 3.0",
    }
}

sampler_params = {
    "maxlike": {
        # max_posterior only matters if you have non-uniform priors
        # on parameters. It won't make any difference here.
        "max_posterior": True,
        "tolerance": 0.001
    },
    "emcee": {
        "walkers": 16,
        "samples": 1000,
        "steps": 100,
    }
}

def prepare_pipeline():
    modules = [kindust_module]
    pipe = cosmosis.LikelihoodPipeline(params, values=values, modules=modules)
    return pipe

pipe = None

# with MPIPool() as pool:
#     if pool.is_master():
#         modules = [kindust_module]
#         pipe = cosmosis.LikelihoodPipeline(params, values=values, modules=modules)

#         sampler = cosmosis.samplers.MaxlikeSampler(sampler_params, pipe)
#         sampler.config()
#         sampler.execute()

#         print("Best-fit quadratic (truth=200):", sampler.output['parameters--los_vel'])
#         print("Best-fit linear (truth=100):", sampler.output['parameters--sigma'])
#         print("Best-fit constant (truth=1):", sampler.output['parameters--av'])

#     sampler = cosmosis.samplers.EmceeSampler(sampler_params, pipe, pool=pool)
#     sampler.config()
#     sampler_main_loop(sampler, None, pool, pool.is_master())
#     run_count_total = pool.comm.allreduce(pipe.run_count)
#     run_count_ok_total = pool.comm.allreduce(pipe.run_count_ok)

#     if pool.is_master():
#         print("Evaluations: ", run_count_total)
#         print("Successes: ", run_count_total)

with MPIPool() as pool:
    no_subprocesses = os.environ.get("COSMOSIS_NO_SUBPROCESS", "") not in ["", "0"]

    if pipe is None:
        cleanup_pipeline = True

        if pool.is_master():
            
            pipe = prepare_pipeline()
        else:
            with stdout_redirected():
                pipe = prepare_pipeline()

        if pipe.do_fast_slow:
            pipe.setup_fast_subspaces()

    if pool.is_master():
        pipe = prepare_pipeline()

    sampler = cosmosis.samplers.EmceeSampler(sampler_params, pipe, pool=pool)
    sampler_main_loop(sampler, None, pool, pool.is_master())
    run_count_total = pool.comm.allreduce(pipe.run_count)
    run_count_ok_total = pool.comm.allreduce(pipe.run_count_ok)

    if pool.is_master():
         print("Evaluations: ", run_count_total)
         print("Successes: ", run_count_total)

# print("Best-fit quadratic (truth=200):", sampler.output['parameters--los_vel'])
# print("Best-fit linear (truth=100):", sampler.output['parameters--sigma'])
# print("Best-fit constant (truth=1):", sampler.output['parameters--av'])

