"""
This module contains the tools to build a pipeline
"""

import importlib.util
import sys
import os
import argparse
import subprocess
import multiprocessing
import configparser
import numpy as np

from matplotlib import pyplot as plt

from hbsps.output import make_ini_file, Reader
from hbsps import prepare_spectra
from hbsps import kinematics
from hbsps.dust_extinction import deredden_spectra, redden_ssp

class MainPipeline(object):
    def __init__(self, pipeline_configuration_list, n_cores_list=None):
        self.pipelines_config = pipeline_configuration_list
        if n_cores_list is None:
            self.n_cores_list = [1] * len(pipeline_configuration_list)
        else:
            self.n_cores_list = n_cores_list

    def run_command(self, command):
        print(f"Running command {command}")
        subprocess.call(command, shell=True)

    def get_cosmosis_result(self, ini_file):
        return Reader(ini_file)

    def execute_all(self):
        print("Executing all pipelines")
        for pipeline_config, n_cores in zip(
            self.pipelines_config, self.n_cores_list):
            output_dir = os.path.dirname(pipeline_config['output']['filename'])
            filename = (pipeline_config['pipeline']['modules'].replace(" ", "_")
                        + "_auto.ini")
            filename = os.path.join(output_dir, filename)
            make_ini_file(filename, pipeline_config)
            if n_cores > 1:
                command = f"mpiexec -n {n_cores} cosmosis --mpi {filename}"
            else:
                command = f"cosmosis {filename}"
            # self.run_command(command)
            reader = self.get_cosmosis_result(filename)
            module = reader.ini['pipeline']['modules']
            ssp_wl, ssp_sed, ssp_weights = reader.load_ssp_model()
            config = {}
            prepare_spectra.prepare_observed_spectra(
                reader.ini_data, config, module=module)
            prepare_spectra.prepare_ssp_data(reader.ini_data, config,
                                             module=module)
            prepare_spectra.prepare_extinction_law(reader.ini_data, config,
                                                   module=module)
            print(config)
            last_sampler = pipeline_config['runtime']['sampler'].split(
                " ")[-1].strip(" ")
            if last_sampler == "maxlike":
                reader.load_chain()
                params = {}
                for key in reader.chain.keys():
                    if "parameters" in key:
                        params[key.replace("parameters--", "")] = reader.chain[key][0]
                ssp_weights = ssp_weights[-1]
            else:
                # Post-process the results
                command = f"cosmosis-postprocess {pipeline_config['output']['filename']}.txt -o {output_dir}"
                self.run_command(command)
                params = reader.load_processed_file(kind="medians")
            print(params)
            sed, mask = kinematics.convolve_ssp(config,
                                                params['sigma'],
                                                params['los_vel'])
            config['ssp_sed'] = sed
            config['ssp_wl'] = config['wavelength']
            config['mask'] = mask
            # Dust extinction
            redden_ssp(config, params['av'], r=3.1)
            flux_model = np.sum(config['ssp_sed'] * ssp_weights[:, np.newaxis], axis=0)
            plt.figure()
            plt.plot(config['wavelength'], flux_model)
            plt.plot(config['wavelength'], config['flux'])
            plt.show()