"""This module performs the general configuration of parameters used in BESTA"""
import os
import yaml

from astropy import cosmology as astropy_cosmology

# Get configuration file
config_file_path = os.getenv("besta_config", default=os.path.join(
                        os.path.dirname(__file__), "besta-config.yml"))

if not os.path.isfile(config_file_path):
    raise FileNotFoundError(
        f"Input configuration file {config_file_path} could not be found")
else:
    with open(config_file_path, 'r') as file:
        config_file = yaml.safe_load(file)

# Setup the configuration
# Adopted cosmology
if "cosmology" in config_file:
    if "name" in config_file["cosmology"]:
        cosmology = getattr(astropy_cosmology, config_file["cosmology"]["name"])
    else:
        raise KeyError("Configuration of cosmology requires the adopted astropy"
                       + "cosmology name")
    # Initialising some cosmologies require some input parameters
    if "args" in config_file["cosmology"]:
        cosmology(**config_file["cosmology"]["args"])
else:
    print("Using default cosmology")
    cosmology = astropy_cosmology.FlatLambdaCDM(H0=70., Om0=0.28)

# Kinematics
if "kinematics" in config_file:
    kinematics = config_file["kinematics"]
    if not "lsf_sigma_truncation" in kinematics:
        kinematics["lsf_sigma_truncation"] = 5
    if not "extra_velocity_buffer" in kinematics:
        kinematics["extra_velocity_buffer"] = 800.0
