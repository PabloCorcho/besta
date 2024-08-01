import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits, ascii
from astropy import table

import hbsps.specBasics as specBasics
import hbsps.prepare_fit as prepare_fit

from pst import SSP
from pst.utils import flux_conserving_interpolation

import extinction

import cosmosis


def save_ssp(filename, config, **extra_args):
    # Create a header
    print(f"Saving SSP model to FITS file: {filename}")
    p_header = fits.Header()
    p_header["hierarch velscale"] = (config["velscale"], "pixel resolution in km/s")
    p_header["hierarch oversampling"] = (config["oversampling"], "oversampling factor")
    p_header["hierarch extra_pixels"] = (
        config["extra_pixels"],
        "extra pixels at edges",
    )

    for k, v in extra_args.items():
        if type(v) is list:
            p_header["hierarch  " + k] = ", ".join(v)
        else:
            p_header["hierarch  " + k] = v
    metal_edges, age_edges = config['ssp_model'].get_ssp_logedges()
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(header=p_header),
            fits.ImageHDU(name="WAVE", data=config["ssp_wl"]),
            fits.ImageHDU(name="SED", data=config["ssp_sed"]),
            fits.ImageHDU(name="METALS_EDGES", data=metal_edges.value),
            fits.ImageHDU(name="AGES_EDGES", data=age_edges.value),
        ]
    )
    hdul.writeto(filename, overwrite=True)
    hdul.close()


def make_ini_file(filename, config):
    print(f"Writing .ini file: {filename}")
    with open(filename, "w") as f:
        f.write(f"; File generated automatically\n")
        for section in config.keys():
            # Ignore the Values section
            if section == 'Values':
                continue
            f.write(f"[{section}]\n")
            for key, value in config[section].items():
                content = f"{key} = "
                if type(value) is str:
                    content += " " + value
                elif type(value) is list:
                    content += " ".join([str(v) for v in value])
                # elif (type(value) is float) or (type(value) is int):
                #     content += str(value)
                elif value is None:
                    content += "None"
                else:
                    content += str(value)
                f.write(f"{content}\n")
        f.write(f"; \(ﾟ▽ﾟ)/")


def make_values_file(config, ssp_auto_fill=True):
    """Make a values.ini file from the configuration."""
    values_filename = config["pipeline"]["values"]
    values_section = f"Values"

    if ssp_auto_fill:
        module = config["pipeline"]["modules"].split(" ")[-1]
        n_ssp = 1
        if "ageRange" in config[module]:
            n_ssp *= len(config[module]["ageRange"]) - 1
        if "metRange" in config[module]:
            n_ssp *= len(config[module]["metRange"]) - 1
        else:
            ssp_auto_fill = False

    if values_section in config:
        print(f"Creating values file: {values_filename}")
        with open(values_filename, "w") as f:
            f.write("[parameters]\n")
            if ssp_auto_fill:
                for i in range(n_ssp - 1):
                    f.write(f"ssp{i+1} = -6 {np.log10(1/n_ssp)} 0\n")
                return
            for name, lims in config[values_section].items():
                if type(lims) is str:
                    f.write(f"{name} = {lims}\n")
                else:
                    f.write(f"{name} = {lims[0]} {(lims[0] + lims[1]) / 2} {lims[1]}\n")
    return


class Reader(object):
    def __init__(self, ini_file):
        self.ini_file = ini_file
        self.ini, self.ini_data = self.read_ini_file(self.ini_file)
        self.config = {}

        self.last_module = (
            self.ini["pipeline"]["modules"].split(" ")[-1].replace(" ", "")
        )

    def load_processed_file(self, kind="medians"):
        filename = os.path.join(
            os.path.dirname(self.ini["output"]["filename"]), f"{kind}.txt"
        )
        if os.path.isfile(filename):
            parameters = {}
            with open(filename) as f:
                for line in f.readlines():
                    line_content = (
                        line.replace("\n", "").replace("\t", "  ").split("  ")
                    )
                    if line_content[0] == "#":
                        continue
                    if "parameters" in line_content[0]:
                        name = line_content[0].replace("parameters--", "")
                        if kind == "maxlike":
                            parameters[name] = {
                                f"{kind}": float(line_content[1]),
                            }
                        else:
                            parameters[name] = {
                                f"{kind}": float(line_content[1]),
                                "sigma": float(line_content[2]),
                            }
            return parameters
        else:
            return None

    def load_chain(self, include_ssp_weights=False):
        path = self.ini["output"]["filename"]
        if ".txt" not in path:
            path += ".txt"
        self.chain = self.read_chain_file(path)
        if include_ssp_weights:
            self.load_ssp_weights()
        self.table_chain = table.Table(self.chain)

    def load_observation(self):
        last_module = self.ini["pipeline"]["modules"].split(" ")[-1].replace(" ", "")
        prepare_fit.prepare_observed_spectra(
            cosmosis_options=self.ini_data, config=self.config, module=last_module
        )

    def load_ssp_model(self):
        """Load an SSP model from a file"""
        last_module = self.ini["pipeline"]["modules"].split(" ")[-1].replace(" ", "")
        prepare_fit.prepare_ssp_model(
                cosmosis_options=self.ini_data, config=self.config, module=last_module
            )
        prepare_fit.prepare_ssp_model_preprocessing(
            cosmosis_options=self.ini_data, config=self.config, module=last_module)

    def load_sfh_model(self):
        last_module = self.ini["pipeline"]["modules"].split(" ")[-1].replace(" ", "")
        prepare_fit.prepare_sfh_model(
            cosmosis_options=self.ini_data, config=self.config, module=last_module
        )

    def load_ssp_weights(self):
        """Load the SSP weights used during the fit.

        Description
        -----------
        Some modules do not store the SSP weights on the final COSMOSIS file and
        therefore they have to be stored separatedly. This method provides the
        tools for reading them.
        """
        print("Loading SSP weights from file")
        path_to_weights_file = os.path.join(
            os.path.dirname(self.ini["output"]["filename"]), "SSP_weights.dat"
        )
        weights = ascii.read(path_to_weights_file)
        # weights_matrix = np.loadtxt(path_to_weights_file, delimiter=',')
        # Add the new SSP keys
        parameters = [k for k in self.chain.keys() if "parameters" in k]
        chain_params = np.array([self.chain[p] for p in parameters])
        matching_params = np.array([weights[p].value for p in parameters])

        ssp_params = np.array([weights[p].value for p in weights.keys() if "ssp" in p])
        for key in weights.keys():
            if "parameters--ssp" in key:
                self.chain[key] = np.zeros_like(self.chain["post"], dtype=float)

        # Add the values of each SSP
        for ith, params in enumerate(chain_params.T):
            pos = np.argmin(
                np.sum((params[:, np.newaxis] - matching_params) ** 2, axis=0)
            )
            for jth in range(ssp_params.shape[0]):
                self.chain[f"parameters--ssp{jth + 1}"][ith] = ssp_params[jth, pos]

    def load_extinction_model(self):
        last_module = self.ini["pipeline"]["modules"].split(" ")[-1].replace(" ", "")
        prepare_fit.prepare_extinction_law(
            cosmosis_options=self.ini_data, config=self.config, module=last_module
        )

    def get_chain_percentiles(self, pct=[0.5, 0.16, 0.50, 0.84, 0.95]):
        parameters = [par for par in self.chain.keys() if "parameters" in par]
        pct_resutls = {"percentiles": np.array(pct)}
        for par in parameters:
            sort_pos = np.argsort(self.chain[par])
            cum_distrib = np.cumsum(self.chain["weight"][sort_pos])
            cum_distrib /= cum_distrib[-1]
            pct_resutls[par] = np.interp(pct, cum_distrib, self.chain[par][sort_pos])
        return pct_resutls

    def get_maxlike_solution(self, log_prob="post"):
        good_sample = self.chain[log_prob] != 0
        maxlike_pos = np.argmax(self.chain[log_prob][good_sample])
        solution = {}
        for k, v in self.chain.items():
            if "parameters" in k:
                solution[k.replace("parameters--", "")] = v[good_sample][maxlike_pos]
        return solution

    def get_pct_solutions(self, pct=99, log_prob="post"):
        good_sample = self.chain[log_prob] != 0
        maxlike_pos = np.argmax(self.chain[log_prob][good_sample])
        # Normalize the weights
        weights = self.chain[log_prob][good_sample] - self.chain[log_prob][good_sample][maxlike_pos]
        weights = np.exp(weights / 2)
        weights /= np.nansum(weights)
        # From the highest to the lowest weight
        sort = np.argsort(weights)
        cum_weights = np.cumsum(weights[sort][::-1])
        last_sample = np.searchsorted(cum_weights, pct / 100)

        solution = {'weights': weights[sort][-last_sample:]}    
        for k, v in self.chain.items():
            if "parameters" in k:
                solution[k.replace("parameters--", "")] = v[
                    good_sample][sort][-last_sample:]
        return solution

    def compute_solution_from_pct(self, pct_results, pct=0.5):
        print("Computing solution from percentiles")
        column = np.argmin(np.abs(pct_results["percentiles"] - pct))
        los_vel = pct_results["parameters--los_vel"][column]
        redshift = np.exp(los_vel / specBasics.constants.c.to("km/s").value) - 1
        los_sigma = pct_results["parameters--sigma"][column]

        ssp_weights = np.zeros(self.n_ssp)
        for i in range(1, self.n_ssp + 1):
            ssp_weights[i - 1] = 10 ** pct_results[f"parameters--ssp{i}"][column]
        ssp_weights /= ssp_weights.sum()
        syn_spectra = np.sum(
            ssp_weights.reshape(self.ssp.L_lambda.shape[:-1])[:, :, np.newaxis]
            * self.ssp.L_lambda,
            axis=(0, 1),
        )
        sigma_pix = los_sigma / (
            self.ini["HBSPS_SFH"]["velscale"][0]
            / self.ini["HBSPS_SFH"]["oversampling"][0]
        )
        syn_spectra = specBasics.smoothSpectrumFast(syn_spectra, sigma_pix)
        syn_spectra = flux_conserving_interpolation(
            self.observation["wavelength_fit"],
            self.ssp.wavelength * (1 + redshift),
            syn_spectra,
        )
        if "ExtinctionLaw" in self.ini["HBSPS_SFH"]:
            av = 10 ** pct_results["parameters--av"][column]
            extinction_law = getattr(extinction, self.ini["HBSPS_SFH"]["ExtinctionLaw"])
            syn_spectra *= 10 ** (
                -0.4 * extinction_law(self.observation["wavelength_fit"], av)
            ) / 10 ** (
                -0.4
                * extinction_law(
                    np.array([self.ini["HBSPS_SFH"]["wlNormRange"].mean()]), av
                )
            )
        else:
            av = None

        plt.figure()
        plt.subplot(111)
        plt.fill_between(
            self.observation["wavelength_fit"],
            self.observation["flux_fit"] - self.observation["cov_fit"] ** 0.5,
            self.observation["flux_fit"] + self.observation["cov_fit"] ** 0.5,
            color="k",
            alpha=0.3,
        )
        plt.plot(
            self.observation["wavelength_fit"],
            self.observation["flux_fit"],
            color="r",
            label="Obs",
        )
        plt.plot(
            self.observation["wavelength_fit"], syn_spectra, color="lime", label="Synth"
        )
        plt.annotate(
            f"v={los_vel:.1f} (km/s)"
            + "\n"
            + r"$\sigma=$"
            + f"{los_sigma:.1f} (km/s)"
            + "\n"
            + f"Av={av:.2f} (mag)",
            xy=(0.05, 0.95),
            xycoords="axes fraction",
            va="top",
        )
        plt.plot(
            self.observation["wavelength_fit"],
            self.observation["flux_fit"] - syn_spectra,
            color="orange",
            label="Residual",
        )
        plt.legend(ncol=3, bbox_to_anchor=(0.5, 1.01), loc="lower center")
        plt.show()

    @staticmethod
    def read_ini_file(path):
        print("Reading ini file: ", path)
        ini_info = {}
        with open(path, "r") as f:
            for line in f.readlines():
                line = line.replace("\n", "")
                if (len(line) == 0) or (line[0] == ";"):
                    continue
                if line[0] == "[":
                    module = line.strip("[]")
                    ini_info[module] = {}
                else:
                    components = line.split("=")
                    name = components[0].strip(" ")
                    str_value = components[1].replace(" ", "")
                    str_value = str_value.replace(".", "")
                    str_value = str_value.replace("e", "")
                    str_value = str_value.replace("-", "")
                    if not str_value.isnumeric():
                        ini_info[module][name] = components[1].strip(" ")
                    else:
                        numbers = [n for n in components[1].split(" ") if len(n) > 0]
                        if ("." in components[1]) or ("e" in components[1]):
                            # Float number
                            ini_info[module][name] = np.array(numbers, dtype=float)
                        else:
                            # int number
                            ini_info[module][name] = np.array(numbers, dtype=int)
        ini = cosmosis.runtime.Inifile(path)
        return ini_info, ini

    @staticmethod
    def read_chain_file(path):
        print("Reading chain results: ", path)
        with open(path, "r") as f:
            header = f.readline().strip("#")
            columns = header.replace("\n", "").split("\t")
        matrix = np.atleast_2d(np.loadtxt(path))
        results = {}
        ssp_weights = np.zeros(matrix.shape[0])
        last_ssp = 0
        for ith, par in enumerate(columns):
            results[par] = matrix[:, ith]
            if "ssp" in par:
                last_ssp += 1
                ssp_weights += 10 ** matrix[:, ith]
        if last_ssp > 0:
            print(f"Adding extra SSP {last_ssp + 1}")
            results[f"parameters--ssp{last_ssp + 1}"] = np.log10(
                np.clip(1 - ssp_weights, a_min=1e-4, a_max=None)
            )
        return results
