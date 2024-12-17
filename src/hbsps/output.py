import os
import numpy as np
import cosmosis

from hbsps.postprocess import read_results_file


def make_ini_file(filename, config):
    """Create a .ini file from an input configuration.

    Parameters
    ----------
    filename : str
        Output file name.
    config : dict
        Dictionary containing the configuration parameters.
    """
    print(f"Writing .ini file: {filename}")
    with open(filename, "w") as f:
        f.write(f"; File generated automatically\n")
        for section in config.keys():
            # Ignore the Values section
            if section == "Values":
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


def make_values_file(config, overwrite=True):
    """Make a values.ini file from the configuration.

    Parameters
    ----------
    config : dict
        Configuration parameters
    """
    values_filename = config["pipeline"]["values"]
    values_section = f"Values"

    if os.path.isfile(values_filename):
        print(f"File containing the .ini priors already exists at {values_filename}")
        if not overwrite:
            return
        else:
            print("Overwritting file")
    if values_section in config:
        print(f"Creating values file: {values_filename}")
        with open(values_filename, "w") as f:
            f.write("[parameters]\n")
            for name, lims in config[values_section].items():
                if type(lims) is str:
                    f.write(f"{name} = {lims}\n")
                else:
                    f.write(f"{name} = {lims[0]} {(lims[0] + lims[1]) / 2} {lims[1]}\n")


class Reader(object):
    """Cosmosis Data Reader

    Attributes
    ----------
    ini_file :
    ini :
    ini_data :
    config :
    last_module :
    """

    @property
    def ini(self):
        """Cosmosis .ini configuration file"""
        return self._ini

    @ini.setter
    def ini(self, value):
        self._ini = value

    @property
    def ini_file(self):
        """Cosmosis configuration file path."""
        return self._ini_file

    @ini_file.setter
    def ini_file(self, value):
        self._ini_file = value

    @property
    def ini_values(self):
        """Cosmosis configuration values."""
        return self._ini_values

    @ini_values.setter
    def ini_values(self, value):
        self._ini_values = value

    @property
    def last_module(self):
        """Last module used in the pipeline."""
        return self.ini["pipeline"]["modules"].split(" ")[-1].replace(" ", "")

    @property
    def config(self):
        """Pipeline configuration."""
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    @property
    def results_table(self):
        return self._results_table

    @results_table.setter
    def results_table(self, value):
        self._results_table = value

    def __init__(self, ini_file):
        self.ini_file = ini_file
        self.ini = self.read_ini_file(self.ini_file)
        self.ini_values = self.read_ini_values_file(self.ini["pipeline"]["values"])
        self.config = {}

    # TODO> READ COSMOSIS PROCESSED FILES
    # def load_processed_file(self, kind="medians"):
    #     filename = os.path.join(
    #         os.path.dirname(self.ini["output"]["filename"]), f"{kind}.txt"
    #     )
    #     if os.path.isfile(filename):
    #         parameters = {}
    #         with open(filename) as f:
    #             for line in f.readlines():
    #                 line_content = (
    #                     line.replace("\n", "").replace("\t", "  ").split("  ")
    #                 )
    #                 if line_content[0] == "#":
    #                     continue
    #                 if "parameters" in line_content[0]:
    #                     name = line_content[0].replace("parameters--", "")
    #                     if kind == "maxlike":
    #                         parameters[name] = {
    #                             f"{kind}": float(line_content[1]),
    #                         }
    #                     else:
    #                         parameters[name] = {
    #                             f"{kind}": float(line_content[1]),
    #                             "sigma": float(line_content[2]),
    #                         }
    #         return parameters
    #     else:
    #         return None

    def load_results(self):
        """Load the cosmosis run results associated to the ``ini`` file."""
        path = self.ini["output"]["filename"]
        if ".txt" not in path:
            path += ".txt"
        self.results_table = read_results_file(path)

    def get_chain_percentiles(self, pct=[0.5, 0.16, 0.50, 0.84, 0.95]):
        """Compute the percentile values of the parameter chains.

        Parameters
        ----------
        pct : list
            List of input percentiles.

        Returns
        -------
        pct_results : dict
            Dictionary containing the percentiles associated to each parameter.
        """
        parameters = [par for par in self.results_table.keys() if "parameters" in par]
        pct_results = {"percentiles": np.array(pct)}
        for par in parameters:
            sort_pos = np.argsort(self.results_table[par])
            cum_distrib = np.cumsum(self.results_table["weight"][sort_pos])
            cum_distrib /= cum_distrib[-1]
            pct_results[par] = np.interp(
                pct, cum_distrib, self.results_table[par][sort_pos]
            )
        return pct_results

    def get_maxlike_solution(self, log_prob="post"):
        """Get the maximum likelihood solution.

        Obtain the maximum likelihood solution from the ``results_table``

        Parameters
        ----------
        log_prob : str, optional
            Column name to use for computing the maximum likelihood. Default is
            ``post``.

        Returns
        -------
        solution : dict
            Dictionary containing the solution with the maximum likelihood.
        """
        good_sample = self.results_table[log_prob] != 0
        maxlike_pos = np.nanargmax(self.results_table[good_sample][log_prob].value)
        solution = {}
        for k, v in self.results_table.items():
            if "parameters" in k:
                solution[k.replace("parameters--", "")] = v[good_sample][maxlike_pos]

        return solution

    def solution_to_datablock(self, solution, section="parameters"):
        """Convert a solution into a DataBlock.

        Parameters
        ----------
        solution : dict
            A dictionary containing the parameter values.
        section : str, optional
            Name of the DataBlock section where to store the solution. Default
            is ``parameters``.

        Returns
        -------
        datablock : DataBlock
            The DataBlock containing the input solution.
        """
        keys = list(solution.keys())
        values = list(solution.values())
        strip_keys = []
        for k in keys:
            if section in k:
                strip_keys.append(k.replace(f"{section}--", ""))
            else:
                strip_keys.append(k)
        solution = {k: v for k, v in zip(strip_keys, values)}

        for parameter in self.ini_values["parameters"].keys():
            if parameter not in solution:
                print(f"Parameter {parameter} was set constant, adding default value")
                solution[parameter] = self.ini_values["parameters"][parameter]

        datablock = cosmosis.DataBlock()
        for k, v in solution.items():
            datablock[section, k] = v
        return datablock

    def get_pct_solutions(self, pct=99, log_prob="post"):
        """Return the top percentile solutions.

        This method sorts all solutions based on their posterior probablity
        and returns a given fraction.

        Parameters
        ----------
        pct : float, optional
            Fraction of the solutions to return, e.g. ``pct=99`` will return
            the 1 per cent with the highest probability.
        log_prob : str, optional
            Column name to use for computing the maximum likelihood. Default is
            ``post``.

        Returns
        -------
        all_solutions : list
            A list containing the solutions in the form of dictionaries.
        """
        good_sample = self.results_table[log_prob] != 0
        maxlike_pos = np.argmax(self.results_table[log_prob][good_sample])
        # Normalize the weights
        weights = (
            self.results_table[log_prob][good_sample]
            - self.results_table[log_prob][good_sample][maxlike_pos]
        )
        weights = np.exp(weights)
        weights /= np.nansum(weights)
        # From the highest to the lowest weight
        sort = np.argsort(weights)
        cum_weights = np.cumsum(weights[sort][::-1])
        last_sample = np.searchsorted(cum_weights, pct / 100)
        print(f"Selecting solutions from {last_sample}")
        all_solutions = []
        for i in range(-last_sample, 0):
            solution = {"weights": weights[sort][i]}
            for k, v in self.results_table.items():
                if "parameters" in k:
                    solution[k.replace("parameters--", "")] = v[good_sample][sort][i]
            all_solutions.append(solution)
        return all_solutions

    @staticmethod
    def read_ini_file(path):
        """Read the cosmosis configuration .ini file.

        Parameters
        ----------
        path : str
            Path to the file.

        Returns
        -------
        ini : dict
            Dictionary containing the information from the ini file.
        """
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
                        if len(numbers) == 1:
                            if ("." in components[1]) or ("e" in components[1]):
                                # Float number
                                ini_info[module][name] = float(numbers[0])
                            else:
                                # int number
                                ini_info[module][name] = int(numbers[0])
                        else:
                            if ("." in components[1]) or ("e" in components[1]):
                                # Float number
                                ini_info[module][name] = np.array(numbers, dtype=float)
                            else:
                                # int number
                                ini_info[module][name] = np.array(numbers, dtype=int)
        return ini_info

    @staticmethod
    def read_ini_values_file(path):
        """Read the cosmosis configuration .ini file.

        Parameters
        ----------
        path : str
            Path to the file.

        Returns
        -------
        ini_values : dict
            Dictionary containing the ini_values information.
        """
        print("Reading ini values file: ", path)
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
                        if len(numbers) == 1:
                            if ("." in components[1]) or ("e" in components[1]):
                                # Float number
                                ini_info[module][name] = float(numbers[0])
                            else:
                                # int number
                                ini_info[module][name] = int(numbers[0])
                        else:
                            if ("." in components[1]) or ("e" in components[1]):
                                # Float number
                                ini_info[module][name] = np.array(numbers, dtype=float)
                            else:
                                # int number
                                ini_info[module][name] = np.array(numbers, dtype=int)
        return ini_info
