"""Module dedicated to read and manipulate the output products of BESTA."""
import os
import functools
import numpy as np
import psutil
import cosmosis
from astropy.table import Table

from besta import pipeline_modules

def convert_bytes(size_bytes, to_unit):
    to_unit = to_unit.upper()
    if to_unit == 'B':
        return size_bytes
    elif to_unit == 'KB':
        return size_bytes / 1024
    elif to_unit == 'MB':
        return size_bytes / (1024 ** 2)
    elif to_unit == 'GB':
        return size_bytes / (1024 ** 3)
    else:
        raise ValueError("Unit must be 'B', 'KB', 'MB', or 'GB'")

def predict_array_memory(array_shape, dtype=np.float64, unit='MB'):
    """
    Predict the memory required for a NumPy array with given shape and data type.

    Parameters
    ----------
    array_shape : tuple
        Shape of the array (e.g., (1000, 1000) for a 1000x1000 array).
    dtype : numpy.dtype, optional
        NumPy data type (default is np.float64).
    unit : {'B', 'KB', 'MB', 'GB'}, optional
        Unit for the returned memory size (default is 'MB').

    Returns
    -------
    float
        Estimated memory requirement in the specified unit.

    Raises
    ------
    ValueError
        If invalid dtype or unit is provided.

    Examples
    --------
    >>> predict_array_memory((1000, 1000))
    7.63  # MB for float64 array

    >>> predict_array_memory((500, 500, 500), np.float32, 'GB')
    0.47  # GB for float32 array
    """
    try:
        dtype_obj = np.dtype(dtype)
        element_size = dtype_obj.itemsize
    except:
        raise ValueError("Invalid dtype provided")
    
    total_elements = np.prod(array_shape)
    total_bytes = total_elements * element_size
    return convert_bytes(total_bytes, unit)

def check_array_memory(array_shape, dtype=np.float64, unit='MB', safety_margin=0.2):
    """
    Check if the current machine has enough RAM for creating a given array.

    Parameters
    ----------
    array_shape : tuple
        Shape of the array (e.g., (1000, 1000) for a 1000x1000 array).
    dtype : numpy.dtype, optional
        NumPy data type (default is np.float64).
    unit : {'B', 'KB', 'MB', 'GB'}, optional
        Unit for error message reporting (default is 'MB').
    safety_margin : bool
        Additional margin (fraction of the array size) for ensuring good performance.
    Returns
    -------
    bool
        True if sufficient memory is available.

    Raises
    ------
    MemoryError
        If insufficient memory is available for the array.
    ValueError
        If invalid dtype, shape, or unit is provided.

    Examples
    --------
    >>> check_array_memory((10000, 10000))
    True  # If 800MB+ available

    >>> check_array_memory((50000, 50000))
    MemoryError: Insufficient memory available...
    """
    # Calculate required memory
    try:
        required_mem = predict_array_memory(array_shape, dtype, 'B')
    except ValueError as e:
        raise ValueError(f"Invalid input parameters: {str(e)}")
    
    # Get available memory
    available_mem = psutil.virtual_memory().available

    # Add safety margin
    total_needed = required_mem * (1 + safety_margin)

    # Convert for error message

    if total_needed > available_mem:
        req_mem_display = convert_bytes(required_mem, unit)
        avail_mem_display = convert_bytes(available_mem, unit)
        needed_mem_display = convert_bytes(total_needed, unit)

        raise MemoryError(
            f"Insufficient memory available. "
            f"Required: {req_mem_display:.2f} {unit} (plus safety margin), "
            f"Available: {avail_mem_display:.2f} {unit}, "
            f"Needed: {needed_mem_display:.2f} {unit}"
        )
    
    return True


def expand_env_vars(arg_spec=0):
    """
    Decorator that expands environment variables in a specified argument.

    The target argument can be specified either by position (int) or name (str).
    If no argument is specified, expands the first positional argument (default).

    Parameters
    ----------
    arg_spec : int or str, optional
        Either the position (0-based index) or name of the argument to expand.
        Default is 0 (first positional argument).

    Returns
    -------
    callable
        A decorator function that wraps the original function.

    Examples
    --------
    # Expand first positional argument (default)
    >>> @expand_env_vars()
    ... def load_file(path):
    ...     print(path)
    >>> load_file("$HOME/test.txt")

    # Expand named argument
    >>> @expand_env_vars('filename')
    ... def process_file(filename, mode='r'):
    ...     print(filename, mode)
    >>> process_file("${TMPDIR}/data.txt")

    # Expand argument at position 1
    >>> @expand_env_vars(1)
    ... def save_data(header, filepath):
    ...     print(header, filepath)
    >>> save_data("results", "$APPDIR/output.dat")
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Handle different argument specifications
            if isinstance(arg_spec, int):
                # Positional argument expansion
                if arg_spec < len(args):
                    expanded = os.path.expandvars(args[arg_spec])
                    args = args[:arg_spec] + (expanded,) + args[arg_spec+1:]
            elif isinstance(arg_spec, str):
                # Keyword argument expansion
                if arg_spec in kwargs:
                    kwargs[arg_spec] = os.path.expandvars(kwargs[arg_spec])
            else:
                raise TypeError("arg_spec must be int (position) or str (argument name)")

            return func(*args, **kwargs)
        return wrapper
    return decorator

@expand_env_vars()
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
        f.write(f"; File generated automatically by BESTA\n")
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
    values_filename = os.path.expandvars(config["pipeline"]["values"])
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
            f.write(f"; File generated automatically by BESTA\n")
            f.write("[parameters]\n")
            for name, lims in config[values_section].items():
                if type(lims) is str:
                    f.write(f"{name} = {lims}\n")
                else:
                    f.write(f"{name} = {lims[0]} {(lims[0] + lims[1]) / 2} {lims[1]}\n")
            f.write(f"; \(ﾟ▽ﾟ)/")

@expand_env_vars()
def read_results_file(path):
    """Read the results produced during a CosmoSIS run.

    Parameters
    ----------
    path : str
        Path to the file containing the cosmosis results

    Returns
    -------
    table : :class:`astropy.table.Table`
        Table containing the results.
    """
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip("#")
        columns = header.replace("\n", "").split("\t")
    matrix = np.atleast_2d(np.loadtxt(path))
    table = Table()
    if matrix.size > 1:
        for ith, c in enumerate(columns):
            table.add_column(matrix.T[ith], name=c.lower())
    return table


class Reader(object):
    r"""CosmoSIS run results reader.
    
    This class is meant to improve the accessibility to the results from a
    CosmoSIS run.

    The Reader can be initialised either from a .ini conguration file or from
    the file containing the results of the run (which implicitly includes the
    configuration information in the header).

    Currently, there are two factory methods to initialise the ``Reader``.

        >>> from besta.io import Reader
        >>> reader = Reader.from_ini_file(path_to_ini_config_file)
        >>> reader = Reader.from_results_file(path_to_results_file)

    From here, users may want to load the results of the run into a table:

        >>> reader.load_results()
        >>> reader.results_table.info

    The ``Reader`` also provides tools for extracting some solutions among the full
    run. For instance, users may be interested on using the solution that maximizes
    the likelihood (i.e., the best-fit solution):

        >>> best_fit = reader.get_maxlike_solution(log_prob="post", as_datablock=True)

    The first argument tells the function to use the row with the highest
    value of ``"post"`` (the column naming convention depends on the CosmoSIS
    sampler used).

    Alternatively, users might be interested in selecting a set of solutions based
    on their posterior probability or likelihood. For that purpose, the ``Reader``
    includes two methods that allow to pull a subset of the solutions:

    - Users can retrieve a fraction of the solutions with the highest posterior
    by calling:
        
        >>> reader.get_top_frac_solutions(frac=1)

    which will return the first top percent of all the solutions.

    - Users can retrieve a fraction of the solutions that accounts for a given
    fraction of the cumulative posterior.

        >>> reader.get_pct_solutions(pct=99)

    which will return the solutions that account for 99% of the total posterior.

    Besides selecting solutions, the ``Reader`` also provides access to the
    modules used during the sampling, allowing to recompute the observable quantities
    or to evaluate the fit:

        >>> module = reader.last_module
        >>> flux_model = module.make_observable(best_fit, parse=True)
    """

    @property
    def ini(self) -> dict:
        """Cosmosis .ini configuration file"""
        return self._ini

    @ini.setter
    def ini(self, value):
        self._ini = value

    @property
    def ini_file(self) -> str:
        """Path to the CosmoSIS configuration file."""
        return getattr(self, "_ini_file", None)

    @ini_file.setter
    def ini_file(self, value):
        self._ini_file = value

    @property
    def ini_values(self) -> dict:
        """Cosmosis configuration values."""
        return self._ini_values

    @ini_values.setter
    def ini_values(self, value):
        self._ini_values = value

    @property
    def values_file(self) -> str:
        """Path to the CosmoSIS (prior) values configuration file."""
        return getattr(self, "_ini_file", None)

    @values_file.setter
    def values_file(self, value):
        self._values_file = value

    @property
    def last_module_name(self) -> str:
        """Name of the last module used in the pipeline."""
        return self.ini["pipeline"]["modules"].split(" ")[-1].replace(" ", "")

    @property
    def last_module(self):
        """An instance of the last pipeline module used in the run."""
        return getattr(pipeline_modules, self.last_module_name + "Module")(
            self.ini)

    @property
    def config(self) -> dict:
        """Pipeline module configuration."""
        return self._config

    @config.setter
    def config(self, value):
        self._config = value

    @property
    def results_table(self) -> Table:
        """Table containing the CosmoSIS run results."""
        return self._results_table

    @results_table.setter
    def results_table(self, value):
        self._results_table = value

    @property
    def results_file(self) -> str:
        """Path to the file containing the CosmoSIS run results."""
        return self._results_file

    @results_file.setter
    def results_file(self, value):
        self._results_file = value

    def __init__(self, ini_file=None, results_file=None):
        if ini_file is not None:
            self.ini_file = ini_file
            self.ini = self.read_ini_file(self.ini_file)
        elif results_file is not None:
            self.results_file = results_file
            self.ini = self.read_ini_file_from_results(self.results_file)

        self.ini_values = self.read_ini_values_file(self.ini["pipeline"]["values"])
        self.config = {}

    def load_results(self):
        """Load the cosmosis run results associated to the ``ini`` file."""
        path = self.ini["output"]["filename"]
        if ".txt" not in path:
            path += ".txt"
        self.results_table = read_results_file(path)

    def get_maxlike_solution(self, log_prob="post", as_datablock=False,
                             **kwargs):
        """Get the maximum likelihood solution.

        Obtain the maximum likelihood solution from the ``results_table``

        Parameters
        ----------
        log_prob : str, optional
            Column name to use for computing the maximum likelihood. Default is
            ``post``.
        as_datablock : bool, optional
            If ``True``, the solution is converted to a :class:`DataBlock`.
        **kwargs :
            Additional arguments for converting the solution into a DataBlock.

        Returns
        -------
        solution : dict
            Dictionary containing the solution with the maximum likelihood.
        
        See also
        --------
        :func:`solution_to_datablock`
        """
        good_sample = self.results_table[log_prob] != 0
        maxlike_pos = np.nanargmax(self.results_table[good_sample][log_prob].value)
        solution = {}
        for k, v in self.results_table.items():
            if "parameters" in k:
                solution[k.replace("parameters--", "")] = v[good_sample][maxlike_pos]
        if as_datablock:
            solution = self.solution_to_datablock(solution, **kwargs)
        return solution

    def get_top_frac_solutions(self, frac=1, log_prob="post", as_datablock=False,
                          **kwargs):
        """Return a fraction of the solutions sorted by their posterior probability.

        This method provides a given fraction of the solutions with the highest
        posterior probability.

        Parameters
        ----------
        frac : float, optional
            Fraction of the solutions to return, e.g. ``pct=1`` will return
            the top 1 per cent with the highest probability.
        log_prob : str, optional
            Column name to use for computing the maximum likelihood. Default is
            ``post``.
        as_datablock : bool, optional
            If ``True``, the solution is converted to a :class:`DataBlock`.
        **kwargs :
            Additional arguments for converting the solution into a DataBlock.

        Returns
        -------
        all_solutions : list
            A list containing the solutions.
        
        See also
        --------
        :func:`solution_to_datablock`
        """
        good_sample = self.results_table[log_prob] != 0
        post_sort = np.argsort(self.results_table[log_prob][good_sample])
        first_row = int(post_sort.size / 100 * frac)
        solutions = self.results_table[post_sort][-first_row:]
        if as_datablock:
            all_solutions = [self.solution_to_datablock(
                dict(zip(solutions.keys(), sol[:]))) for sol in solutions]
        else:
            all_solutions = [dict(zip(solutions.keys(), sol[:])) for sol in solutions]
        return all_solutions

    def get_pct_solutions(self, pct=99, log_prob="post", as_datablock=False,
                          **kwargs):
        """Return the top percentile solutions.

        This method sorts all solutions based on their posterior probablity
        and returns a given fraction.

        Parameters
        ----------
        pct : float, optional
            Fraction of the solutions to return, e.g. ``pct=99`` will return
            the top 1 per cent with the highest probability.
        log_prob : str, optional
            Column name to use for computing the maximum likelihood. Default is
            ``post``.
        as_datablock : bool, optional
            If ``True``, the solution is converted to a :class:`DataBlock`.
        **kwargs :
            Additional arguments for converting the solution into a DataBlock.

        Returns
        -------
        all_solutions : list
            A list containing the solutions.
        
        See also
        --------
        :func:`solution_to_datablock`
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
            if as_datablock:
                solution = self.solution_to_datablock(solution, **kwargs)
            all_solutions.append(solution)
        return all_solutions

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

    @classmethod
    @expand_env_vars(1)
    def read_ini_file(cls, path):
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
        
        with open(path, "r") as f:
            return cls._parse_ini_lines(f.readlines())

    @classmethod
    @expand_env_vars(1)
    def read_ini_file_from_results(cls, path):
        with open(path, "r") as file:
            file_lines = file.readlines()
            line_start, line_end = [ith for ith, f in enumerate(file_lines) if (
                "START_OF_PARAMS_INI" in f) or ("END_OF_PARAMS_INI" in f)]
            return cls._parse_ini_lines(file_lines[line_start + 1:line_end])

    @staticmethod
    def _parse_ini_lines(lines):
        ini_info = {}
        for line in lines:
            line = line.strip("## ").replace("\n", "")
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
    @expand_env_vars()
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

    @classmethod
    def from_ini_file(cls, path_to_ini):
        return cls(ini_file=path_to_ini)

    @classmethod
    def from_results_file(cls, path_to_results):
        return cls(results_file=path_to_results)
