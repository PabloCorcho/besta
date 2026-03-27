"""Module dedicated to read and manipulate the output products of BESTA."""
import os
import functools
import re
import importlib.util
import sys
from pathlib import Path

import numpy as np

import psutil
import cosmosis
from cosmosis import Inifile
from cosmosis.datablock import DataBlock, SectionOptions
from astropy.table import Table

from besta import pipeline_modules
from besta.logging import get_logger
from besta.utils import expand_env_vars

logger = get_logger(__name__)


_NUM_RE = re.compile(r"""
    ^[+-]?
    (?:
        (?:\d+(?:\.\d*)?) | (?:\.\d+)
    )
    (?:[eE][+-]?\d+)?$
""", re.VERBOSE)

def _split_tokens(s: str) -> list[str]:
    # Protect quoted strings with spaces
    if s.startswith(("'", '"')) and s.endswith(("'", '"')) and s[0] == s[-1]:
        return [s[1:-1]]
    # split on whitespace and/or commas
    if "," in s:
        s = s.replace(",", " ")
    return [t for t in s.split() if t]

def _parse_scalar(token: str):
    low = token.lower()
    if low in {"true", "yes", "on"}:
        return True
    if low in {"false", "no", "off"}:
        return False
    if low in {"none", "null"}:
        return "none"

    if _NUM_RE.match(token):
        # choose int vs float
        if any(c in token for c in ".eE"):
            return float(token)
        return int(token)

    # keep as string (unquote if it looks quoted)
    if (len(token) >= 2) and ((token[0] == token[-1] == "'") or (token[0] == token[-1] == '"')):
        return token[1:-1]
    return token

def _parse_value(value: str):
    if value == "":
        return ""

    tokens = _split_tokens(value)
    # If it's a single token, return a scalar
    if len(tokens) == 1:
        return _parse_scalar(tokens[0])

    # Multi-token: try numeric array first; otherwise keep as list of scalars
    scalars = [_parse_scalar(t) for t in tokens]
    if all(isinstance(x, (int, float)) for x in scalars):
        dtype = float if any(isinstance(x, float) for x in scalars) else int
        return np.array(scalars, dtype=dtype)
    return scalars

def _ini_file_to_dict(path):
    ini = Inifile(path)
    ini_dict = {}
    values = [ini.items(s) for s in ini.sections()]
    for sec, params in zip(ini.sections(), values):
        ini_dict[sec] = {}
        for k, v in params:
            ini_dict[sec][k] = _parse_value(v)
    return ini_dict

def _ini_string_to_dict(text):
    ini = Inifile(None)
    ini.read_string(text)
    ini_dict = {}
    values = [ini.items(s) for s in ini.sections()]
    for sec, params in zip(ini.sections(), values):
        ini_dict[sec] = {}
        for k, v in params:
            ini_dict[sec][k] = _parse_value(v)
    return ini_dict

@expand_env_vars()
def make_ini_file(filename, config, ignore_sec="values"):
    """Create a .ini file from an input configuration.

    Parameters
    ----------
    filename : str
        Output file name.
    config : dict
        Dictionary containing the configuration parameters.
    """
    logger.info("Writing .ini file: %s", filename)
    with open(filename, "w") as f:
        f.write(f"; File generated automatically by BESTA\n")
        for section in config.keys():
            # Ignore the Values section
            if section.lower() == ignore_sec.lower():
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
        f.write(r"; \(ﾟ▽ﾟ)/")


def make_values_file(config, overwrite=True, values_sec="values"):
    """Make a values.ini file from the configuration.

    Parameters
    ----------
    config : dict
        Configuration parameters
    """
    values_filename = os.path.expandvars(config["pipeline"]["values"])

    if os.path.isfile(values_filename):
        logger.warning(
            "File containing the .ini priors already exists at %s", values_filename
        )
        if not overwrite:
            return
        else:
            logger.info("Overwriting file")
    if values_sec in config:
        logger.info("Writing values file to: %s", values_filename)
        make_ini_file(values_filename, config[values_sec], ignore_sec=None)

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

def load_class_from_path(file_path, class_name):
    """Load a class object from a Python file path at runtime."""

    file_path = Path(file_path)

    # Create module spec
    spec = importlib.util.spec_from_file_location(
        file_path.stem,  # module name
        file_path
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[file_path.stem] = module
    spec.loader.exec_module(module)

    return getattr(module, class_name)

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
    def modules(self) -> list:
        """List of modules used in the pipeline as specified in the ini file."""
        m = self.ini["pipeline"]["modules"]
        if isinstance(m, str):
            return [m]
        return m

    @property
    def module_names(self) -> list:
        """List of module names used in the pipeline."""
        return [mod.replace(" ", "").split("_")[0] + "Module" for mod in self.modules]

    def get_module(self, module_name):
        """Get a pipeline module by name.

        Parameters
        ----------
        module_name : str
            Name of the module to retrieve.

        Returns
        -------
        module : instance
            An instance of the requested pipeline module.
        """
        if module_name not in self.modules:
            raise ValueError(f"Module {module_name} not found in the pipeline.")
        module = load_class_from_path(self.ini[module_name]["file"], "module")
        return module(self.ini, alias=module_name)
        # if not hasattr(pipeline_modules, module_class):
        #     raise ValueError(
        #         f"Module class {module_class} not found in besta.pipeline_modules.")
        # return getattr(pipeline_modules, module_class)(options)

    #TODO: deprecate
    @property
    def last_module(self):
        """An instance of the last pipeline module used in the run."""
        return self.get_module(self.modules[-1])

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
        else:
            raise ValueError("Must provide either ini or results file")

        self.ini_values = self.read_ini_file(self.ini["pipeline"]["values"])
        self.ini_values_free = {
            (sect, key): val
            for sect, params in self.ini_values.items()
            for key, val in params.items() if not isinstance(val, (int, float))
        }
        self.ini_values_fixed = {
            (sect, key): val
            for sect, params in self.ini_values.items()
            for key, val in params.items() if (sect, key) not in self.ini_values_free
        }

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
        good_sample = np.isfinite(self.results_table[log_prob])
        tab = self.results_table[good_sample]
        maxlike_pos = np.nanargmax(tab[log_prob].value)
        solution = {}
        if as_datablock:
            return self.solution_to_datablock(tab[maxlike_pos], **kwargs)
        for (sect, name) in self.ini_values_free.keys():
            solution[f"{sect}--{name}"] = tab[f"{sect}--{name}"][maxlike_pos]
        return solution

    def get_top_frac_solutions(self, frac=1, log_prob="post", as_datablock=False,
                          **kwargs):
        """Return a fraction of the solutions sorted by their posterior probability.

        This method provides a given fraction of the solutions with the highest
        posterior probability.

        Parameters
        ----------
        frac : float, optional
            Percentage of the solutions to return, e.g. ``frac=10`` will return
            the top 10 per cent with the highest probability.
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
        assert frac > 0 and frac <= 100, "Fraction must be in (0, 100]"
        good_sample = np.isfinite(self.results_table[log_prob])
        tab = self.results_table[good_sample]
        post_sort = np.argsort(tab[log_prob])
        # Select the top frac per cent
        first_row = max(1, np.ceil(post_sort.size / 100 * frac))
        solutions = tab[post_sort][-first_row:]
        if as_datablock:
            all_solutions = [self.solution_to_datablock(sol) for sol in solutions]
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
        if not (0 < pct <= 100):
            raise ValueError("pct must be in (0, 100].")

        lp = np.asarray(self.results_table[log_prob])
        good = np.isfinite(lp)
        tab = self.results_table[good]
        if len(tab) == 0:
            return []

        lp = np.asarray(tab[log_prob], dtype=float)
        # normalized weights
        lp_shift = lp - np.max(lp)
        w = np.exp(lp_shift)
        w_sum = np.sum(w)
        if not np.isfinite(w_sum) or w_sum <= 0:
            # fallback: pick the max only
            idx_sorted = np.array([np.argmax(lp)])
            selected = tab[idx_sorted]
        else:
            w /= w_sum

            # Sort by decreasing weight
            idx_sorted = np.argsort(w)[::-1]
            w_sorted = w[idx_sorted]
            cum = np.cumsum(w_sorted)

            target = pct / 100.0
            # first index where cum >= target, inclusive
            k = int(np.searchsorted(cum, target, side="left")) + 1
            selected = tab[idx_sorted[:k]]

        if as_datablock:
            return [self.solution_to_datablock(dict(row), **kwargs) if not isinstance(row, dict)
                    else self.solution_to_datablock(row, **kwargs)
                    for row in selected]

        colnames = list(selected.colnames)
        return [{name: row[name] for name in colnames} for row in selected]

    def solution_to_datablock(self, solution: dict):
        """Convert a solution into a DataBlock.

        Parameters
        ----------
        solution : dict-like
            A dictionary-like containing the parameter values.

        Returns
        -------
        datablock : DataBlock
            The DataBlock containing the input solution.
        """
        datablock = cosmosis.DataBlock()
        for (sect, name) in self.ini_values_free.keys():
            datablock[sect, name] = solution[f'{sect}--{name}']
        for (sect, name), v in self.ini_values_fixed.items():
            datablock[sect, name] = v
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
        logger.info("Reading ini file: %s", path)
        return _ini_file_to_dict(path)

    @classmethod
    @expand_env_vars(1)
    def read_ini_file_from_results(cls, path):
        with open(path, "r") as file:
            file_lines = file.readlines()
            line_start, line_end = [ith for ith, f in enumerate(file_lines) if (
                "START_OF_PARAMS_INI" in f) or ("END_OF_PARAMS_INI" in f)]
            content = "".join([l.replace("## ", "") for l in file_lines[line_start + 1:line_end]])
            return _ini_string_to_dict(content)

    @classmethod
    def from_ini_file(cls, path_to_ini):
        return cls(ini_file=path_to_ini)

    @classmethod
    def from_results_file(cls, path_to_results):
        return cls(results_file=path_to_results)
