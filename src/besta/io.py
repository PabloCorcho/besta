"""Module dedicated to read and manipulate the output products of BESTA."""
import os
import functools
import re
import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import psutil
import cosmosis
from cosmosis import Inifile
from cosmosis.datablock import DataBlock, SectionOptions
from astropy.table import Table

from besta.logging import get_logger, setup_logging
from besta.utils import expand_env_vars

logger = get_logger(__name__)


_NUM_RE = re.compile(r"""
    ^[+-]?
    (?:
        (?:\d+(?:\.\d*)?) | (?:\.\d+)
    )
    (?:[eE][+-]?\d+)?$
""", re.VERBOSE)

def _split_top_level(s: str, *, split_on_whitespace=True) -> list[str]:
    """Split a string into top-level tokens, respecting quotes and nested parentheses/brackets."""
    s = s.strip()
    if not s:
        return []

    tokens = []
    current = []
    quote = None
    escape = False
    paren_depth = 0
    bracket_depth = 0

    def flush_current():
        token = "".join(current).strip()
        if token:
            tokens.append(token)
        current.clear()

    for char in s:
        if quote is not None:
            current.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue

        if char in {"'", '"'}:
            quote = char
            current.append(char)
        elif char == "(":
            paren_depth += 1
            current.append(char)
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)
            current.append(char)
        elif char == "[":
            bracket_depth += 1
            current.append(char)
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
            current.append(char)
        elif paren_depth == 0 and bracket_depth == 0 and (
            char == "," or (split_on_whitespace and char.isspace())
        ):
            flush_current()
        else:
            current.append(char)

    flush_current()
    return tokens

def _split_tokens(s: str) -> list[str]:
    """Split a string into tokens, respecting quotes and nested parentheses/brackets."""
    return _split_top_level(s, split_on_whitespace=True)

def _is_wrapped_group(token: str) -> bool:
    """Check if a token is a wrapped group like (1, 2, 3) or [1, 2, 3]."""
    token = token.strip()
    if len(token) < 2:
        return False

    pairs = {"(": ")", "[": "]"}
    opening = token[0]
    closing = pairs.get(opening)
    if closing is None or token[-1] != closing:
        return False

    quote = None
    escape = False
    depth = 0
    for index, char in enumerate(token):
        if quote is not None:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue

        if char in {"'", '"'}:
            quote = char
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0 and index != len(token) - 1:
                return False

    return depth == 0 and quote is None


def _sequence_to_numpy(values, *, force_sequence=False):
    """Coerce a list of values into a numpy array if all elements are numeric, otherwise keep as list."""
    if not values:
        return []
    if len(values) == 1 and not force_sequence:
        return values[0]

    if all(isinstance(x, (int, float)) for x in values):
        dtype = float if any(isinstance(x, float) for x in values) else int
        return np.array(values, dtype=dtype)
    return values


def _parse_group(token: str):
    inner = token[1:-1].strip()
    if not inner:
        return []

    values = [_parse_token(part) for part in _split_tokens(inner)]
    return _sequence_to_numpy(values, force_sequence=True)

def _parse_scalar(token: str):
    low = token.lower()
    if low in {"t", "true", "yes", "on"}:
        return True
    if low in {"f", "false", "no", "off"}:
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

def _parse_token(token: str):
    token = token.strip()
    if _is_wrapped_group(token):
        return _parse_group(token)
    return _parse_scalar(token)


def _split_keyword_arg(token: str):
    quote = None
    escape = False
    paren_depth = 0
    bracket_depth = 0

    for index, char in enumerate(token):
        if quote is not None:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue

        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(0, paren_depth - 1)
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif char == "=" and paren_depth == 0 and bracket_depth == 0:
            key = token[:index].strip()
            value = token[index + 1 :].strip()
            if not key:
                raise ValueError(f"Invalid keyword argument: {token!r}")
            return key, value

    return None

def _parse_value(value: str):
    if value == "":
        return ""

    tokens = _split_tokens(value)
    parsed = [_parse_token(token) for token in tokens]
    return _sequence_to_numpy(parsed)

def string_to_func_args(text: str):
    """Parse a string of the form 'arg1, arg2, key=value' into separate args and kwargs.
    
    Parameters
    ----------
    text : str
        Input string containing positional and keyword arguments.
    
    Returns
    -------
    args : list
        List of positional arguments.
    kwargs : dict
        Dictionary of keyword arguments.
    """
    args = []
    kwargs = {}
    seen_keyword = False

    for token in _split_top_level(text, split_on_whitespace=False):
        key_value = _split_keyword_arg(token)
        if key_value is None:
            if seen_keyword:
                raise ValueError(
                    "Positional arguments cannot appear after keyword arguments."
                )
            args.append(_parse_value(token))
            continue

        seen_keyword = True
        key, value = key_value
        kwargs[key] = _parse_value(value)

    return args, kwargs

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
def read_results_file(path, delimiter="\t"):
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
        header = f.readline()
    if not header.startswith("#"):
        raise ValueError("Expected first line header starting with '#'.")

    columns = [col.strip().lower() for col in header.strip("# \n").split(delimiter)]
    matrix = np.atleast_2d(np.loadtxt(path, delimiter=delimiter, comments="#"))
    table = Table()
    if matrix.size <= 1:
        return table
    if matrix.shape[1] != len(columns):
        raise ValueError(
            f"Data has {matrix.shape[1]} columns but header lists {len(columns)}."
        )
    for ith, name in enumerate(columns):
        table[name] = matrix[:, ith]
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

def parse_table_format(path):
    """Parse the format of a table file based on its extension.
    
    Parameters
    ----------
    path : str
        Path to the table file.
    
    Returns
    -------
    format : str
        Format of the table file (e.g., "csv", "fits", "ascii").
    """
    extension = os.path.splitext(path)[1].lower()
    if extension in [".csv"]:
        format = "csv"
    elif extension in [".fits"]:
        format = "fits"
    elif extension in [".txt", ".dat"]:
        format = "ascii"
    else:
        logger.warning(f"Could not guess file format from extension '{extension}'. Defaulting to ASCII.")
        format = "ascii"

    return format

def burn_table(table, nwalkers: int, burn_in: int) -> Table:
    """Discard the first `burn_in` samples per walker from the table."""
    nrows = len(table)
    expected = nwalkers * burn_in
    if nrows < expected:
        raise ValueError(f"Not enough rows in table ({nrows}) for burn_in={burn_in} and nwalkers={nwalkers} (expected at least {expected}).")
    # Keep rows after burn-in for each walker
    mask = np.ones(nrows, dtype=bool)
    for w in range(nwalkers):
        start = w * burn_in
        end = (w + 1) * burn_in
        mask[start:end] = False
    return table[mask]

def _select_parameter_keys(
    table: Table,
    *,
    parameter_prefix: str = "--",
    parameter_keys: Optional[Sequence[str]] = None,
) -> List[str]:
    if parameter_keys is not None:
        keys = list(parameter_keys)
    else:
        keys = [k for k in table.colnames if parameter_prefix in k]
    if len(keys) == 0:
        raise ValueError("No parameter keys found/selected.")
    return keys

def _split_param_key(key: str, prefix: str = "--") -> Tuple[str, str]:
    """
    Split a parameter key into (section, name) using the delimiter/prefix.

    If the key cannot be split, returns ("", key).
    """
    if prefix in key:
        sect, name = key.split(prefix, 1)
        return sect, name
    return "", key


###################
# Main reader class
###################

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
        """Set the parsed CosmoSIS configuration dictionary."""
        self._ini = value

    @property
    def ini_file(self) -> str:
        """Path to the CosmoSIS configuration file."""
        return getattr(self, "_ini_file", None)

    @ini_file.setter
    def ini_file(self, value):
        """Set the path to the CosmoSIS configuration file."""
        self._ini_file = value

    @property
    def ini_values(self) -> dict:
        """Cosmosis configuration values."""
        return self._ini_values

    @ini_values.setter
    def ini_values(self, value):
        """Set the parsed CosmoSIS values dictionary."""
        self._ini_values = value

    @property
    def values_file(self) -> str:
        """Path to the CosmoSIS (prior) values configuration file."""
        return getattr(self, "_values_file", None)

    @values_file.setter
    def values_file(self, value):
        """Set the path to the CosmoSIS values file."""
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
        logger.debug(f"Loaded module {module_name} from {self.ini[module_name]['file']}")
        return module(self.ini, alias=module_name)

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
        """Set the cached pipeline-module configuration."""
        self._config = value

    @property
    def results_table(self) -> Table:
        """Table containing the CosmoSIS run results."""
        return self._results_table

    @results_table.setter
    def results_table(self, value):
        """Set the table containing CosmoSIS run results."""
        self._results_table = value

    @property
    def results_file(self) -> str:
        """Path to the file containing the CosmoSIS run results."""
        return self._results_file

    @results_file.setter
    def results_file(self, value):
        """Set the path to the CosmoSIS results file."""
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
        
        # setup logging based on the first module in the pipeline (if any)
        if self.modules:
            logging_console = self.ini[self.modules[0]].get("logging_console")
            logging_level = self.ini[self.modules[0]].get("logging_level", "INFO").upper()
            logging_overwrite = self.ini[self.modules[0]].get("logging_overwrite", False)
            logging_file = self.ini[self.modules[0]].get("logging_file", None)

            setup_logging(level=logging_level, log_file=logging_file,
                          overwrite=logging_overwrite, console=logging_console)


    def load_results(self, burn_in=0, nwalkers=1):
        """Load the cosmosis run results associated to the ``ini`` file.
        
        Parameters
        ----------
        burn_in : int, optional
            Number of initial samples to discard per walker. Default is 0 (no burn-in).
        nwalkers : int, optional
            Number of walkers used during the sampling. Required if burn_in > 0. Default is 1.
        """
        path = self.ini["output"]["filename"]
        if ".txt" not in path:
            path += ".txt"
        self.results_table = read_results_file(path)
        if burn_in > 0:
            self.results_table = burn_table(
                self.results_table,
                nwalkers=self.ini["pipeline"].get("nwalkers", nwalkers), burn_in=burn_in)

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

        if as_datablock:
            return self.solution_to_datablock(tab[maxlike_pos], **kwargs)

        else:
            return self.solution_to_dict(tab[maxlike_pos], **kwargs)

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
        if not (0 < frac <= 100):
            raise ValueError("Fraction must be in (0, 100].")
        good_sample = np.isfinite(self.results_table[log_prob])
        tab = self.results_table[good_sample]
        post_sort = np.argsort(tab[log_prob])
        # Select the top frac per cent
        first_row = max(1, np.ceil(post_sort.size / 100 * frac))
        solutions = tab[post_sort][-first_row:]
        if as_datablock:
            all_solutions = [self.solution_to_datablock(sol, **kwargs) for sol in solutions]
        else:
            all_solutions = [self.solution_to_dict(sol, **kwargs) for sol in solutions]
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

        return [self.solution_to_dict(row) for row in selected]

    def solution_to_dict(self, sample, *, add_fixed=True, extra_params=None):
        """TODO"""

        solution = {}

        for (sect, name) in self.ini_values_free.keys():
            solution[f"{sect}--{name}"] = sample[f"{sect}--{name}"]

        if add_fixed:
            for (sect, name), v in self.ini_values_fixed.items():
                solution[f"{sect}--{name}"] = v
        
        if extra_params is not None:
            for sect, name in extra_params:
                solution[f"{sect}--{name}"] = sample[f'{sect}--{name}']

        return solution

        
    def solution_to_datablock(self, solution: dict, add_fixed=True, extra_params=None):
        """Convert a solution into a DataBlock.

        Parameters
        ----------
        solution : dict-like
            A dictionary-like containing the parameter values.
        add_fixed : bool, optional
            If True, add the default values of the fixed parameters from the ini
            values config file. Default is True.
        extra_params : iterable, optional
            An iterable containing pairs (section, name) of additional parameters
            contained in ``solution`` to be included in the datablock.

        Returns
        -------
        datablock : DataBlock
            The DataBlock containing the input solution.
        """
        datablock = cosmosis.DataBlock()
        for (sect, name) in self.ini_values_free.keys():
            datablock[sect, name] = solution[f'{sect}--{name}']
        
        if add_fixed:
            for (sect, name), v in self.ini_values_fixed.items():
                datablock[sect, name] = v
        
        if extra_params is not None:
            for sect, name in extra_params:
                datablock[sect, name] = solution[f'{sect}--{name}']

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
        """Read the embedded CosmoSIS ini block from a results file.

        Parameters
        ----------
        path : str
            Path to the results file containing the ini block.

        Returns
        -------
        dict
            Parsed ini configuration stored in the results header.
        """
        with open(path, "r") as file:
            file_lines = file.readlines()
            line_start, line_end = [ith for ith, f in enumerate(file_lines) if (
                "START_OF_PARAMS_INI" in f) or ("END_OF_PARAMS_INI" in f)]
            content = "".join([l.replace("## ", "") for l in file_lines[line_start + 1:line_end]])
            return _ini_string_to_dict(content)

    @classmethod
    def from_ini_file(cls, path_to_ini):
        """Create a reader from a CosmoSIS ini file."""
        return cls(ini_file=path_to_ini)

    @classmethod
    def from_results_file(cls, path_to_results):
        """Create a reader from a CosmoSIS results file."""
        return cls(results_file=path_to_results)
