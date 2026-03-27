"""General utility helpers used across BESTA."""

import os
import functools
import numpy as np
import psutil

def mkdir(path: str) -> None:
    """Create a directory path if it does not already exist."""

    os.makedirs(path, exist_ok=True)

def available_memory_bytes() -> int:
    """Return the currently available system memory in bytes."""

    return int(psutil.virtual_memory().available)

def convert_bytes(size_bytes, to_unit):
    """Convert a byte count into ``B``, ``KB``, ``MB``, or ``GB``."""

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
    except Exception as e:
        raise ValueError(f"Invalid dtype provided: {e}")
    
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
    safety_margin : float, optional
        Additional margin (fraction of the array size) for ensuring good performance.
        Default is 0.2 (20% more than the predicted memory requirement).
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
    available_mem = available_memory_bytes()

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
