import os
import functools
import numpy as np
import psutil

def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def available_memory_bytes() -> int:
    return int(psutil.virtual_memory().available)
