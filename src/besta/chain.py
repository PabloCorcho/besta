"""Pass the result of one pipeline step to the next one.

A multi-step :class:`besta.pipeline.MainPipeline` can use a point estimate of
the previous run (MAP, posterior mean or posterior median) for the parameters
that are free in the values file of the next run. Parameters can be

- ``"fix"``: fixed to the estimate (default),
- ``"start"``: given the estimate as starting value, keeping their range, or
- ``"gaussian"``: given a Gaussian prior centred on the estimate, with the
  posterior width of the previous run (truncated to the range of the values
  file).

The estimate is computed according to the sampler of the previous run, since
what its results file contains depends on it (optimisers return the best
fit(s), MCMC samplers an unweighted chain that needs a burn-in, nested samplers
weighted samples). Original files are never modified: the next run uses copies
of its ini, values and priors files, and a ``.json`` file records what was
passed.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
from astropy.table import Table

from besta import io
from besta import postprocess
from besta.logging import get_logger

logger = get_logger(__name__)

ESTIMATORS = ("map", "mean", "median")
MODES = ("fix", "start", "gaussian")

DEFAULT_CHAIN_SETTINGS = {
    "estimator": "map",
    "mode": "fix",
    "include": None,
    "sigma_scale": 1.0,
    "burn_in": "auto",
}

# Kind of results written by each CosmoSIS sampler (``Sampler.sampler_outputs``)
_OPTIMIZERS = {"maxlike", "pmaxlike", "minuit", "gridmax", "test"}
_CHAINS = {"emcee", "zeus", "kombine", "metropolis", "pymc"}
_WEIGHTED = {"nautilus", "dynesty", "pocomc", "pmc", "multinest", "polychord", "abc"}
_NO_POSTERIOR = {"apriori", "latinhypercube", "grid", "list", "snake", "star",
                 "fisher", "importance", "cosmopower"}


def sampler_kind(name: str) -> str:
    """Kind of results of a CosmoSIS sampler.

    Returns ``"optimizer"`` (best fit rows, no posterior), ``"chain"``
    (unweighted chain, needs a burn-in), ``"weighted"`` (weighted samples,
    ``log_weight`` or ``weight`` column), ``"none"`` (the sampler does not draw
    from a posterior) or ``"unknown"``.
    """
    name = str(name).strip().lower()
    if name in _OPTIMIZERS:
        return "optimizer"
    if name in _CHAINS:
        return "chain"
    if name in _WEIGHTED:
        return "weighted"
    if name in _NO_POSTERIOR:
        return "none"
    return "unknown"


def normalize_chain_settings(chain) -> Optional[dict]:
    """Validate the ``chain`` argument of :meth:`MainPipeline.execute_all`.

    Parameters
    ----------
    chain : None, bool or dict
        ``None`` or ``False`` disable passing results between steps; ``True``
        uses the defaults. A dict may set ``estimator`` (``"map"``, ``"mean"``
        or ``"median"``), ``mode`` (``"fix"``, ``"start"`` or ``"gaussian"``),
        ``include`` (allow-list of sections, ``"section"``, or parameters,
        ``"section--name"``; if given, only those are passed, otherwise every
        parameter that is free in both runs), ``sigma_scale`` (factor applied
        to the posterior width of the Gaussian priors) and ``burn_in`` (see
        :func:`besta.postprocess.parse_burn_in`).

    Returns
    -------
    dict or None
        Settings with all keys, or ``None`` if disabled.
    """
    if chain is None or chain is False:
        return None
    if chain is True:
        chain = {}
    if not isinstance(chain, Mapping):
        raise TypeError("chain must be None, a bool or a dict.")
    unknown = set(chain) - set(DEFAULT_CHAIN_SETTINGS)
    if unknown:
        raise ValueError(f"Unknown chain settings: {sorted(unknown)}; "
                         f"expected a subset of {sorted(DEFAULT_CHAIN_SETTINGS)}.")
    out = {**DEFAULT_CHAIN_SETTINGS, **chain}
    out["estimator"] = str(out["estimator"]).lower()
    out["mode"] = str(out["mode"]).lower()
    if out["estimator"] not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, got {out['estimator']!r}.")
    if out["mode"] not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {out['mode']!r}.")
    if out["include"] is not None:
        include = [out["include"]] if isinstance(out["include"], str) else list(out["include"])
        out["include"] = [str(item).strip().lower() for item in include]
    out["sigma_scale"] = float(out["sigma_scale"])
    if not np.isfinite(out["sigma_scale"]) or out["sigma_scale"] <= 0:
        raise ValueError("sigma_scale must be finite and positive.")
    out["burn_in"] = postprocess.parse_burn_in(out["burn_in"])
    return out


# --------------------------------------------------------------------------- #
# Results of a finished step
# --------------------------------------------------------------------------- #
@dataclass
class StepResult:
    """Results table and configuration of a finished step."""

    table: Table
    ini: Mapping[str, Any]
    sampler: str
    kind: str

    @classmethod
    def from_reader(cls, reader) -> "StepResult":
        """Build it from a :class:`besta.io.Reader` with the results loaded.

        Raises
        ------
        ValueError
            If the last sampler of the run does not draw from a posterior.
        """
        sampler = str(reader.samplers[-1]).strip().lower()
        kind = sampler_kind(sampler)
        if kind == "none":
            raise ValueError(
                f"Cannot pass the results of sampler '{sampler}' to the next step: "
                "it does not draw from a posterior. Use maxlike, an MCMC sampler "
                "(emcee, zeus, ...) or a nested sampler (nautilus, dynesty, ...).")
        if kind == "unknown":
            logger.warning("Unknown sampler '%s': treating its results as an "
                           "unweighted chain unless it has a weight column.", sampler)
        return cls(reader.results_table, reader.ini, sampler, kind)


def _weights(table: Table) -> Optional[np.ndarray]:
    """Sample weights from the ``log_weight`` or ``weight`` column (``None`` if absent)."""
    if "log_weight" in table.colnames:
        log_w = np.asarray(table["log_weight"], dtype=float)
        return np.exp(log_w - np.nanmax(log_w))
    if "weight" in table.colnames:
        return np.asarray(table["weight"], dtype=float)
    return None


def point_estimates(result: StepResult, columns: List[str], estimator: str = "map",
                    burn_in="auto") -> Tuple[Dict[str, float], Dict[str, float], dict]:
    """Point estimate and posterior width of columns of a results table.

    What is computed depends on the sampler of the run:

    - optimizers (``maxlike``, ...): the best fit (highest ``post``). Mean and
      median are not defined, so the best fit is used with a warning; the
      width is NaN.
    - MCMC samplers: the burn-in is discarded (``burn_in``, automatic by
      default for samplers with walkers) and the samples are equally weighted.
    - nested samplers: samples are weighted with ``log_weight`` or ``weight``.

    Parameters
    ----------
    result : StepResult
    columns : list of str
        Table columns (``"section--name"``, lower case) to summarise.
    estimator : {"map", "mean", "median"}
    burn_in : float or "auto"

    Returns
    -------
    estimates, widths : dict
        Value and weighted standard deviation of each column.
    info : dict
        Sampler, number of samples used and the estimator actually applied.
    """
    table, kind = result.table, result.kind
    used = estimator
    if "post" not in table.colnames:
        raise ValueError("The results table has no 'post' column.")

    weights = None
    n_burn = 0
    if kind == "optimizer":
        if estimator != "map":
            logger.warning("Sampler '%s' gives no posterior: using its best fit "
                           "instead of the %s.", result.sampler, estimator)
            used = "map"
    elif kind == "chain" or (kind == "unknown" and _weights(table) is None):
        walkers = postprocess._sampler_walkers(result.ini)
        n_burn = postprocess._burn_in_steps(
            table, [c for c in columns if c in table.colnames], burn_in,
            walkers or 1, has_walkers=walkers is not None)
        if n_burn > 0:
            table = io.burn_table(table, nwalkers=walkers or 1, burn_in=n_burn)
        elif burn_in == "auto" and walkers is None:
            logger.warning("Sampler '%s' has no walkers: no burn-in removed. Set "
                           "'burn_in' in the chain settings to discard samples.",
                           result.sampler)
    else:
        weights = _weights(table)
        if weights is None:
            logger.warning("No weight column in the results of '%s': using "
                           "equal weights.", result.sampler)

    post = np.asarray(table["post"], dtype=float)
    finite = np.isfinite(post)
    if not finite.any():
        raise ValueError("No sample with finite posterior in the results table.")
    w = np.ones(len(table)) if weights is None else np.where(np.isfinite(weights), weights, 0.0)
    w = np.where(finite, w, 0.0)
    if w.sum() <= 0:
        raise ValueError("The sample weights of the results table are all zero.")
    best = int(np.nanargmax(np.where(finite & (w > 0), post, -np.inf)))

    estimates, widths = {}, {}
    for col in columns:
        x = np.asarray(table[col], dtype=float)
        ok = np.isfinite(x) & (w > 0)
        if not ok.any():
            continue
        mean = np.average(x[ok], weights=w[ok])
        sd = np.sqrt(np.average((x[ok] - mean) ** 2, weights=w[ok]))
        widths[col] = float(sd) if kind != "optimizer" else np.nan
        if used == "map":
            estimates[col] = float(x[best])
        elif used == "mean":
            estimates[col] = float(mean)
        else:
            estimates[col] = float(postprocess.weighted_quantile(x[ok], w[ok], [0.5])[0])
    info = {"sampler": result.sampler, "kind": kind, "estimator": used,
            "n_samples": int(np.count_nonzero(w > 0)), "burn_in_steps": int(n_burn)}
    return estimates, widths, info


# --------------------------------------------------------------------------- #
# Text-level editing of ini, values and priors files (keeps comments)
# --------------------------------------------------------------------------- #
_SECTION = re.compile(r"^\s*\[([^\]]+)\]\s*$")
_OPTION = re.compile(r"^(\s*)([^=;#\s][^=]*?)(\s*=\s*)(.*?)\s*$")


def _iter_options(text: str):
    """Yield ``(line_index, section, name, value)`` for every option of an ini text."""
    section = ""
    for i, line in enumerate(text.splitlines()):
        m = _SECTION.match(line)
        if m:
            section = m.group(1).strip()
            continue
        m = _OPTION.match(line)
        if m and not line.lstrip().startswith((";", "#")):
            yield i, section, m.group(2).strip(), m.group(4)


def _read_option(text: str, section: str, name: str) -> Optional[str]:
    for _, sec, key, value in _iter_options(text):
        if sec.lower() == section.lower() and key.lower() == name.lower():
            return value
    return None


def _set_option(text: str, section: str, name: str, value: str) -> str:
    """Replace an option of an ini text, or add it at the start of its section."""
    lines = text.splitlines()
    for i, sec, key, _ in _iter_options(text):
        if sec.lower() == section.lower() and key.lower() == name.lower():
            m = _OPTION.match(lines[i])
            lines[i] = f"{m.group(1)}{m.group(2)}{m.group(3)}{value}"
            return "\n".join(lines) + "\n"
    for i, line in enumerate(lines):
        m = _SECTION.match(line)
        if m and m.group(1).strip().lower() == section.lower():
            lines.insert(i + 1, f"{name} = {value}")
            return "\n".join(lines) + "\n"
    return "\n".join(lines + [f"[{section}]", f"{name} = {value}"]) + "\n"


def _remove_options(text: str, keys: set) -> str:
    """Remove the options ``(section, name)`` (lower case) from an ini text."""
    drop = {i for i, sec, key, _ in _iter_options(text)
            if (sec.lower(), key.lower()) in keys}
    return "\n".join(line for i, line in enumerate(text.splitlines()) if i not in drop) + "\n"


def _parse_values(text: str) -> Dict[Tuple[str, str], Tuple[str, ...]]:
    """Tokens of the value of each option of a values file, keyed by lower-case (section, name)."""
    out = {}
    for _, sec, key, value in _iter_options(text):
        out[(sec.lower(), key.lower())] = tuple(value.split(";")[0].split())
    return out


def _path(value: str, base: str) -> str:
    path = os.path.expandvars(value.strip().strip("\"'"))
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(base, path)
    return path


def _sfh_signature(ini: Mapping[str, Any]) -> Dict[str, str]:
    """Options that define the SFH parametrisation of the first SFH module of a run."""
    modules = postprocess.find_sfh_modules(ini)
    if not modules:
        return {}
    options = ini[modules[0]]
    return {str(k).lower(): str(v) for k, v in options.items()
            if str(k).lower().startswith("sfh") or str(k).lower() in {
                "use_transforms", "latent_space", "max_dlogsfr", "min_time",
                "min_last_interval", "sect_name"}}


def _sfh_sections(ini: Mapping[str, Any]) -> set:
    modules = postprocess.find_sfh_modules(ini)
    if not modules:
        return set()
    try:
        return {postprocess.build_sfh_model(ini, modules[0]).sect_name.lower()}
    except Exception:                          # pragma: no cover - defensive
        return {"stars.sfh"}


# --------------------------------------------------------------------------- #
# Preparation of the next step
# --------------------------------------------------------------------------- #
def prepare_step(ini_file: str, result: StepResult, settings: dict, step: int,
                 values_file: Optional[str] = None) -> str:
    """Write the ini (and values/priors) files of the next step.

    Parameters
    ----------
    ini_file : str
        Ini file of the next step.
    result : StepResult
        Results of the previous step.
    settings : dict
        Output of :func:`normalize_chain_settings`.
    step : int
        Index of the next step (used in the names of the new files).
    values_file : str, optional
        Values file to use instead of the ``[pipeline] values`` of the ini.

    Returns
    -------
    str
        Path of the ini file to run. It is ``ini_file`` itself if nothing could
        be passed.
    """
    ini_text = open(ini_file, encoding="utf-8").read()
    base = os.path.dirname(os.path.abspath(ini_file))
    values_path = values_file or _read_option(ini_text, "pipeline", "values")
    if values_path is None:
        logger.warning("Step %d has no values file: nothing to pass.", step)
        return ini_file
    values_path = _path(values_path, base)
    values_text = open(values_path, encoding="utf-8").read()
    next_ini = io.Reader.read_ini_file(ini_file)

    # Parameters that are free in the next run and available in the results
    include = settings["include"]
    candidates, skipped = {}, {}
    for (sec, name), tokens in _parse_values(values_text).items():
        key = f"{sec}--{name}"
        if len(tokens) != 3:
            continue                                   # fixed (or not a parameter)
        if include is not None and sec not in include and key not in include:
            continue
        if key not in result.table.colnames:
            skipped[key] = "not in the results of the previous step"
            continue
        candidates[key] = tokens
    for entry in include or []:
        if not any(k == entry or k.startswith(entry + "--") for k in candidates) \
                and not any(k == entry or k.startswith(entry + "--") for k in skipped):
            logger.warning("Chain 'include' entry '%s' matches no free parameter "
                           "of step %d.", entry, step)
    if not candidates:
        logger.warning("No parameter to pass to step %d.", step)
        return ini_file

    # The meaning of the SFH parameters depends on the SFH configuration
    sfh_sections = _sfh_sections(next_ini)
    if any(k.split("--")[0] in sfh_sections for k in candidates):
        if _sfh_signature(result.ini) != _sfh_signature(next_ini):
            raise ValueError(
                "The SFH configuration of step {} differs from the previous one, so "
                "their SFH parameters do not mean the same. Restrict the chain "
                "'include' list to other sections.".format(step))

    estimates, widths, info = point_estimates(
        result, list(candidates), settings["estimator"], settings["burn_in"])
    mode = settings["mode"]
    if mode == "gaussian" and result.kind == "optimizer":
        raise ValueError(f"mode='gaussian' needs a posterior width, but sampler "
                         f"'{result.sampler}' only returns the best fit.")

    out_dir = os.path.dirname(os.path.abspath(
        _path(str(next_ini.get("output", {}).get("filename", ini_file)), base)))
    stem = f"chain_step{step}"
    report, edits, gaussians = {}, {}, {}
    for key, tokens in candidates.items():
        if key not in estimates:
            skipped[key] = "no finite samples"
            continue
        lo, start, hi = tokens
        value, clamped = estimates[key], False
        if not float(lo) <= value <= float(hi):
            value, clamped = float(np.clip(value, float(lo), float(hi))), True
            logger.warning("%s: estimate %.6g outside [%s, %s]: using %.6g.",
                           key, estimates[key], lo, hi, value)
        action = mode
        if mode == "gaussian":
            sigma = widths[key] * settings["sigma_scale"]
            if not np.isfinite(sigma) or sigma <= 0:
                logger.warning("%s: no posterior width: starting at the estimate "
                               "without a Gaussian prior.", key)
                action = "start"
            else:
                gaussians[key] = (value, sigma)
        edits[key] = f"{value:.12g}" if mode == "fix" else f"{lo} {value:.12g} {hi}"
        report[key] = {"value": value, "width": widths.get(key), "range": [lo, hi],
                       "action": action, "clamped": clamped}

    # Values file
    new_values = values_text
    for key, rhs in edits.items():
        sec, name = key.split("--", 1)
        original = [(s, k) for _, s, k, _ in _iter_options(values_text)
                    if s.lower() == sec and k.lower() == name][0]
        new_values = _set_option(new_values, original[0], original[1], rhs)
    new_values_path = os.path.join(
        out_dir, f"{os.path.splitext(os.path.basename(values_path))[0]}_{stem}.ini")
    with open(new_values_path, "w", encoding="utf-8") as f:
        f.write(new_values)
    new_ini = _set_option(ini_text, "pipeline", "values", new_values_path)

    # Priors files: a parameter may only be defined once
    if gaussians:
        keys = {tuple(k.split("--", 1)) for k in gaussians}
        prior_files = (_read_option(ini_text, "pipeline", "priors") or "").split()
        new_priors = []
        for i, pf in enumerate(prior_files):
            text = _remove_options(open(_path(pf, base), encoding="utf-8").read(), keys)
            path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(pf))[0]}_{stem}_{i}.ini")
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            new_priors.append(path)
        lines = ["; Gaussian priors from the previous step (besta.chain)"]
        for sec in sorted({k.split("--")[0] for k in gaussians}):
            lines.append(f"[{sec}]")
            lines += [f"{k.split('--', 1)[1]} = gaussian {mu:.12g} {sg:.12g}"
                      for k, (mu, sg) in gaussians.items() if k.split("--")[0] == sec]
        gauss_path = os.path.join(out_dir, f"priors_{stem}.ini")
        with open(gauss_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        new_priors.append(gauss_path)
        new_ini = _set_option(new_ini, "pipeline", "priors", " ".join(new_priors))

    new_ini_path = os.path.join(
        out_dir, f"{os.path.splitext(os.path.basename(ini_file))[0]}_{stem}.ini")
    with open(new_ini_path, "w", encoding="utf-8") as f:
        f.write(new_ini)
    with open(os.path.splitext(new_ini_path)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump({"step": step, "settings": settings, "previous": info,
                   "parameters": report, "skipped": skipped,
                   "ini": new_ini_path, "values": new_values_path}, f, indent=2,
                  default=str)
    logger.info("Step %d: %s of the previous step (%s, %d samples) passed as '%s' "
                "to %d parameters: %s", step, info["estimator"], info["sampler"],
                info["n_samples"], mode, len(report), ", ".join(report))
    return new_ini_path
