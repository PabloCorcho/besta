from __future__ import annotations

from time import perf_counter

import numpy as np
import pytest

from besta.pipeline_modules import likelihoods
from besta.pipeline_modules.base_module import BaseModule


class _DummySpectraModule(BaseModule):
    name = "DummySpectra"

    def __init__(self, options):
        super().__init__(options, likelihood_kind="spectra")

    def make_observable(self, *args, **kwargs):
        raise NotImplementedError

    def execute(self, *args, **kwargs):
        raise NotImplementedError

    def plot_solution(self, *args, **kwargs):
        raise NotImplementedError


class _DummyPhotometryModule(BaseModule):
    name = "DummyPhotometry"

    def __init__(self, options):
        super().__init__(options, likelihood_kind="photometry")

    def make_observable(self, *args, **kwargs):
        raise NotImplementedError

    def execute(self, *args, **kwargs):
        raise NotImplementedError

    def plot_solution(self, *args, **kwargs):
        raise NotImplementedError


class _DummyObservableSpectraModule(BaseModule):
    name = "DummyObservableSpectra"

    def __init__(self, options):
        super().__init__(options, likelihood_kind="spectra")

    def make_observable(self, flux):
        return flux, np.ones_like(flux)

    def execute(self, *args, **kwargs):
        raise NotImplementedError

    def plot_solution(self, *args, **kwargs):
        raise NotImplementedError


class _DummyObservablePhotometryModule(BaseModule):
    name = "DummyObservablePhotometry"

    def __init__(self, options):
        super().__init__(options, likelihood_kind="photometry")

    def make_observable(self, flux):
        return flux, np.ones_like(flux)

    def execute(self, *args, **kwargs):
        raise NotImplementedError

    def plot_solution(self, *args, **kwargs):
        raise NotImplementedError


@pytest.mark.skipif(not likelihoods.NUMBA_AVAILABLE, reason="numba is not available")
def test_likelihood_method_selection_from_init():
    spectra_mod = _DummySpectraModule({"DummySpectra": {"likelihood_method": "numba"}})
    photometry_mod = _DummyPhotometryModule({"DummyPhotometry": {"likelihood_method": "numba"}})

    assert spectra_mod.likelihood_kind == "spectra"
    assert photometry_mod.likelihood_kind == "photometry"
    assert spectra_mod.likelihood_method == "numba"
    assert photometry_mod.likelihood_method == "numba"
    assert spectra_mod.log_like is likelihoods.spectra_loglike_numba
    assert photometry_mod.log_like is not None


def _benchmark(func, *args, repeats: int = 200):
    start = perf_counter()
    value = None
    for _ in range(repeats):
        value = func(*args)
    return perf_counter() - start, value


@pytest.mark.skipif(not likelihoods.NUMBA_AVAILABLE, reason="numba is not available")
def test_likelihood_backends_match_and_report_performance(capsys):
    rng = np.random.default_rng(123)

    # Spectra: inverse-variance Gaussian with no limits.
    spectra_data = rng.normal(size=20000)
    spectra_model = spectra_data + rng.normal(scale=0.05, size=20000)
    spectra_ivar = np.full(20000, 25.0)

    spectra_numpy = likelihoods.spectra_loglike_numpy
    spectra_numba = likelihoods.make_spectra_loglike("numba")
    spectra_numba(spectra_data, spectra_model, spectra_ivar)  # warmup compile
    t_np, v_np = _benchmark(spectra_numpy, spectra_data, spectra_model, spectra_ivar)
    t_nb, v_nb = _benchmark(spectra_numba, spectra_data, spectra_model, spectra_ivar)
    assert np.isclose(v_np, v_nb)

    # Photometry: run both the no-limits fast path and the limits-aware path.
    photo_data = rng.normal(size=2000)
    photo_model = photo_data + rng.normal(scale=0.05, size=2000)
    photo_var = np.full(2000, 0.04)
    photo_weights = rng.uniform(0.2, 1.0, size=2000)
    no_limits = (None, None)
    upper = np.zeros(2000, dtype=bool)
    upper[::73] = True
    lower = np.zeros(2000, dtype=bool)
    lower[::89] = True
    lower[::73] = False

    photo_numpy = likelihoods.photometry_loglike_numpy
    photo_numba = likelihoods.make_photometry_loglike("numba")
    photo_numba(photo_data, photo_model, photo_var, photo_weights, *no_limits)  # warmup compile
    t_pn, v_pn = _benchmark(photo_numpy, photo_data, photo_model, photo_var, photo_weights, *no_limits)
    t_pj, v_pj = _benchmark(photo_numba, photo_data, photo_model, photo_var, photo_weights, *no_limits)
    assert np.isclose(v_pn, v_pj)

    t_pl, v_pl = _benchmark(
        photo_numpy,
        photo_data,
        photo_model,
        photo_var,
        photo_weights,
        upper,
        lower,
    )
    t_pjl, v_pjl = _benchmark(
        photo_numba,
        photo_data,
        photo_model,
        photo_var,
        photo_weights,
        upper,
        lower,
    )
    assert np.isclose(v_pl, v_pjl)

    print(
        "Likelihood timing report (seconds over repeated calls):\n"
        f"  spectra  numpy={t_np:.6f}  numba={t_nb:.6f}  speedup={t_np / t_nb:.2f}x\n"
        f"  photo(no limits) numpy={t_pn:.6f}  numba={t_pj:.6f}  speedup={t_pn / t_pj:.2f}x\n"
        f"  photo(limits)    numpy={t_pl:.6f}  numba={t_pjl:.6f}  speedup={t_pl / t_pjl:.2f}x"
    )

    captured = capsys.readouterr()
    assert "Likelihood timing report" in captured.out


@pytest.mark.parametrize(
    ("module_cls", "section"),
    [
        (_DummyObservableSpectraModule, "DummyObservableSpectra"),
        (_DummyObservablePhotometryModule, "DummyObservablePhotometry"),
    ],
)
def test_save_observables_stores_first_tuple_element(module_cls, section):
    module = module_cls({section: {"save_observables": True}})
    flux = np.array([1.0, 2.0, 3.0])

    returned = module.make_observable(flux)
    assert isinstance(returned, tuple)
    assert len(module._observables_list) == 1

    stored = module._observables_list[0]
    assert isinstance(stored, np.ndarray)
    np.testing.assert_allclose(stored, np.array([1.0, 2.0, 3.0]))

    # Returned tuple values can be modified without mutating stored history.
    returned[0][0] = 99.0
    np.testing.assert_allclose(module._observables_list[0], np.array([1.0, 2.0, 3.0]))


def test_save_observables_disabled_keeps_empty_storage():
    module = _DummyObservableSpectraModule({"DummyObservableSpectra": {"save_observables": False}})
    _ = module.make_observable(np.array([1.0, 2.0]))

    assert module._observables_list == []
