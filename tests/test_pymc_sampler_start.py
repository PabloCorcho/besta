from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pymc")

from cosmosis.output import InMemoryOutput

from besta.samplers.pymc_sampler import PymcSampler


class DummyPipeline:
    def __init__(self, start=(-0.5, 0.5)):
        self.varied_params = [object(), object()]
        self.modules = []
        self._start = np.asarray(start, dtype=float)

    def output_names(self):
        return ["p0", "p1"]

    def start_vector(self):
        return self._start.copy()

    def normalize_vector(self, p):
        return (np.asarray(p, dtype=float) + 2.0) / 4.0

    def denormalize_vector(self, u):
        return 4.0 * np.asarray(u, dtype=float) - 2.0

    def run_results(self, p):
        p = np.asarray(p, dtype=float)
        post = -2.0e20 if p[0] >= p[1] else -0.5 * float(np.dot(p, p))
        return SimpleNamespace(
            post=post,
            prior=0.0,
            like=post,
            extra=np.empty(0),
        )


def _write_ini(path: Path, chains: int = 4) -> Path:
    path.write_text(
        (
            "[runtime]\n"
            "sampler = pymc\n\n"
            "[pymc]\n"
            "samples = 1\n"
            f"chains = {chains}\n"
            "step_method = metropolis\n"
            "progressbar = F\n"
            "seed = 13\n"
            "start_jitter = 0.05\n"
        ),
        encoding="utf-8",
    )
    return path


def test_start_uses_previous_sampler_peak_and_builds_valid_chains(tmp_path):
    ini = _write_ini(tmp_path / "pymc.ini")
    pipeline = DummyPipeline()
    sampler = PymcSampler(str(ini), pipeline, InMemoryOutput())
    peak = np.array([-1.0, 1.0])
    sampler.distribution_hints.set_peak(peak, -1.0)

    sampler.config()

    expected_center = pipeline.normalize_vector(peak)
    assert len(sampler.initial_points_unit) == 4
    np.testing.assert_allclose(sampler.initial_points_unit[0], expected_center)
    assert any(
        not np.allclose(point, expected_center)
        for point in sampler.initial_points_unit[1:]
    )
    for point in sampler.initial_points_unit:
        assert np.all(point > 0.0)
        assert np.all(point < 1.0)
        assert pipeline.run_results(pipeline.denormalize_vector(point)).post > -1.0e19


def test_start_falls_back_to_values_file_point(tmp_path):
    ini = _write_ini(tmp_path / "pymc.ini", chains=1)
    pipeline = DummyPipeline(start=(-0.25, 0.75))
    sampler = PymcSampler(str(ini), pipeline, InMemoryOutput())

    sampler.config()

    np.testing.assert_allclose(
        sampler.initial_points_unit[0],
        pipeline.normalize_vector(pipeline.start_vector()),
    )


def test_invalid_start_is_rejected_with_actionable_error(tmp_path):
    ini = _write_ini(tmp_path / "pymc.ini", chains=1)
    sampler = PymcSampler(
        str(ini),
        DummyPipeline(start=(1.0, -1.0)),
        InMemoryOutput(),
    )

    with pytest.raises(ValueError, match="starting point is invalid"):
        sampler.config()
