from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cosmosis.output import InMemoryOutput

from besta import io
from besta.pipeline import MainPipeline
from besta.samplers.pymc_sampler import PymcSampler


def _write_ini(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_pymc_sampler_smoke_with_dummy_pipeline(tmp_path):
    class DummyPipeline:
        def __init__(self):
            self.varied_params = [object(), object()]
            self.modules = []

        def output_names(self):
            # Varied params + one extra output.
            return ["p0", "p1", "extra0"]

        def denormalize_vector(self, u):
            # Map unit-cube to a bounded physical space.
            return 4.0 * np.asarray(u, dtype=float) - 2.0

        def run_results(self, p):
            p = np.asarray(p, dtype=float)
            post = -0.5 * float(np.dot(p, p))
            prior = -0.1 * float(np.dot(p, p))
            like = post - prior
            extra = np.array([float(p.sum())], dtype=float)
            return SimpleNamespace(post=post, prior=prior, like=like, extra=extra)

    ini_path = _write_ini(
        tmp_path / "pymc_smoke.ini",
        """
[runtime]
sampler = pymc

[pymc]
samples = 24
nsteps = 8
burn_fraction = 0.0
chains = 1
progressbar = F
seed = 11
""".strip(),
    )

    output = InMemoryOutput()
    sampler = PymcSampler(str(ini_path), DummyPipeline(), output)
    sampler.config()
    sampler.execute()

    assert sampler.is_converged()
    assert sampler.num_samples > 0
    assert len(output.rows) == sampler.num_samples

    col_names = [c[0] for c in output.columns]
    assert "prior" in col_names
    assert "post" in col_names
    assert "like" in col_names


def test_pymc_sampler_accepts_slice_step_method(tmp_path):
    class DummyPipeline:
        def __init__(self):
            self.varied_params = [object(), object()]
            self.modules = []

        def output_names(self):
            return ["p0", "p1", "extra0"]

        def denormalize_vector(self, u):
            return 4.0 * np.asarray(u, dtype=float) - 2.0

        def run_results(self, p):
            p = np.asarray(p, dtype=float)
            post = -0.5 * float(np.dot(p, p))
            prior = -0.1 * float(np.dot(p, p))
            like = post - prior
            extra = np.array([float(p.sum())], dtype=float)
            return SimpleNamespace(post=post, prior=prior, like=like, extra=extra)

    ini_path = _write_ini(
        tmp_path / "pymc_slice.ini",
        """
[runtime]
sampler = pymc

[pymc]
samples = 12
nsteps = 4
burn_fraction = 0.0
chains = 1
progressbar = F
seed = 5
step_method = slice
""".strip(),
    )

    output = InMemoryOutput()
    sampler = PymcSampler(str(ini_path), DummyPipeline(), output)
    sampler.config()
    sampler.execute()

    assert sampler.is_converged()
    assert sampler.num_samples > 0


def test_pymc_sampler_rejects_gradient_step_methods(tmp_path):
    class DummyPipeline:
        def __init__(self):
            self.varied_params = [object(), object()]
            self.modules = []

        def output_names(self):
            return ["p0", "p1", "extra0"]

        def denormalize_vector(self, u):
            return 4.0 * np.asarray(u, dtype=float) - 2.0

        def run_results(self, p):
            p = np.asarray(p, dtype=float)
            post = -0.5 * float(np.dot(p, p))
            prior = -0.1 * float(np.dot(p, p))
            like = post - prior
            extra = np.array([float(p.sum())], dtype=float)
            return SimpleNamespace(post=post, prior=prior, like=like, extra=extra)

    ini_path = _write_ini(
        tmp_path / "pymc_nuts.ini",
        """
[runtime]
sampler = pymc

[pymc]
samples = 6
nsteps = 3
burn_fraction = 0.0
chains = 1
progressbar = F
seed = 1
step_method = nuts
""".strip(),
    )

    output = InMemoryOutput()
    sampler = PymcSampler(str(ini_path), DummyPipeline(), output)
    sampler.config()
    with pytest.raises(ValueError, match="requires gradients"):
        sampler.execute()


def test_pymc_sampler_full_besta_run(tmp_path):
    cosmosis_exe = shutil.which("cosmosis")
    if cosmosis_exe is None:
        pytest.skip("cosmosis executable not available in PATH")

    module_path = tmp_path / "dummy_like_module.py"
    module_path.write_text(
        """
def setup(options):
    return {}


def execute(block, config):
    p1 = block['parameters', 'p1']
    p2 = block['parameters', 'p2']
    like = -0.5 * (p1**2 + p2**2)
    block['extra', 'sum_p'] = p1 + p2
    block['likelihoods', 'dummy_like'] = like
    return 0


def cleanup(config):
    return 0
""".strip(),
        encoding="utf-8",
    )

    output_root = tmp_path / "pymc_full_run"
    values_path = tmp_path / "values.ini"
    values_path.write_text(
        """
[parameters]
p1 = -2.0 0.0 2.0
p2 = -2.0 0.0 2.0
""".strip(),
        encoding="utf-8",
    )

    sampler_path = Path(__file__).resolve().parents[1] / "src" / "besta" / "samplers" / "pymc_sampler.py"

    config = {
        "runtime": {
            "sampler": "pymc",
            "import_samplers": str(sampler_path),
        },
        "pymc": {
            "samples": 16,
            "nsteps": 8,
            "burn_fraction": 0.0,
            "chains": 1,
            "progressbar": False,
            "seed": 3,
        },
        "output": {
            "filename": str(output_root),
            "format": "text",
        },
        "pipeline": {
            "modules": "DummyLike",
            "values": str(values_path),
            "likelihoods": "dummy",
            "quiet": "T",
            "debug": "T",
            "extra_output": "extra/sum_p",
        },
        "DummyLike": {
            "file": str(module_path),
        },
    }

    runner = MainPipeline(
        [config],
        n_cores_list=[1],
        ini_values_files=[str(values_path)],
    )
    status = runner.execute_all(plot_result=False)
    assert status == 0

    results_file = Path(str(output_root) + ".txt")
    assert results_file.exists()

    table = io.read_results_file(str(results_file))
    assert len(table) > 0
    assert "post" in table.colnames
    assert "prior" in table.colnames
    assert "like" in table.colnames
