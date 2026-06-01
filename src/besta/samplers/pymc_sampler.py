"""PyMC sampler compatible with modern PyMC versions.

This module is designed to be loaded by CosmoSIS using the runtime option
`import_samplers = /path/to/pymc_sampler.py`.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from cosmosis.runtime import logs
from cosmosis.samplers import ParallelSampler


def _build_logpost_op(pipeline):
    """Create a PyTensor Op that evaluates the CosmoSIS log-posterior."""
    try:
        from pytensor import wrap_py as _wrap_py
    except ImportError:  # pragma: no cover - compatibility with older PyTensor
        from pytensor.compile.ops import as_op as _wrap_py
    import pytensor.tensor as pt

    @_wrap_py(itypes=[pt.dvector], otypes=[pt.dscalar])
    def logpost_from_unit_cube(theta_unit):
        theta_unit = np.asarray(theta_unit, dtype=float)
        if theta_unit.ndim != 1:
            return np.array(-np.inf, dtype=np.float64)
        if not np.all(np.isfinite(theta_unit)):
            return np.array(-np.inf, dtype=np.float64)

        try:
            params = pipeline.denormalize_vector(theta_unit)
        except Exception:
            return np.array(-np.inf, dtype=np.float64)

        result = pipeline.run_results(params)
        post = float(result.post)
        if not np.isfinite(post):
            return np.array(-np.inf, dtype=np.float64)
        return np.array(post, dtype=np.float64)

    return logpost_from_unit_cube


class PymcSampler(ParallelSampler):
    """Modern PyMC sampler for CosmoSIS pipelines.

    Notes
    -----
    - Uses a Metropolis step method to avoid requiring gradients through the
      black-box pipeline likelihood.
    - Samples in normalized unit-cube coordinates and denormalizes via
      `pipeline.denormalize_vector` when evaluating the posterior.
    """

    sampler_outputs = [("prior", float), ("post", float), ("like", float)]
    parallel_output = False
    supports_smp = False

    def config(self):
        try:
            import pymc as pm
        except ImportError as exc:
            raise ImportError(
                "PyMC is required for besta.samplers.PymcSampler. "
                "Install it in your active environment (for example: pip install pymc)."
            ) from exc

        self.pm = pm
        self._model = None
        self._done = False

        self.ndim = len(self.pipeline.varied_params)
        self.samples = max(1, self.read_ini("samples", int, 1000))
        self.nsteps = max(1, self.read_ini("nsteps", int, self.samples))
        self.chains = max(1, self.read_ini("chains", int, 1))
        self.target_accept = float(self.read_ini("target_accept", float, 0.8))
        self.progressbar = bool(self.read_ini("progressbar", bool, False))

        fburn = self.read_ini("burn_fraction", float, 0.0)
        if 0.0 <= fburn < 1.0:
            self.nburn = int(fburn * self.samples)
        else:
            self.nburn = max(0, int(fburn))

        seed_raw = self.read_ini("seed", int, -1)
        self.random_seed = None if seed_raw < 0 else seed_raw

        self._logpost_op = _build_logpost_op(self.pipeline)

        with self.pm.Model() as model:
            theta_unit = self.pm.Uniform(
                "theta_unit", lower=0.0, upper=1.0, shape=(self.ndim,)
            )
            self.pm.Potential("cosmosis_logpost", self._logpost_op(theta_unit))
        self._model = model

    def _evaluate_samples(self, theta_unit: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate pipeline diagnostics for sampled points and stream to output."""
        traces = []
        posts = []

        for u in theta_unit:
            try:
                params = np.asarray(self.pipeline.denormalize_vector(u), dtype=float)
            except Exception:
                continue

            result = self.pipeline.run_results(params)
            prior = float(result.prior)
            post = float(result.post)
            like = float(result.like)

            if not np.isfinite(post):
                continue

            extra = np.asarray(result.extra)
            self.output.parameters(params, extra, prior, post, like)

            traces.append(params)
            posts.append(post)

        if traces:
            traces_arr = np.asarray(traces, dtype=float)
            posts_arr = np.asarray(posts, dtype=float)
            self.distribution_hints.set_from_sample(traces_arr, posts_arr)
            return traces_arr, posts_arr

        return np.empty((0, self.ndim), dtype=float), np.empty((0,), dtype=float)

    def execute(self):
        if self._done:
            return

        draws = max(1, self.samples)
        tune = max(0, self.nburn)

        logs.overview(
            f"Running PyMC with draws={draws}, tune={tune}, chains={self.chains}, "
            f"nsteps={self.nsteps}"
        )

        with self._model:
            idata = self.pm.sample(
                draws=draws,
                tune=tune,
                chains=self.chains,
                cores=1,
                random_seed=self.random_seed,
                progressbar=self.progressbar,
                compute_convergence_checks=False,
                discard_tuned_samples=True,
                return_inferencedata=True,
                step=self.pm.Metropolis(),
            )

        theta = np.asarray(idata.posterior["theta_unit"])
        theta = theta.reshape((-1, self.ndim))

        traces, posts = self._evaluate_samples(theta)
        self.num_samples = traces.shape[0]
        self._done = True

        if posts.size:
            logs.overview(
                f"Done PyMC sampling with {self.num_samples} samples. "
                f"max(post)={posts.max():.6g}"
            )
        else:
            logs.overview("Done PyMC sampling but no finite-posterior samples were recorded.")

    def worker(self):
        while not self.is_converged():
            self.execute()

    def is_converged(self):
        return self._done


# legacy alias
PyMCSampler = PymcSampler
