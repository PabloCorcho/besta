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

    def _make_step_method(self):
        """Build the configured PyMC step method."""
        name = self.step_method_name

        if name == "metropolis":
            return self.pm.Metropolis()
        if name == "demetropolis":
            return self.pm.DEMetropolis()
        if name == "demetropolisz":
            return self.pm.DEMetropolisZ()
        if name == "slice":
            return self.pm.Slice()

        if name in {"nuts", "hmc", "hamiltonianmc"}:
            raise ValueError(
                f"step_method={name!r} requires gradients, but this sampler wraps "
                "a black-box CosmoSIS likelihood and does not provide gradients. "
                "Use one of: metropolis, demetropolis, demetropolisz, slice."
            )

        raise ValueError(
            f"Unknown step_method={name!r}. "
            "Valid options are: metropolis, demetropolis, demetropolisz, slice."
        )

    def _posterior_at_unit_point(self, theta_unit: np.ndarray) -> float:
        """Evaluate the posterior used to validate a proposed chain start."""
        params = self.pipeline.denormalize_vector(theta_unit)
        return float(self.pipeline.run_results(params).post)

    def _build_initial_points(self) -> list[np.ndarray]:
        """Build valid per-chain starts using the standard CosmoSIS policy.

        ``Sampler.start_estimate`` prefers a distribution hint from an earlier
        sampler (for example MaxLike), then honors ``start_method`` and
        ``start_input``, and finally falls back to the values-file start.
        """
        start = np.asarray(self.start_estimate(), dtype=float)
        if start.shape != (self.ndim,):
            raise ValueError(
                "The PyMC starting point has shape "
                f"{start.shape}; expected ({self.ndim},)."
            )

        center = np.asarray(self.pipeline.normalize_vector(start), dtype=float)
        if not np.all(np.isfinite(center)):
            raise ValueError("The PyMC starting point contains non-finite values.")
        if np.any(center < 0.0) or np.any(center > 1.0):
            raise ValueError("The PyMC starting point lies outside the parameter bounds.")

        # PyMC transforms bounded variables internally, so exact boundary
        # values would map to infinite unconstrained coordinates.
        center = np.clip(center, self.start_edge_buffer, 1.0 - self.start_edge_buffer)
        center_post = self._posterior_at_unit_point(center)
        if not np.isfinite(center_post) or center_post <= self.start_min_posterior:
            raise ValueError(
                "The selected PyMC starting point is invalid: "
                f"posterior={center_post:.6g}. Run an optimizer first or select "
                "another [pymc] start_method/start_input."
            )

        points = [center]
        if self.chains == 1:
            return points

        if self.start_jitter == 0.0:
            if self.step_method_name in {"demetropolis", "demetropolisz"}:
                raise ValueError(
                    "DEMetropolis methods require dispersed chain starts; set "
                    "[pymc] start_jitter to a positive value."
                )
            return [center.copy() for _ in range(self.chains)]

        rng = np.random.default_rng(self.random_seed)
        for chain_index in range(1, self.chains):
            for _ in range(self.start_attempts):
                candidate = center + rng.normal(
                    loc=0.0,
                    scale=self.start_jitter,
                    size=self.ndim,
                )
                candidate = np.clip(
                    candidate,
                    self.start_edge_buffer,
                    1.0 - self.start_edge_buffer,
                )
                candidate_post = self._posterior_at_unit_point(candidate)
                if (
                    np.isfinite(candidate_post)
                    and candidate_post > self.start_min_posterior
                ):
                    points.append(candidate)
                    break
            else:
                raise ValueError(
                    "Could not generate a valid initial point for PyMC chain "
                    f"{chain_index + 1} after {self.start_attempts} attempts. "
                    "Reduce [pymc] start_jitter or provide a better start."
                )

        return points

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
        self.step_method_name = self.read_ini("step_method", str, "demetropolisz").strip().lower()

        self.start_jitter = self.read_ini("start_jitter", float, 1.0e-3)
        self.start_attempts = self.read_ini("start_attempts", int, 1000)
        self.start_edge_buffer = self.read_ini("start_edge_buffer", float, 1.0e-6)
        self.start_min_posterior = self.read_ini(
            "start_min_posterior", float, -1.0e19
        )
        if self.start_jitter < 0.0:
            raise ValueError("[pymc] start_jitter must be non-negative.")
        if self.start_attempts < 1:
            raise ValueError("[pymc] start_attempts must be at least one.")
        if not 0.0 < self.start_edge_buffer < 0.5:
            raise ValueError("[pymc] start_edge_buffer must lie between 0 and 0.5.")

        fburn = self.read_ini("burn_fraction", float, 0.0)
        if 0.0 <= fburn < 1.0:
            self.nburn = int(fburn * self.samples)
        else:
            self.nburn = max(0, int(fburn))

        seed_raw = self.read_ini("seed", int, -1)
        self.random_seed = None if seed_raw < 0 else seed_raw

        self.initial_points_unit = self._build_initial_points()
        initial_values = [
            {"theta_unit": point.copy()} for point in self.initial_points_unit
        ]

        logs.overview(
            "Initialized PyMC chains from the CosmoSIS starting point "
            f"with unit-cube jitter={self.start_jitter:.6g}"
        )

        self._logpost_op = _build_logpost_op(self.pipeline)

        with self.pm.Model() as model:
            theta_unit = self.pm.Uniform(
                "theta_unit",
                lower=0.0,
                upper=1.0,
                shape=(self.ndim,),
                initval=self.initial_points_unit[0],
            )
            self.pm.Potential("cosmosis_logpost", self._logpost_op(theta_unit))
        self._model = model
        self._initial_values = initial_values

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
                step=self._make_step_method(),
                initvals=self._initial_values,
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
