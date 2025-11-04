# tests/test_grid_module.py
import os
import json
import numpy as np
import pytest

# Import the module under test
# Adjust if your module path differs
from besta.grid import (
    ModelGrid,
    HierarchicalGridBinner,
    GridFitter,
)

# ---------------------------
# Small deterministic helpers
# ---------------------------

class DummyLikelihood:
    """
    Simple Gaussian product likelihood with per-dimension sigmas.
    logL_i = -0.5 * sum_j ((x_j - X_ij)/sigma_j)^2
    """
    def log_likelihood(self, x_eval, sigma_eval, X_eval):
        sigma_eval = np.asarray(sigma_eval)
        sigma_eval = np.maximum(sigma_eval, 1e-12)
        diff = (X_eval - x_eval[None, :]) / sigma_eval[None, :]
        return -0.5 * np.sum(diff**2, axis=1)

class DummyFlatPrior:
    """log P(model) = 0 everywhere."""
    def log_prob_for_models(self, targets, observables=None):
        return np.zeros(targets.shape[0], dtype=float)


@pytest.fixture
def tiny_grid():
    # 6 models, 2 observables, 2 targets
    # Arrange observables on a line to make nearest-neighbour checks easy
    X = np.array([
        [-2.0, -2.0],
        [-1.0, -1.0],
        [-0.5, -0.5],
        [ 0.0,  0.0],
        [ 0.5,  0.5],
        [ 1.0,  1.0],
    ], dtype=float)
    # Make targets simple functions of observables for easy expectations
    T = np.column_stack([
        X[:, 0] + X[:, 1],         # t0 = sum
        X[:, 0] * X[:, 1],         # t1 = product
    ])
    names_obs = ["o0", "o1"]
    names_tgt = ["sum", "prod"]
    w = np.ones(X.shape[0], dtype=float)
    g = ModelGrid(observables=X, targets=T,
                  observable_names=names_obs,
                  target_names=names_tgt,
                  weights=w, meta={"unit_o0": "mag"})
    return g

# ---------------------------
# ModelGrid tests
# ---------------------------

def test_modelgrid_basic_properties(tiny_grid):
    g = tiny_grid
    assert g.n_models == 6
    assert g.n_observables == 2
    assert g.n_targets == 2
    assert g.observable_names == ["o0", "o1"]
    assert g.target_names == ["sum", "prod"]
    # select
    sub = g.select(np.array([1, 3, 5]))
    assert sub.n_models == 3
    assert sub.observables.shape == (3, 2)
    assert sub.targets.shape == (3, 2)

def test_modelgrid_standardiser_roundtrip(tiny_grid):
    g = tiny_grid
    g.fit_standardiser()
    X = g.transform_observables(g.observables)
    # mean ~0, std ~1 for each column
    np.testing.assert_allclose(np.nanmean(X, axis=0), [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(np.nanstd(X, axis=0, ddof=0), [1.0, 1.0], atol=1e-12)

def test_modelgrid_dict_roundtrip(tiny_grid):
    g = tiny_grid
    d = g.to_dict()
    g2 = ModelGrid.from_dict(d)
    np.testing.assert_allclose(g2.observables, g.observables)
    np.testing.assert_allclose(g2.targets, g.targets)
    assert g2.observable_names == g.observable_names
    assert g2.target_names == g.target_names
    assert g2.meta == g.meta

@pytest.mark.filterwarnings("ignore:.*overflow.*")
def test_modelgrid_fits_roundtrip(tmp_path, tiny_grid):
    path = tmp_path / "grid.fits"
    tiny_grid.to_fits_table(str(path), overwrite=True)
    g2 = ModelGrid.from_fits_table(str(path))
    np.testing.assert_allclose(g2.observables, tiny_grid.observables)
    np.testing.assert_allclose(g2.targets, tiny_grid.targets)
    assert g2.observable_names == tiny_grid.observable_names
    assert g2.target_names == tiny_grid.target_names

def test_modelgrid_hdf5_roundtrip(tmp_path, tiny_grid):
    path = tmp_path / "grid.h5"
    tiny_grid.to_hdf5(str(path), overwrite=True)
    g2 = ModelGrid.from_hdf5(str(path))
    np.testing.assert_allclose(g2.observables, tiny_grid.observables)
    np.testing.assert_allclose(g2.targets, tiny_grid.targets)
    assert g2.observable_names == tiny_grid.observable_names
    assert g2.target_names == tiny_grid.target_names

# ---------------------------
# HierarchicalGridBinner tests
# ---------------------------

def test_binner_fit_and_candidates(tiny_grid):
    g = tiny_grid
    b = HierarchicalGridBinner(dims=[0, 1], levels=3, base_bins=4)
    b.fit(g)
    assert b.mu is not None and b.sd is not None
    assert len(b.bin_edges) == 3
    # Query near [0,0] with small uncertainty
    y = np.array([0.05, 0.02])
    s = np.array([0.1, 0.1])
    idx, lev = b.candidates(y_native=y, sigmas_native=s, target_factor=2.0, expand_factor=2.0)
    assert idx.size > 0
    # Expect to include middle model index 3 most likely
    assert 3 in idx

# ---------------------------
# GridFitter tests
# ---------------------------

def test_posterior_over_models_normalization_and_order(tiny_grid):
    g = tiny_grid
    fitter = GridFitter(grid=g, likelihood=DummyLikelihood(), prior=DummyFlatPrior(), use_standardised=True)
    x = np.array([0.0, 0.0])
    sig = np.array([0.2, 0.2])
    # All models as candidates
    w = fitter.posterior_over_models(x_native=x, sigma_native=sig, candidate_idx=None)
    # Normalization
    np.testing.assert_allclose(w.sum(), 1.0, rtol=0, atol=1e-12)
    # The model at [0,0] should receive the largest weight (index 3)
    assert np.argmax(w) == 3

def test_posterior_over_target_histogram(tiny_grid):
    g = tiny_grid
    fitter = GridFitter(grid=g, likelihood=DummyLikelihood(), prior=DummyFlatPrior(), use_standardised=False)
    x = np.array([0.3, 0.3])
    sig = np.array([0.2, 0.2])
    # Target 0 is "sum" which ranges roughly [-4, 2]
    bins = np.linspace(-4.5, 2.5, 15)
    post, centers = fitter.posterior_over_target(x_native=x, sigma_native=sig, target_col="sum", bins=bins)
    assert post.shape == (bins.size - 1,)
    np.testing.assert_allclose(post.sum(), 1.0, atol=1e-12)

@pytest.mark.parametrize("backend", ["thread"])  # process backend may not pickle custom objects
def test_fit_batch_serial_and_threaded(tiny_grid, backend):
    g = tiny_grid
    binner = HierarchicalGridBinner(dims=[0, 1], levels=3, base_bins=4)
    binner.fit(g)

    fitter = GridFitter(grid=g, likelihood=DummyLikelihood(), prior=DummyFlatPrior(), use_standardised=True)

    # Build a mini batch
    X = np.array([
        [-0.1, -0.1],
        [ 0.0,  0.0],
        [ 0.9,  0.9],
    ])
    SIG = 0.2 * np.ones_like(X)

    # Stats over both targets
    bins_sum = np.linspace(-4.5, 2.5, 16)
    bins_prod = np.linspace(-2.5, 1.5, 16)

    out = fitter.fit_batch(
        X_native=X,
        SIG_native=SIG,
        binner=binner,
        target_factor=2.0,
        expand_factor=2.0,
        n_jobs=2,
        backend=backend,
        stats_for=["sum", "prod"],
        stats_bins=[bins_sum, bins_prod],
        find_multimodal=False,
        return_posts_for_stats=True,
        batch_size=2,
        verbose=False,
    )

    # Basic shape checks
    assert len(out["post_models"]) == X.shape[0]
    assert len(out["candidates"]) == X.shape[0]
    assert "stats" in out and "sum" in out["stats"] and "prod" in out["stats"]

    ssum = out["stats"]["sum"]
    for key in ["centers", "mean", "std", "map", "q16", "q50", "q84", "lo68", "hi68", "nmodes"]:
        assert key in ssum
    # Per-object stats exist
    assert ssum["mean"].shape == (X.shape[0],)

    # Posterior arrays per target if requested
    assert "posts_target" in out
    assert out["posts_target"]["sum"].shape == (X.shape[0], bins_sum.size - 1)

def test_corner_for_targets_smoke(tiny_grid):
    g = tiny_grid
    fitter = GridFitter(grid=g, likelihood=DummyLikelihood(), prior=DummyFlatPrior(), use_standardised=False)
    x = np.array([0.0, 0.0])
    sig = np.array([0.2, 0.2])
    fig, axes, summary = fitter.corner_for_targets(
        x_native=x,
        sigma_native=sig,
        target_cols=["sum", "prod"],
        bins=20,
        suptitle="Test",
        color=None,
    )
    assert axes.shape == (2, 2)
    assert "mean" in summary and summary["mean"].shape == (2,)
    # Clean up figure to avoid leaks in CI
    import matplotlib.pyplot as plt
    plt.close(fig)

def test_compute_pit_and_photoz_metrics_wrappers():
    # Simple synthetic discrete posterior: each row places all mass
    # on its bin just right of z_true -> PIT ~ fraction inside that bin, here 0.5
    z_true = np.array([0.1, 0.3, 0.9])
    edges = np.array([0.0, 0.2, 0.6, 1.2])
    posts = np.zeros((3, 3))
    posts[:, 1] = 1.0  # put mass in middle bin
    pit = GridFitter.compute_pit(z_true, posts, edges)
    assert pit.shape == (3,)
    assert np.all((pit >= 0.0) & (pit <= 1.0))

    # Photo-z metrics: identical z => near-zero errors
    z_est = z_true.copy()
    m = GridFitter.photoz_metrics(z_true, z_est)
    # We only check presence of keys; exact definitions may vary
    for k in ["bias", "nmad", "outlier", "rmse"]:
        assert k in m


# ---------------------------
# Edge cases / error handling
# ---------------------------

def test_modelgrid_shape_validation():
    with pytest.raises(ValueError):
        ModelGrid(
            observables=np.zeros((3, 2)),
            targets=np.zeros((4, 2)),  # mismatched N
            observable_names=["o0", "o1"],
            target_names=["t0", "t1"],
        )

def test_fit_batch_stats_bins_validation(tiny_grid):
    g = tiny_grid
    fitter = GridFitter(grid=g, likelihood=DummyLikelihood(), prior=DummyFlatPrior(), use_standardised=False)
    X = np.array([[0.0, 0.0]])
    SIG = np.array([[0.2, 0.2]])
    with pytest.raises(ValueError):
        fitter.fit_batch(
            X_native=X, SIG_native=SIG,
            stats_for=["sum"], stats_bins=None
        )

def test_binner_without_fit_raises(tiny_grid):
    g = tiny_grid
    b = HierarchicalGridBinner(dims=[0, 1])
    with pytest.raises(RuntimeError):
        b.choose_level(sigmas_native=np.array([0.1, 0.1]))
