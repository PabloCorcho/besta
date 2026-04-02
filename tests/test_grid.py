import numpy as np
import pytest

from besta.grid.grid import ModelGrid, GridFitter, _truncate_posterior_mass
from besta.grid.binning import RectBinner, KDTreeBinner, HashedGridBinner
from besta.grid.transforms import LinearStandardiser, MagTransform
from besta.grid.prob import (
    FlatPrior,
    GaussianPrior1D,
    UniformPrior1D,
    DeltaPrior1D,
    ExponentialPrior1D,
    ExponentialTruncatedPrior1D,
    PowerLawPrior1D,
    EmpiricalHistogramPrior1D,
    EmpiricalFlatteningPriorND,
    MagDependentRedshiftPrior,
    CompositePrior,
    ObservableCompositePrior,
    GaussianProductLikelihood,
    CensoredSizeLikelihood,
    CompositeLikelihood,
    NumbaGaussianProductLikelihood,
    posterior_over_models,
)


def _make_grid(n_side=5, with_weights=True):
    xv, yv = np.meshgrid(np.linspace(0.0, 1.0, n_side), np.linspace(0.0, 1.0, n_side))
    x = xv.ravel()
    y = yv.ravel()
    targets = np.column_stack([x, y])
    observables = np.column_stack(
        [
            2.0 * x + 0.5 * y,
            -1.0 * x + 3.0 * y,
            0.1 + x + y,
        ]
    )
    weights = None
    if with_weights:
        weights = np.linspace(1.0, 2.0, targets.shape[0])
    return ModelGrid(
        observables=observables,
        targets=targets,
        observable_names=["o1", "o2", "mag"],
        target_names=["x", "y"],
        weights=weights,
        meta={"source": "unit-test"},
    )


def test_modelgrid_validation_errors():
    obs = np.ones((3, 2))
    tgt = np.ones((3, 2))
    with pytest.raises(ValueError):
        ModelGrid(obs[0], tgt, ["a", "b"], ["x", "y"])
    with pytest.raises(ValueError):
        ModelGrid(obs, tgt[0], ["a", "b"], ["x", "y"])
    with pytest.raises(ValueError):
        ModelGrid(obs, np.ones((2, 2)), ["a", "b"], ["x", "y"])
    with pytest.raises(ValueError):
        ModelGrid(obs, tgt, ["a"], ["x", "y"])
    with pytest.raises(ValueError):
        ModelGrid(obs, tgt, ["a", "b"], ["x"], weights=np.ones(3))
    with pytest.raises(ValueError):
        ModelGrid(obs, tgt, ["a", "b"], ["x", "y"], check_boundaries=7)


def test_modelgrid_select_and_standardisers():
    grid = _make_grid(n_side=4)
    grid.fit_standardiser()
    grid.fit_target_standardiser()
    idx = np.array([0, 2, 5])

    sub = grid.select(idx, observables=["o1", "mag"], targets=[1], standardisers=True)

    assert sub.n_models == 3
    assert sub.observable_names == ["o1", "mag"]
    assert sub.target_names == ["y"]
    assert sub.observable_standardiser.is_fit
    assert sub.target_standardiser.is_fit


def test_modelgrid_standardiser_transform_roundtrip_and_runtime_errors():
    grid = _make_grid()
    with pytest.raises(RuntimeError):
        grid.transform_observables(grid.observables)
    with pytest.raises(RuntimeError):
        grid.transform_targets(grid.targets)

    grid.fit_standardiser()
    grid.fit_target_standardiser()
    xo = grid.transform_observables(grid.observables)
    yo = grid.transform_targets(grid.targets)
    np.testing.assert_allclose(grid.inverse_transform_observables(xo), grid.observables)
    np.testing.assert_allclose(grid.inverse_transform_targets(yo), grid.targets)


def test_modelgrid_standardiser_caches_follow_grid_and_subset():
    grid = _make_grid(n_side=4)
    grid.fit_standardiser()
    grid.fit_target_standardiser()

    np.testing.assert_allclose(
        grid._observables_standardized, grid.transform_observables(grid.observables)
    )
    np.testing.assert_allclose(
        grid._targets_standardized, grid.transform_targets(grid.targets)
    )

    idx = np.array([0, 3, 5])
    sub = grid.select(idx, observables=[0, 2], targets=[1], standardisers=True)
    np.testing.assert_allclose(
        sub._observables_standardized,
        grid._observables_standardized[idx][:, [0, 2]],
    )
    np.testing.assert_allclose(
        sub._targets_standardized,
        grid._targets_standardized[idx][:, [1]],
    )


def test_modelgrid_kdtree_requires_fit_when_standardized():
    grid = _make_grid()
    with pytest.raises(RuntimeError):
        grid.get_kdtree(standardize=True)
    tree = grid.get_kdtree(standardize=False)
    assert tree.n == grid.n_models


def test_modelgrid_interpolation_modes_and_boundaries():
    boundary = lambda q: np.all((q >= 0.0) & (q <= 1.0), axis=1)
    grid = _make_grid(n_side=5)
    grid.check_boundaries = boundary

    q = np.array([[0.25, 0.75]])
    out_local = grid.interpolate_observables(q, method="local_linear")
    expected = np.array([[2.0 * 0.25 + 0.5 * 0.75, -0.25 + 3.0 * 0.75, 0.1 + 0.25 + 0.75]])
    np.testing.assert_allclose(out_local, expected, atol=1e-6)

    out_near = grid.interpolate_observables([0.2, 0.8], method="nearest")
    assert out_near.shape == (1, 3)

    out_idw = grid.interpolate_observables(q, method="idw", k=8)
    assert out_idw.shape == (1, 3)

    bad = grid.interpolate_observables(np.array([[1.2, 0.5]]), fill_value=-99.0)
    np.testing.assert_allclose(bad, -99.0)

    with pytest.raises(ValueError):
        grid.interpolate_observables(np.array([[0.1, 0.2, 0.3]]))


def test_modelgrid_to_from_dict_preserves_standardisers():
    grid = _make_grid()
    grid.fit_standardiser()
    grid.fit_target_standardiser()
    payload = grid.to_dict()
    loaded = ModelGrid.from_dict(payload)

    np.testing.assert_allclose(loaded.observables, grid.observables)
    np.testing.assert_allclose(loaded.targets, grid.targets)
    assert loaded.observable_standardiser.is_fit
    assert loaded.target_standardiser.is_fit


def test_modelgrid_hdf5_and_fits_roundtrip(tmp_path):
    grid = _make_grid()

    h5 = tmp_path / "grid.h5"
    fits = tmp_path / "grid.fits"

    grid.to_hdf5(str(h5), overwrite=True)
    g_h5 = ModelGrid.from_hdf5(str(h5))
    np.testing.assert_allclose(g_h5.observables, grid.observables)
    np.testing.assert_allclose(g_h5.targets, grid.targets)

    grid.to_fits_table(str(fits), overwrite=True)
    g_fits = ModelGrid.from_fits_table(str(fits))
    np.testing.assert_allclose(g_fits.observables, grid.observables)
    np.testing.assert_allclose(g_fits.targets, grid.targets)


def test_linear_standardiser_and_mag_transform_roundtrip():
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    std = LinearStandardiser().fit(X)
    Xz = std.transform(X)
    np.testing.assert_allclose(std.inverse_transform(Xz), X)

    state = std.to_dict()
    std2 = LinearStandardiser.from_dict(state)
    np.testing.assert_allclose(std2.transform(X), Xz)

    mt = MagTransform(zero_point=25.0)
    flux = np.array([1e-1, 1.0, 10.0])
    mag = mt.flux_to_mag(flux)
    np.testing.assert_allclose(mt.mag_to_flux(mag), flux)


def test_rect_binner_fit_candidates_and_persistence(tmp_path):
    grid = _make_grid(n_side=7)
    rb = RectBinner(
        dims=[0, 1, 2],
        levels=3,
        base_bins=3,
        mode="target_k",
        target_k=8,
        transform="pca_whiten",
    )
    rb.fit(grid)

    q = grid.observables[5]
    sig = np.full(grid.n_observables, 0.05)
    idx, lev = rb.candidates(q, sig)
    assert idx.size > 0
    assert isinstance(lev, int)

    path = tmp_path / "rect_binner.json"
    rb.save(str(path))
    rb2 = RectBinner.load(str(path))
    idx2, _ = rb2.candidates(q, sig)
    assert idx2.size > 0


def test_kdtree_binner_modes_and_persistence(tmp_path):
    grid = _make_grid(n_side=6)
    kb = KDTreeBinner(dims=[0, 1, 2], select_mode="knn", target_k=10, transform="pca_whiten")
    kb.fit(grid)

    q = grid.observables[7]
    idx_knn, k = kb.candidates(q)
    assert idx_knn.size == 10
    assert k == 10

    kb.select_mode = "radius"
    kb.radius_shape = "ellipsoid"
    idx_rad, aux = kb.candidates(q, np.full(grid.n_observables, 0.08))
    assert idx_rad.size > 0
    assert aux is None

    path = tmp_path / "kdtree_binner.json"
    kb.save(str(path))
    kb2 = KDTreeBinner.load(str(path))
    kb2.select_mode = "knn"
    idx2, _ = kb2.candidates(q)
    assert idx2.size > 0


def test_hashed_grid_binner_modes_and_persistence(tmp_path):
    grid = _make_grid(n_side=6)
    hb = HashedGridBinner(
        dims=[0, 1, 2],
        select_mode="knn",
        target_k=10,
        transform="pca_whiten",
        target_cell_occupancy=6,
        max_expand_steps=6,
    )
    hb.fit(grid)

    q = grid.observables[7]
    idx_knn, k = hb.candidates(q)
    assert idx_knn.size == 10
    assert k == 10
    assert idx_knn[0] == 7

    hb.select_mode = "radius"
    hb.radius_shape = "ellipsoid"
    idx_rad, aux = hb.candidates(q, np.full(grid.n_observables, 0.08))
    assert idx_rad.size > 0
    assert aux is None

    path = tmp_path / "hashed_binner.json"
    hb.save(str(path))
    hb2 = HashedGridBinner.load(str(path))
    hb2.select_mode = "knn"
    idx2, _ = hb2.candidates(q)
    assert idx2.size > 0


def test_flat_and_elementary_priors_shapes_and_support():
    grid = _make_grid(n_side=4)
    t = grid.targets

    assert FlatPrior().log_prob_for_models(t).shape == (grid.n_models,)

    lp_g = GaussianPrior1D(target_col=0, mu=0.5, sigma=0.2).log_prob_for_models(t)
    assert np.all(np.isfinite(lp_g))

    lp_u = UniformPrior1D(target_col=0, low=0.2, high=0.8).log_prob_for_models(t)
    assert np.any(np.isneginf(lp_u))

    lp_d = DeltaPrior1D(target_col=0, value=(1.0 / 3.0), tolerance=1e-6).log_prob_for_models(t)
    assert np.any(np.isfinite(lp_d))

    lp_e = ExponentialPrior1D(target_col=0, scale=0.5).log_prob_for_models(t)
    assert np.all(np.isfinite(lp_e))

    lp_et = ExponentialTruncatedPrior1D(target_col=0, scale=0.5, t_max=0.9).log_prob_for_models(t)
    assert np.any(np.isneginf(lp_et))

    lp_pw = PowerLawPrior1D(target_col=0, alpha=1.0, t_min=0.1, t_max=1.0).log_prob_for_models(t)
    assert np.any(np.isfinite(lp_pw))


def test_empirical_priors_fit_and_evaluate():
    grid = _make_grid(n_side=8)
    t = grid.targets

    hist = EmpiricalHistogramPrior1D(target_col=0, edges=np.linspace(0, 1, 11))
    with pytest.raises(RuntimeError):
        hist.log_prob_for_models(t)
    hist.fit_from_targets(t)
    lp_h = hist.log_prob_for_models(t)
    assert lp_h.shape == (grid.n_models,)

    ef = EmpiricalFlatteningPriorND(
        target_cols=[0, 1],
        edges_list=[np.linspace(0, 1, 7), np.linspace(0, 1, 7)],
        mode="factorised",
    ).fit_from_targets(t)
    lp_ef = ef.log_prob_for_models(t)
    assert lp_ef.shape == (grid.n_models,)

    ej = EmpiricalFlatteningPriorND(
        target_cols=[0, 1],
        edges_list=[np.linspace(0, 1, 7), np.linspace(0, 1, 7)],
        mode="joint",
    ).fit_from_targets(t)
    lp_ej = ej.log_prob_for_models(t)
    assert lp_ej.shape == (grid.n_models,)


def test_composite_priors_and_observable_dependent_prior():
    grid = _make_grid(n_side=6)

    p1 = GaussianPrior1D(target_col=0, mu=0.4, sigma=0.3)
    p2 = UniformPrior1D(target_col=1, low=0.1, high=0.9)
    cp = CompositePrior(priors=[p1, p2], weights=[1.0, 0.5])
    lp = cp.log_prob_for_models(grid.targets)
    assert lp.shape == (grid.n_models,)

    zprior = MagDependentRedshiftPrior(
        z_col=0,
        mag_observable_index=2,
        z_edges=np.linspace(0.0, 1.0, 11),
        m_edges=np.linspace(grid.observables[:, 2].min(), grid.observables[:, 2].max(), 11),
    ).fit_from_grid(grid.observables, grid.targets)

    ocp = ObservableCompositePrior(priors=[p1, zprior])
    with pytest.raises(ValueError):
        ocp.log_prob_for_models(grid.targets)
    lp_obs = ocp.log_prob_for_models(grid.targets, observables=grid.observables)
    assert lp_obs.shape == (grid.n_models,)

    with pytest.raises(TypeError):
        CompositePrior(priors=[zprior])


def test_likelihoods_and_posterior_helper_normalization():
    grid = _make_grid(n_side=5)
    x = grid.observables[10]
    sig = np.full(grid.n_observables, 0.05)

    gp = GaussianProductLikelihood(bandwidth_floor=1e-3)
    ll = gp.log_likelihood(x, sig, grid.observables)
    assert ll.shape == (grid.n_models,)
    assert np.argmax(ll) == 10

    cl = CensoredSizeLikelihood(
        phot_indices=[0, 1],
        size_index=2,
        s_min=float(grid.observables[:, 2].mean()),
        bandwidth_floor=1e-3,
    )
    ll2 = cl.log_likelihood(x, sig, grid.observables)
    assert ll2.shape == (grid.n_models,)

    comp = CompositeLikelihood([gp, gp])
    llc = comp.log_likelihood(x, sig, grid.observables)
    np.testing.assert_allclose(llc, 2.0 * ll)

    nb = NumbaGaussianProductLikelihood(bandwidth_floor=1e-3)
    lln = nb.log_likelihood(x, sig, grid.observables)
    assert lln.shape == (grid.n_models,)

    w = posterior_over_models(
        x_native=x,
        sigma_native=sig,
        X_models=grid.observables,
        targets_models=grid.targets,
        likelihood=gp,
        prior=FlatPrior(),
        model_weights=grid.weights,
    )
    np.testing.assert_allclose(w.sum(), 1.0)
    assert np.all(w >= 0)


def test_truncate_posterior_mass_behavior():
    idx = np.arange(6)
    w = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    kept_idx, kept_w, meta = _truncate_posterior_mass(idx, w, keep_mass=0.75, min_candidates=2)
    assert kept_idx.size >= 2
    np.testing.assert_allclose(kept_w.sum(), 1.0)
    assert meta["mass_kept"] >= 0.75


def test_truncate_posterior_mass_fast_paths_preserve_limits_and_ties():
    idx = np.arange(6)
    w = np.array([0.4, 0.3, 0.15, 0.15, 0.0, 0.0])

    kept_idx, kept_w, meta = _truncate_posterior_mass(
        idx, w, keep_mass=1.0, max_candidates=2
    )
    assert kept_idx.tolist() == [0, 1]
    np.testing.assert_allclose(kept_w.sum(), 1.0)
    assert meta["n_after"] == 2

    kept_idx, kept_w, meta = _truncate_posterior_mass(
        idx, w, keep_mass=0.8, keep_ties=True
    )
    assert kept_idx.size == 4
    assert set(kept_idx.tolist()) == {0, 1, 2, 3}
    np.testing.assert_allclose(kept_w.sum(), 1.0)
    assert meta["mass_kept"] >= 0.8


def test_gridfitter_posterior_over_models_and_target():
    grid = _make_grid(n_side=6)
    fitter = GridFitter(grid, likelihood=GaussianProductLikelihood(bandwidth_floor=1e-3), prior=FlatPrior())

    x = grid.observables[8]
    sig = np.full(grid.n_observables, 0.05)

    cand = np.array([1, 2, 8, 9, 10])
    w = fitter.posterior_over_models(x, sig, candidate_idx=cand)
    np.testing.assert_allclose(w.sum(), 1.0)
    assert w.shape == (cand.size,)

    bins = np.linspace(0, 1, 21)
    post, centers = fitter.posterior_over_target(x, sig, target_col="x", bins=bins, candidate_idx=cand)
    assert post.shape == (bins.size - 1,)
    assert centers.shape == (bins.size - 1,)
    np.testing.assert_allclose(np.sum(post * np.diff(bins)), 1.0, atol=1e-6)

    with pytest.raises(KeyError):
        fitter.posterior_over_target(x, sig, target_col="unknown", bins=bins)


def test_gridfitter_fit_batch_iter_list_and_hdf5(tmp_path):
    grid = _make_grid(n_side=6)
    fitter = GridFitter(grid, likelihood=GaussianProductLikelihood(bandwidth_floor=1e-3), prior=FlatPrior())

    X = grid.observables[:8]
    S = np.full_like(X, 0.05)

    binner = KDTreeBinner(dims=[0, 1, 2], select_mode="knn", target_k=12)
    binner.fit(grid)

    out_list = fitter.fit_batch(
        X,
        S,
        binner=binner,
        n_jobs=1,
        stats_for=["x"],
        stats_bins=[np.linspace(0.0, 1.0, 12)],
        return_mode="list",
        posterior_keep_mass=0.9,
    )
    assert len(out_list) == X.shape[0]
    assert all("stats" in r for r in out_list)

    capped = fitter.fit_batch(
        X[:2],
        S[:2],
        binner=binner,
        n_jobs=1,
        return_mode="list",
        posterior_keep_mass=1.0,
        posterior_keep_max_candidates=3,
    )
    assert all(r["candidates"].size <= 3 for r in capped)

    out_iter = fitter.fit_batch(X, S, binner=binner, n_jobs=1, return_mode="iter")
    assert len(list(out_iter)) == X.shape[0]

    h5 = tmp_path / "fit_batch.h5"
    out_write_only = fitter.fit_batch(
        X,
        S,
        binner=binner,
        n_jobs=1,
        return_mode="list",
        output_hdf5_path=str(h5),
        output_hdf5_overwrite=True,
        output_hdf5_write_only=True,
    )
    assert out_write_only == []
    assert h5.exists()


def test_gridfitter_fit_batch_dry_run():
    grid = _make_grid(n_side=4)
    fitter = GridFitter(grid)
    X = grid.observables[:3]
    S = np.full_like(X, 0.1)

    out = fitter.fit_batch(X, S, dry_run=True, return_mode="list")
    assert out == []
