import numpy as np
import pytest

from besta.grid import ModelGrid


def test_interpolate_observables_linear():
    # Simple 2D grid: targets (x,y) and observables equal to (2x, 3y)
    targets = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    observables = np.stack([2 * targets[:, 0], 3 * targets[:, 1]], axis=1)
    grid = ModelGrid(
        observables=observables,
        targets=targets,
        observable_names=["o1", "o2"],
        target_names=["x", "y"],
    )
    query = np.array([[0.5, 0.5]])
    interp = grid.interpolate_observables(query)
    np.testing.assert_allclose(interp, [[1.0, 1.5]], rtol=1e-6)


def test_interpolate_observables_shape_and_errors():
    targets = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    observables = np.stack([targets[:, 0], targets[:, 1]], axis=1)
    grid = ModelGrid(
        observables=observables,
        targets=targets,
        observable_names=["o1", "o2"],
        target_names=["x", "y"],
    )
    # wrong dimension
    with pytest.raises(ValueError):
        grid.interpolate_observables(np.array([[1.0, 2.0, 3.0]]))
    # too few models
    small_grid = ModelGrid(
        observables=observables[:2],
        targets=targets[:2],
        observable_names=["o1", "o2"],
        target_names=["x", "y"],
    )
    with pytest.raises(ValueError):
        small_grid.interpolate_observables(np.array([[0.1, 0.1]]))


def test_interpolate_observables_nearest():
    targets = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    observables = np.stack([2 * targets[:, 0], 3 * targets[:, 1]], axis=1)
    grid = ModelGrid(
        observables=observables,
        targets=targets,
        observable_names=["o1", "o2"],
        target_names=["x", "y"],
    )
    interp = grid.interpolate_observables([0.2, 0.8], method="nearest")
    assert interp.shape == (1, 2)
    assert any(np.allclose(interp, obs) for obs in observables)
