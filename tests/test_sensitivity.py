import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.sensitivity import (
    morris_trajectories, morris_indices, saltelli_samples, sobol_indices,
    scale_to_bounds,
)


def ishigami(X, a=7.0, b=0.1):
    return np.sin(X[:, 0]) + a * np.sin(X[:, 1]) ** 2 + b * X[:, 2] ** 4 * np.sin(X[:, 0])


def test_morris_trajectories_change_one_factor_per_step():

    k = 5
    X = morris_trajectories(k, n_trajectories=10, n_levels=4, rng=0)

    assert X.shape == (10 * (k + 1), k)
    assert X.min() >= 0.0 and X.max() <= 1.0

    for t in range(10):
        steps = np.diff(X[t * (k + 1):(t + 1) * (k + 1)], axis=0)
        changed = np.abs(steps) > 1e-12
        assert (changed.sum(axis=1) == 1).all()
        # Every factor is moved exactly once per trajectory.
        assert (changed.sum(axis=0) == 1).all()


def test_morris_recovers_linear_coefficients():

    k = 4
    coefficients = np.array([3.0, -1.0, 0.0, 0.5])
    X = morris_trajectories(k, n_trajectories=20, rng=1)
    y = X @ coefficients

    result = morris_indices(X, y, k)

    assert np.allclose(result["mu"], coefficients)
    assert np.allclose(result["mu_star"], np.abs(coefficients))
    assert np.allclose(result["sigma"], 0.0, atol=1e-10)


def test_sobol_matches_ishigami_analytic_values():

    k = 3
    n_base = 4096
    bounds = [[-np.pi, np.pi]] * k
    X = scale_to_bounds(saltelli_samples(n_base, k, rng=2), bounds)

    result = sobol_indices(ishigami(X), n_base, k, n_bootstrap=50, rng=3)

    # Analytic values for a=7, b=0.1.
    assert np.allclose(result["S1"], [0.314, 0.442, 0.0], atol=0.05)
    assert np.allclose(result["ST"], [0.558, 0.442, 0.244], atol=0.05)


def test_sobol_drops_rows_with_nan_outputs():

    k = 3
    n_base = 256
    X = scale_to_bounds(saltelli_samples(n_base, k, rng=4), [[-np.pi, np.pi]] * k)
    y = ishigami(X)
    y[2 * n_base + 5] = np.nan   # one AB_1 evaluation failed

    result = sobol_indices(y, n_base, k, n_bootstrap=10, rng=5)

    assert result["n_valid"] == n_base - 1
    assert np.isfinite(result["S1"]).all()


def test_sobol_returns_nan_for_constant_output():

    k = 2
    n_base = 32
    y = np.full(n_base * (k + 2), 3.0)

    result = sobol_indices(y, n_base, k, n_bootstrap=10, rng=6)

    assert np.isnan(result["S1"]).all() and np.isnan(result["ST"]).all()
