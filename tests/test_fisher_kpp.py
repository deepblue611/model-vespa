import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.solver.finite_difference import solve_fisher_kpp


def test_zero_population_stays_zero():

    x = np.linspace(0.0, 10.0, 101)
    u0 = np.zeros_like(x)

    times, solutions = solve_fisher_kpp(
        u0=u0,
        D=1.0,
        r=1.0,
        K=1.0,
        dx=x[1] - x[0],
        dt=0.001,
        n_steps=100,
    )

    assert np.allclose(solutions, 0.0)


def test_population_remains_nonnegative():

    x = np.linspace(0.0, 10.0, 101)

    u0 = np.exp(
        -(x - 5.0) ** 2
    )

    _, solutions = solve_fisher_kpp(
        u0=u0,
        D=1.0,
        r=1.0,
        K=1.0,
        dx=x[1] - x[0],
        dt=0.001,
        n_steps=100,
    )

    assert np.all(solutions >= 0.0)


def test_unstable_timestep_is_rejected():

    x = np.linspace(0.0, 10.0, 101)

    u0 = np.ones_like(x)

    with pytest.raises(ValueError):

        solve_fisher_kpp(
            u0=u0,
            D=1.0,
            r=1.0,
            K=1.0,
            dx=x[1] - x[0],
            dt=0.1,
            n_steps=10,
        )