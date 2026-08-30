import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.model.fisher_kpp import FisherKPP
from src.solver.finite_difference import FiniteDifference2D


def test_zero_population_stays_zero():

    model = FisherKPP(
        D=1.0,
        r=0.1,
        K=1.0
    )

    solver = FiniteDifference2D(
        dx=1.0,
        dy=1.0,
        dt=0.01
    )

    u0 = np.zeros((20, 20))

    u1 = solver.step(u0, model)

    assert np.allclose(u1, 0.0)


def test_population_remains_nonnegative():

    model = FisherKPP(
        D=1.0,
        r=0.1,
        K=1.0
    )

    solver = FiniteDifference2D(
        dx=1.0,
        dy=1.0,
        dt=0.01
    )

    u0 = np.zeros((20, 20))

    u0[10, 10] = 1.0

    u1 = solver.step(u0, model)

    assert np.all(u1 >= 0.0)


def test_laplacian_constant_field():

    solver = FiniteDifference2D(
        dx=1.0,
        dy=1.0,
        dt=0.01
    )

    u = np.ones((20, 20))

    laplacian = solver.laplacian(u)

    assert np.allclose(laplacian, 0.0)


def test_solution_shape():

    model = FisherKPP(
        D=1.0,
        r=0.1,
        K=1.0
    )

    solver = FiniteDifference2D(
        dx=1.0,
        dy=1.0,
        dt=0.01
    )

    u0 = np.zeros((20, 30))

    times, solutions = solver.solve(
        u0=u0,
        model=model,
        steps=10,
        save_every=2
    )

    assert solutions.shape[1:] == (20, 30)
    assert len(times) == len(solutions)


def test_invalid_parameters():

    with pytest.raises(ValueError):
        FisherKPP(
            D=-1.0,
            r=0.1,
            K=1.0
        )

    with pytest.raises(ValueError):
        FisherKPP(
            D=1.0,
            r=-0.1,
            K=1.0
        )

    with pytest.raises(ValueError):
        FisherKPP(
            D=1.0,
            r=0.1,
            K=0.0
        )