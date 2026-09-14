import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.model.vespa_competition import VespaCompetition
from src.solver.keller_segel import KellerSegel2D, stable_dt


def make_model(mask=None, chi_u=0.5):

    shape = (20, 20)

    if mask is None:
        mask = np.ones(shape, dtype=bool)

    C_u = np.full(shape, 2.0)
    S_v = np.full(shape, 0.5)

    return VespaCompetition(
        D_u=1.0,
        chi_u=chi_u,
        alpha=0.1,
        beta=0.05,
        C_u=C_u,
        K_v=1.0,
        S_v=S_v,
        mask=mask,
    )


def make_solver_and_velocity(model, dx=1.0, dy=1.0, dt=0.01):
    solver = KellerSegel2D(dx=dx, dy=dy, dt=dt)
    velocity_x, velocity_y = solver.taxis_velocity(model.C_u, model.chi_u)
    return solver, velocity_x, velocity_y


def test_zero_population_stays_zero():

    model = make_model()
    solver, vx, vy = make_solver_and_velocity(model)

    u0 = np.zeros((20, 20))
    u1 = solver.step(u0, model, vx, vy)

    assert np.allclose(u1, 0.0)


def test_population_remains_nonnegative():

    model = make_model()
    solver, vx, vy = make_solver_and_velocity(model)

    u0 = np.zeros((20, 20))
    u0[10, 10] = 1.0

    u1 = solver.step(u0, model, vx, vy)

    assert np.all(u1 >= 0.0)


def test_sea_cells_stay_at_zero():

    mask = np.ones((20, 20), dtype=bool)
    mask[:, 15:] = False  # right-hand strip is "sea"

    model = make_model(mask=mask)
    solver, vx, vy = make_solver_and_velocity(model)

    u0 = np.zeros((20, 20))
    u0[:, :15] = 1.0  # fill the whole land region

    u1 = u0.copy()
    for _ in range(20):
        u1 = solver.step(u1, model, vx, vy)

    assert np.allclose(u1[:, 15:], 0.0)


def test_no_flux_conserves_mass_with_no_reaction():
    """With growth/competition disabled, pure diffusion + taxis on a
    closed (fully-land) domain must conserve total mass exactly, since
    every boundary face flux is zero by construction."""

    shape = (20, 20)
    mask = np.ones(shape, dtype=bool)

    C_u = np.zeros(shape)
    C_u[:, 10:] = 1.0  # a step in the potential, to exercise taxis

    model = VespaCompetition(
        D_u=1.0, chi_u=0.5, alpha=0.0, beta=0.0,
        C_u=np.where(mask, C_u, 1.0) + 1e-6,  # keep C_u > 0 everywhere
        K_v=0.0, S_v=np.zeros(shape), mask=mask,
    )

    solver, vx, vy = make_solver_and_velocity(model, dt=0.005)

    u0 = np.zeros(shape)
    u0[8:12, 8:12] = 1.0

    total_before = u0.sum()

    u = u0.copy()
    for _ in range(50):
        u = solver.step(u, model, vx, vy)

    assert np.isclose(u.sum(), total_before, rtol=1e-8)


def test_laplacian_constant_field():

    solver = KellerSegel2D(dx=1.0, dy=1.0, dt=0.01)
    mask = np.ones((20, 20), dtype=bool)

    u = np.ones((20, 20))
    laplacian = solver.laplacian(u, mask)

    assert np.allclose(laplacian, 0.0)


def test_solution_shape():

    model = make_model()
    solver, vx, vy = make_solver_and_velocity(model)

    u0 = np.zeros((20, 20))

    times, solutions = solver.solve(
        u0=u0, model=model, velocity_x=vx, velocity_y=vy,
        steps=10, save_every=2,
    )

    assert solutions.shape[1:] == (20, 20)
    assert len(times) == len(solutions)


def test_invalid_parameters():

    shape = (5, 5)
    mask = np.ones(shape, dtype=bool)
    C_u = np.full(shape, 2.0)
    S_v = np.full(shape, 0.5)

    with pytest.raises(ValueError):
        VespaCompetition(
            D_u=-1.0, chi_u=0.1, alpha=0.1, beta=0.1,
            C_u=C_u, K_v=1.0, S_v=S_v, mask=mask,
        )

    with pytest.raises(ValueError):
        VespaCompetition(
            D_u=1.0, chi_u=-0.1, alpha=0.1, beta=0.1,
            C_u=C_u, K_v=1.0, S_v=S_v, mask=mask,
        )

    with pytest.raises(ValueError):
        VespaCompetition(
            D_u=1.0, chi_u=0.1, alpha=0.1, beta=0.1,
            C_u=np.zeros(shape), K_v=1.0, S_v=S_v, mask=mask,
        )


def test_stable_dt_decreases_with_higher_chi():

    velocity_low = (np.full((5, 5), 1.0), np.full((5, 5), 1.0))
    velocity_high = (np.full((5, 5), 10.0), np.full((5, 5), 10.0))

    dt_low = stable_dt(1.0, *velocity_low, dx=1.0, dy=1.0)
    dt_high = stable_dt(1.0, *velocity_high, dx=1.0, dy=1.0)

    assert dt_high < dt_low
