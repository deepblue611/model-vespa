import sys
from pathlib import Path

import numpy as np

# Allow importing from src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.model.fisher_kpp import FisherKPP
from src.solver.finite_difference import FiniteDifference2D
from src.visualization.plot import plot_population


def create_initial_population(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    center_x: float,
    center_y: float,
    radius: float,
    density: float
) -> np.ndarray:
    """
    Create a circular initial population distribution.
    """

    x = np.arange(nx) * dx
    y = np.arange(ny) * dy

    X, Y = np.meshgrid(x, y)

    distance_squared = (
        (X - center_x) ** 2
        + (Y - center_y) ** 2
    )

    u0 = np.zeros((ny, nx))

    u0[distance_squared <= radius ** 2] = density

    return u0


def main():

    # ==========================================================
    # Grid
    # ==========================================================

    nx = 150
    ny = 150

    dx = 1.0
    dy = 1.0

    # ==========================================================
    # Fisher-KPP parameters
    # ==========================================================

    D = 1.0
    r = 0.1
    K = 1.0

    # ==========================================================
    # Time parameters
    # ==========================================================

    dt = 0.1
    total_time = 100.0

    steps = int(total_time / dt)

    save_every = 100

    # ==========================================================
    # Model
    # ==========================================================

    model = FisherKPP(
        D=D,
        r=r,
        K=K
    )

    # ==========================================================
    # Solver
    # ==========================================================

    solver = FiniteDifference2D(
        dx=dx,
        dy=dy,
        dt=dt
    )

    # ==========================================================
    # Initial condition
    # ==========================================================

    center_x = (nx - 1) * dx / 2
    center_y = (ny - 1) * dy / 2

    u0 = create_initial_population(
        nx=nx,
        ny=ny,
        dx=dx,
        dy=dy,
        center_x=center_x,
        center_y=center_y,
        radius=5.0,
        density=K
    )

    # ==========================================================
    # Simulation
    # ==========================================================

    times, solutions = solver.solve(
        u0=u0,
        model=model,
        steps=steps,
        save_every=save_every
    )

    # ==========================================================
    # Results
    # ==========================================================

    print("Simulation completed.")
    print(f"Grid: {nx} x {ny}")
    print(f"D = {D}")
    print(f"r = {r}")
    print(f"K = {K}")
    print(f"dt = {dt}")
    print(f"Total time = {total_time}")
    print(f"Saved snapshots = {len(solutions)}")

    # Final population
    plot_population(
        solutions[-1],
        time=times[-1],
        title="2D Fisher-KPP"
    )


if __name__ == "__main__":
    main()