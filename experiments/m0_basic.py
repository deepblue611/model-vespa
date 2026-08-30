import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.solver.finite_difference import solve_fisher_kpp
from src.visualization.plot import plot_solution


def main():

    # ==========================
    # Model parameters
    # ==========================

    D = 1.0
    r = 1.0
    K = 1.0

    # ==========================
    # Spatial domain
    # ==========================

    L = 100.0
    dx = 0.1

    x = np.arange(0.0, L + dx, dx)

    # ==========================
    # Time parameters
    # ==========================

    dt = 0.001
    T = 50.0

    n_steps = int(T / dt)

    # ==========================
    # Initial condition
    # ==========================

    # Localized initial population.
    #
    # Instead of an exact delta function,
    # we use a Gaussian distribution.

    x0 = 10.0
    sigma = 1.0

    u0 = np.exp(
        -(x - x0) ** 2 / (2.0 * sigma ** 2)
    )

    # ==========================
    # Solve PDE
    # ==========================

    times, solutions = solve_fisher_kpp(
        u0=u0,
        D=D,
        r=r,
        K=K,
        dx=dx,
        dt=dt,
        n_steps=n_steps,
        save_every=100,
    )

    # ==========================
    # Visualization
    # ==========================

    plot_solution(
        x=x,
        times=times,
        solutions=solutions,
    )


if __name__ == "__main__":
    main()