import numpy as np

from src.model.fisher_kpp import fisher_kpp_rhs


def apply_neumann_boundary(u: np.ndarray) -> None:
    """
    Apply zero-flux Neumann boundary conditions:

        du/dx = 0

    at both boundaries.
    """
    u[0] = u[1]
    u[-1] = u[-2]


def solve_fisher_kpp(
    u0: np.ndarray,
    D: float,
    r: float,
    K: float,
    dx: float,
    dt: float,
    n_steps: int,
    save_every: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the 1D Fisher-KPP equation using explicit Euler
    time integration and second-order finite differences.

        du/dt = D*u_xx + r*u*(1-u/K)

    Boundary condition:
        du/dx = 0

    Returns
    -------
    times : np.ndarray
        Saved simulation times.

    solutions : np.ndarray
        Population density.

        Shape:
            (number_of_saved_steps, number_of_grid_points)
    """

    u = np.array(u0, dtype=float, copy=True)

    # Explicit diffusion stability condition:
    # D * dt / dx^2 <= 1/2
    diffusion_number = D * dt / (dx ** 2)

    if diffusion_number > 0.5:
        raise ValueError(
            f"Unstable timestep: D*dt/dx^2 = "
            f"{diffusion_number:.4f} > 0.5"
        )

    solutions = [u.copy()]
    times = [0.0]

    for step in range(1, n_steps + 1):

        rhs = fisher_kpp_rhs(
            u=u,
            D=D,
            r=r,
            K=K,
            dx=dx,
        )

        u[1:-1] += dt * rhs

        apply_neumann_boundary(u)

        # Numerical sanity check
        if np.any(~np.isfinite(u)):
            raise FloatingPointError(
                "Non-finite population density encountered."
            )

        if np.any(u < -1e-12):
            raise FloatingPointError(
                "Negative population density encountered."
            )

        # Remove tiny numerical negative values.
        u[u < 0.0] = 0.0

        if step % save_every == 0:
            solutions.append(u.copy())
            times.append(step * dt)

    return np.array(times), np.array(solutions)