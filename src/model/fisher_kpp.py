import numpy as np


def logistic_growth(u: np.ndarray, r: float, K: float) -> np.ndarray:
    """
    Logistic growth term:

        f(u) = r * u * (1 - u / K)

    Parameters
    ----------
    u : np.ndarray
        Population density.
    r : float
        Intrinsic growth rate.
    K : float
        Carrying capacity.

    Returns
    -------
    np.ndarray
        Growth rate at each spatial point.
    """
    return r * u * (1.0 - u / K)


def fisher_kpp_rhs(
    u: np.ndarray,
    D: float,
    r: float,
    K: float,
    dx: float,
) -> np.ndarray:
    """
    Compute the spatial RHS of the 1D Fisher-KPP equation:

        du/dt = D * d²u/dx² + r*u*(1-u/K)

    Interior points only.

    Boundary points are handled separately by the solver.
    """
    diffusion = D * (
        u[2:] - 2.0 * u[1:-1] + u[:-2]
    ) / (dx ** 2)

    growth = logistic_growth(u[1:-1], r, K)

    return diffusion + growth