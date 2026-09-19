import numpy as np


class VespaCompetition:
    """
    Vespa velutina (invasive) vs. Vespa mandarinia (native) competition
    model with environmental-capacity taxis (see vespa_pde_model_spec.md).

        du/dt = D_u * Laplacian(u)
                - chi_u * div(u * grad(C_u(x)))
                + alpha * u * (1 - u / C_u(x))
                - beta * u * v(x)

        v(x) = K_v * S_v(x)   (static field)

    C_u(x) is used both as the taxis potential (organisms move up its
    gradient) and as the logistic carrying capacity, per the spec's
    working assumption.

    Parameters
    ----------
    D_u : float
        Diffusion coefficient of the invasive species.
    chi_u : float
        Taxis sensitivity toward the C_u gradient.
    alpha : float
        Intrinsic logistic growth rate of the invasive species.
    beta : float
        One-way Lotka-Volterra competition coefficient (the native
        species suppresses the invasive one; v is unaffected by u).
    C_u : np.ndarray
        Carrying capacity / taxis potential field, shape (Ny, Nx).
    K_v : float
        Scale factor turning normalized suitability into a native
        species density field.
    S_v : np.ndarray
        Normalized (0-1) environmental suitability for the native
        species, same shape as C_u.
    mask : np.ndarray of bool
        True where the cell is land (part of the simulation domain).
        Every term is forced to zero outside the mask.
    xp : module
        Array module the input arrays belong to -- `numpy` (default) or
        `cupy` for a GPU-resident run. All array ops use this module so
        the same class works on either backend.
    """

    def __init__(
        self,
        D_u: float,
        chi_u: float,
        alpha: float,
        beta: float,
        C_u: np.ndarray,
        K_v: float,
        S_v: np.ndarray,
        mask: np.ndarray,
        xp=np,
    ):
        if D_u < 0:
            raise ValueError("D_u must be non-negative.")

        if chi_u < 0:
            raise ValueError("chi_u must be non-negative.")

        if alpha < 0:
            raise ValueError("alpha must be non-negative.")

        if beta < 0:
            raise ValueError("beta must be non-negative.")

        if C_u.shape != mask.shape or S_v.shape != mask.shape:
            raise ValueError("C_u, S_v and mask must share the same shape.")

        if xp.any(C_u[mask] <= 0):
            raise ValueError("C_u must be positive on every land cell.")

        self.D_u = D_u
        self.chi_u = chi_u
        self.alpha = alpha
        self.beta = beta
        self.mask = mask
        self.xp = xp

        # Sea cells are excluded from every output term below, but C_u
        # still needs a finite, positive value there so that 1 - u/C_u
        # does not divide by zero while the RHS is being computed.
        self.C_u = xp.where(mask, C_u, 1.0)

        self.v = K_v * xp.where(mask, S_v, 0.0)

    def reaction(self, u: np.ndarray) -> np.ndarray:
        """
        Logistic growth minus one-way competition:

            alpha*u*(1 - u/C_u) - beta*u*v
        """
        growth = self.alpha * u * (1.0 - u / self.C_u)
        competition = self.beta * u * self.v

        return self.xp.where(self.mask, growth - competition, 0.0)

    def diffusion(self, laplacian_u: np.ndarray) -> np.ndarray:
        """
        Diffusion term:

            D_u * Laplacian(u)
        """
        return self.D_u * laplacian_u

    def rhs(
        self,
        u: np.ndarray,
        laplacian_u: np.ndarray,
        taxis_divergence: np.ndarray,
    ) -> np.ndarray:
        """
        Right-hand side of the competition-taxis equation.

        Parameters
        ----------
        laplacian_u : np.ndarray
            Precomputed Laplacian of u (from the solver).
        taxis_divergence : np.ndarray
            Precomputed div(u * chi_u * grad(C_u)) (from the solver).
        """
        du_dt = (
            self.diffusion(laplacian_u)
            - taxis_divergence
            + self.reaction(u)
        )

        return self.xp.where(self.mask, du_dt, 0.0)
