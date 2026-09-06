import numpy as np


class FisherKPP:
    """
    2D Fisher-KPP reaction-diffusion model.

        du/dt = D * Laplacian(u) + r*u*(1-u/K)

    Parameters
    ----------
    D : float or np.ndarray
        Diffusion coefficient. Either a scalar (spatially constant)
        or an array D(x,y) matching the population grid shape, e.g.
        from `diffusion.terrain_diffusion`.
    r : float
        Intrinsic growth rate.
    K : float
        Carrying capacity.
    """

    def __init__(self, D: float | np.ndarray, r: float, K: float):
        if np.any(np.asarray(D) < 0):
            raise ValueError("D must be non-negative.")

        if r < 0:
            raise ValueError("r must be non-negative.")

        if K <= 0:
            raise ValueError("K must be positive.")

        self.D = D
        self.r = r
        self.K = K

    def reaction(self, u: np.ndarray) -> np.ndarray:
        """
        Logistic growth term:

            r*u*(1-u/K)
        """
        return self.r * u * (1.0 - u / self.K)

    def diffusion(self, laplacian_u: np.ndarray) -> np.ndarray:
        """
        Diffusion term:

            D * Laplacian(u)
        """
        return self.D * laplacian_u

    def rhs(
        self,
        u: np.ndarray,
        laplacian_u: np.ndarray
    ) -> np.ndarray:
        """
        Right-hand side of the Fisher-KPP equation.
        """
        return self.diffusion(laplacian_u) + self.reaction(u)