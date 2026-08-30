import numpy as np


def constant_diffusion(shape, D: float) -> np.ndarray:
    """
    Create a spatially constant diffusion coefficient.

    Parameters
    ----------
    shape : tuple
        Shape of the population grid.
    D : float
        Diffusion coefficient.

    Returns
    -------
    np.ndarray
        D(x,y) represented as an array.

    Notes
    -----
    M1에서는 아직 상수 D만 사용한다.

    추후 DEM 기반 모델에서는 이 함수를

        terrain_diffusion(dem, ...)

    형태로 확장할 수 있다.
    """

    if D < 0:
        raise ValueError("D must be non-negative.")

    return np.full(shape, D, dtype=float)