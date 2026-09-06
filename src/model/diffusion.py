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


def terrain_diffusion(
    dem: np.ndarray,
    dx: float,
    dy: float,
    D_max: float,
    slope_scale: float,
    power: float = 1.0,
    D_min: float = 0.0
) -> np.ndarray:
    """
    Build a spatially varying diffusion field D(x,y) from DEM slope.

    Steeper terrain slows diffusion:

        D(x,y) = D_min + (D_max - D_min) / (1 + (slope/slope_scale)**power)

    where slope is the local terrain gradient magnitude (rise/run),
    computed from the DEM with `np.gradient`.

    Parameters
    ----------
    dem : np.ndarray
        Elevation grid, shape (Ny, Nx). NaN cells (no data) are
        treated as flat (slope 0) for the gradient calculation.
    dx, dy : float
        Grid spacing in the same units as the DEM's horizontal
        coordinates (meters).
    D_max : float
        Diffusion coefficient on flat terrain (slope = 0).
    slope_scale : float
        Slope (rise/run) at which D has dropped halfway to D_min.
    power : float
        Steepness of the falloff around slope_scale.
    D_min : float
        Diffusion coefficient in the limit of infinite slope.

    Notes
    -----
    Sea-level cells in this DEM are filled with elevation 0, so flat
    ocean gets the same D_max as flat land. This is a known
    simplification for the current prototype -- a land/sea mask is
    needed on top of this for a physically accurate coastline
    barrier.
    """

    if D_max < 0:
        raise ValueError("D_max must be non-negative.")

    if D_min < 0 or D_min > D_max:
        raise ValueError("D_min must satisfy 0 <= D_min <= D_max.")

    if slope_scale <= 0:
        raise ValueError("slope_scale must be positive.")

    filled = np.nan_to_num(dem, nan=0.0)

    dz_dy, dz_dx = np.gradient(filled, dy, dx)
    slope = np.sqrt(dz_dx ** 2 + dz_dy ** 2)

    return D_min + (D_max - D_min) / (1.0 + (slope / slope_scale) ** power)