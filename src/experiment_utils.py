"""
Shared helpers for the M3 (Vespa competition) experiment scripts:
loading/coarsening the preprocessed 500m fields, grid indexing, initial
condition construction, and the GPU-venv relaunch trick. Factored out of
m3_vespa_competition.py so m3_chi_beta_sweep.py doesn't duplicate it.
"""

import sys
from pathlib import Path

import numpy as np


def block_reduce_1d(arr, factor):
    """Average consecutive groups of `factor` cell centers onto a coarser axis."""

    n = arr.shape[0] - arr.shape[0] % factor
    return arr[:n].reshape(n // factor, factor).mean(axis=1)


def block_reduce_fields(land_mask, C_u, S_v, factor):
    """
    Coarsen land_mask/C_u/S_v by block-averaging factor x factor cells.

    A coarse cell is land if a majority of its sub-cells are land; C_u/S_v
    are averaged over land sub-cells only (sea sub-cells, which carry no
    valid value, are excluded rather than dragging the average toward 0).
    """

    ny, nx = land_mask.shape
    ny2, nx2 = ny - ny % factor, nx - nx % factor

    mask_blocks = land_mask[:ny2, :nx2].reshape(ny2 // factor, factor, nx2 // factor, factor)
    land_count = mask_blocks.sum(axis=(1, 3))
    coarse_mask = land_count > (factor * factor) / 2

    def masked_mean(field):
        blocks = field[:ny2, :nx2].reshape(ny2 // factor, factor, nx2 // factor, factor)
        land_sum = np.where(mask_blocks, blocks, 0.0).sum(axis=(1, 3))
        return land_sum / np.maximum(land_count, 1)

    coarse_C_u = np.where(coarse_mask, np.maximum(masked_mean(C_u), 1e-6), 1.0)
    coarse_S_v = np.where(coarse_mask, masked_mean(S_v), 0.0)

    return coarse_mask, coarse_C_u, coarse_S_v


def load_vespa_fields(data_dir: Path, grid_factor: int = 1):
    """Load the 500m fields, optionally block-reduced by `grid_factor`."""

    land_mask = np.load(data_dir / "land_mask_500m.npy")
    C_u = np.load(data_dir / "carrying_capacity_Cu_500m.npy").astype(float)
    S_v = np.load(data_dir / "suitability_Sv_normalized_500m.npy").astype(float)
    x_centers = np.load(data_dir / "grid_x_500m.npy")
    y_centers = np.load(data_dir / "grid_y_500m.npy")

    if grid_factor > 1:
        land_mask, C_u, S_v = block_reduce_fields(land_mask, C_u, S_v, grid_factor)
        x_centers = block_reduce_1d(x_centers, grid_factor)
        y_centers = block_reduce_1d(y_centers, grid_factor)

    return land_mask, C_u, S_v, x_centers, y_centers


def to_grid_index(lon, lat, transformer, x_centers, y_centers):
    px, py = transformer.transform(lon, lat)

    col = int(np.argmin(np.abs(x_centers - px)))
    row = int(np.argmin(np.abs(y_centers - py)))

    return row, col


def nearest_land_cell(mask, row, col):
    """Snap to the nearest land cell if (row, col) itself has no SDM
    coverage (e.g. Busan's exact point falling on a harbor cell)."""

    if mask[row, col]:
        return row, col

    land_rows, land_cols = np.nonzero(mask)
    distances = (land_rows - row) ** 2 + (land_cols - col) ** 2
    nearest = np.argmin(distances)

    return int(land_rows[nearest]), int(land_cols[nearest])


def create_point_source(shape, center_row, center_col, radius_m, density, grid):
    """Circular initial population centered on a single grid cell."""

    ny, nx = shape

    rows = np.arange(ny).reshape(-1, 1)
    cols = np.arange(nx).reshape(1, -1)

    distance = np.sqrt(
        ((rows - center_row) * grid) ** 2
        + ((cols - center_col) * grid) ** 2
    )

    u0 = np.zeros(shape)
    u0[distance <= radius_m] = density

    return u0


def ensure_gpu_interpreter(gpu_python: str):
    """
    CuPy's NVRTC kernel compiler cannot handle this project's Korean-named
    OneDrive path, so BACKEND="gpu" only works from a Python interpreter
    installed at an ASCII-only path (e.g. C:\\gpuvenv). If the running
    interpreter isn't that one, relaunch this same script there and exit.
    """

    if Path(sys.executable).resolve() == Path(gpu_python).resolve():
        return

    import subprocess

    if not Path(gpu_python).exists():
        raise RuntimeError(
            f"BACKEND='gpu' but the GPU venv interpreter was not found at "
            f"{gpu_python} -- set up a venv there with cupy-cuda12x[ctk], "
            f"matplotlib and pyproj installed."
        )

    print(f"BACKEND='gpu': relaunching under {gpu_python} ...")
    result = subprocess.run([gpu_python, sys.argv[0], *sys.argv[1:]])
    sys.exit(result.returncode)
