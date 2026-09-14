"""
Preprocessing for the Vespa velutina / Vespa mandarinia competition-taxis
model (see vespa_pde_model_spec.md).

Builds the two static input fields the model needs on a common 500 m
EPSG:5179 grid:

    C_u(x)  -- Vespa velutina carrying capacity / taxis potential
               (raw units, from data/raw/SAM/density.csv)
    S_v(x)  -- Vespa mandarinia environmental suitability, normalized
               to [0, 1] (from data/raw/jangsu/suitability.csv)

plus a land/sea mask, since both source rasters only contain samples over
the South Korea landmass (no ocean rows) and are not on the same grid or
resolution as each other or as the 500 m target grid.

Both source files are irregular-resolution rasters (density.csv is a
~0.01 deg / ~900 m grid, suitability.csv is a 5 arcmin / ~9 km grid), so
producing the 500 m grid is upsampling (linear interpolation), not the
mean-aggregation downsampling used in dem_downscale.py for the 90 m DEM.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.interpolate import griddata
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DENSITY_CSV = PROJECT_ROOT / "data" / "raw" / "SAM" / "density.csv"
SUITABILITY_CSV = PROJECT_ROOT / "data" / "raw" / "jangsu" / "suitability.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "vespa_fields_500m"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "vespa_fields_500m.csv"
OUTPUT_FIGURE = OUTPUT_DIR / "vespa_fields_500m.png"
OUTPUT_META = OUTPUT_DIR / "meta.json"

INPUT_CRS = "EPSG:4326"
OUTPUT_CRS = "EPSG:5179"
GRID_SIZE = 500.0

# density.csv is a ~0.01 deg lon/lat grid, i.e. an anisotropic ~870-930 m
# (lon) x ~1113 m (lat) rectangle depending on latitude -- its worst-case
# half-diagonal (the distance from a source point to the farthest corner
# of its own cell) is ~725 m. A target cell counts as land if a raw
# density sample exists within this radius; using a value at or below
# that half-diagonal (e.g. the previous 650 m) leaves gaps at the corners
# between source points, which shows up as a periodic lattice of false
# "sea" holes punched through solid land. 750 m clears that with a small
# margin while staying far below the width of any real strait/bay, so it
# still won't bridge genuine sea gaps.
LAND_SUPPORT_RADIUS_M = 750.0


# ============================================================
# Loading
# ============================================================

def load_points(csv_path: Path, value_col: str, transformer: Transformer):
    """Read a lon/lat, value CSV and reproject it to the working CRS."""

    df = pd.read_csv(csv_path)

    x, y = transformer.transform(
        df["X"].to_numpy(),
        df["Y"].to_numpy(),
    )

    values = df[value_col].to_numpy(dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)

    return x[valid], y[valid], values[valid]


# ============================================================
# Target grid
# ============================================================

def build_target_grid(min_x, max_x, min_y, max_y, grid_size):
    """Build a regular grid of cell centers covering [min, max] x/y,
    aligned to grid_size boundaries (same convention as dem_downscale)."""

    origin_x = np.floor(min_x / grid_size) * grid_size
    origin_y = np.floor(min_y / grid_size) * grid_size

    nx = int(np.ceil((max_x - origin_x) / grid_size))
    ny = int(np.ceil((max_y - origin_y) / grid_size))

    x_centers = origin_x + (np.arange(nx) + 0.5) * grid_size
    y_centers = origin_y + (np.arange(ny) + 0.5) * grid_size

    grid_x, grid_y = np.meshgrid(x_centers, y_centers)

    return origin_x, origin_y, x_centers, y_centers, grid_x, grid_y


def land_support_mask(px, py, grid_x, grid_y, radius):
    """True where a target cell has a raw source sample within `radius`."""

    tree = cKDTree(np.column_stack([px, py]))
    targets = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    distance, _ = tree.query(targets, k=1)

    return (distance <= radius).reshape(grid_x.shape)


def interpolate_to_grid(px, py, values, grid_x, grid_y):
    """Linear interpolation of scattered (px, py, values) onto the grid.
    Cells outside the convex hull of the source points come back as NaN."""

    points = np.column_stack([px, py])
    targets = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    interpolated = griddata(points, values, targets, method="linear")

    return interpolated.reshape(grid_x.shape)


def fill_nan_nearest(field):
    """
    Nearest-neighbor fill of any remaining NaN cells.

    This is only to keep finite-difference gradient stencils well-defined
    everywhere (including outside the land mask); the model masks sea
    cells out of every term anyway, so the filled values themselves never
    reach the reported solution.
    """

    nan_mask = np.isnan(field)

    if not nan_mask.any():
        return field

    indices = distance_transform_edt(
        nan_mask,
        return_distances=False,
        return_indices=True,
    )

    return field[tuple(indices)]


# ============================================================
# Figure
# ============================================================

def make_figure(land_mask, C_u, S_v_normalized, x_centers, y_centers):

    extent = [
        x_centers[0] - GRID_SIZE / 2,
        x_centers[-1] + GRID_SIZE / 2,
        y_centers[0] - GRID_SIZE / 2,
        y_centers[-1] + GRID_SIZE / 2,
    ]

    C_u_masked = np.where(land_mask, C_u, np.nan)
    S_v_masked = np.where(land_mask, S_v_normalized, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(land_mask, origin="lower", extent=extent, cmap="gray")
    axes[0].set_title("Land mask")

    image_cu = axes[1].imshow(
        C_u_masked, origin="lower", extent=extent, cmap="viridis"
    )
    axes[1].set_title("$C_u(x)$ -- V. velutina carrying capacity")
    fig.colorbar(image_cu, ax=axes[1], fraction=0.046)

    image_sv = axes[2].imshow(
        S_v_masked, origin="lower", extent=extent, cmap="viridis", vmin=0, vmax=1
    )
    axes[2].set_title("$S_v(x)$ -- V. mandarinia suitability (normalized)")
    fig.colorbar(image_sv, ax=axes[2], fraction=0.046)

    for ax in axes:
        ax.set_xlabel("X (EPSG:5179, m)")
        ax.set_ylabel("Y (EPSG:5179, m)")
        ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE, dpi=200, bbox_inches="tight")

    print(f"\nFigure saved to:\n{OUTPUT_FIGURE}")


# ============================================================
# Main
# ============================================================

def main():

    print("============================================")
    print("Vespa competition-taxis model: field preprocessing")
    print("============================================")

    for path in (DENSITY_CSV, SUITABILITY_CSV):
        if not path.exists():
            raise FileNotFoundError(f"Input file not found:\n{path}")

    transformer = Transformer.from_crs(INPUT_CRS, OUTPUT_CRS, always_xy=True)

    print(f"\nLoading density (C_u):\n{DENSITY_CSV}")
    px_d, py_d, density = load_points(DENSITY_CSV, "density", transformer)
    print(f"  {len(density):,} points")

    print(f"\nLoading suitability (S_v):\n{SUITABILITY_CSV}")
    px_s, py_s, suitability = load_points(SUITABILITY_CSV, "suitability", transformer)
    print(f"  {len(suitability):,} points")

    # --------------------------------------------------------
    # Target grid: union of both sources' extents.
    # --------------------------------------------------------

    min_x = min(px_d.min(), px_s.min())
    max_x = max(px_d.max(), px_s.max())
    min_y = min(py_d.min(), py_s.min())
    max_y = max(py_d.max(), py_s.max())

    origin_x, origin_y, x_centers, y_centers, grid_x, grid_y = build_target_grid(
        min_x, max_x, min_y, max_y, GRID_SIZE
    )

    print(f"\nTarget grid shape (ny, nx): {grid_x.shape}")
    print(f"Grid origin: ({origin_x:.1f}, {origin_y:.1f})")

    # --------------------------------------------------------
    # Land mask, from density.csv's own coverage (the finer, and the
    # species-specific, SDM raster -- see LAND_SUPPORT_RADIUS_M).
    # --------------------------------------------------------

    land_mask = land_support_mask(px_d, py_d, grid_x, grid_y, LAND_SUPPORT_RADIUS_M)

    print(
        f"\nLand cells: {land_mask.sum():,} / {land_mask.size:,} "
        f"({100.0 * land_mask.mean():.1f}%)"
    )

    # --------------------------------------------------------
    # C_u(x): carrying capacity, used as-is (raw units).
    # --------------------------------------------------------

    C_u = interpolate_to_grid(px_d, py_d, density, grid_x, grid_y)
    C_u = fill_nan_nearest(C_u)

    # --------------------------------------------------------
    # S_v(x): suitability, min-max normalized to [0, 1] over land cells.
    # --------------------------------------------------------

    S_v_raw = interpolate_to_grid(px_s, py_s, suitability, grid_x, grid_y)
    S_v_raw = fill_nan_nearest(S_v_raw)

    s_min = float(S_v_raw[land_mask].min())
    s_max = float(S_v_raw[land_mask].max())

    S_v_normalized = np.clip((S_v_raw - s_min) / (s_max - s_min), 0.0, 1.0)

    print(f"\nC_u range over land: {C_u[land_mask].min():.3f} - {C_u[land_mask].max():.3f}")
    print(f"S_v raw range over land: {s_min:.3f} - {s_max:.3f} (-> normalized to 0-1)")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(OUTPUT_DIR / "land_mask_500m.npy", land_mask)
    np.save(OUTPUT_DIR / "carrying_capacity_Cu_500m.npy", C_u.astype(np.float32))
    np.save(OUTPUT_DIR / "suitability_Sv_raw_500m.npy", S_v_raw.astype(np.float32))
    np.save(
        OUTPUT_DIR / "suitability_Sv_normalized_500m.npy",
        S_v_normalized.astype(np.float32),
    )
    np.save(OUTPUT_DIR / "grid_x_500m.npy", x_centers)
    np.save(OUTPUT_DIR / "grid_y_500m.npy", y_centers)

    grid_df = pd.DataFrame({
        "iy": np.repeat(np.arange(grid_x.shape[0]), grid_x.shape[1]),
        "ix": np.tile(np.arange(grid_x.shape[1]), grid_x.shape[0]),
        "x": grid_x.ravel(),
        "y": grid_y.ravel(),
        "land_mask": land_mask.ravel(),
        "carrying_capacity_Cu": C_u.ravel(),
        "suitability_Sv_raw": S_v_raw.ravel(),
        "suitability_Sv_normalized": S_v_normalized.ravel(),
    })
    grid_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nGrid CSV saved to:\n{OUTPUT_CSV}")

    meta = {
        "grid_size_m": GRID_SIZE,
        "crs": OUTPUT_CRS,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "shape_ny_nx": list(grid_x.shape),
        "land_support_radius_m": LAND_SUPPORT_RADIUS_M,
        "land_cells": int(land_mask.sum()),
        "suitability_normalization": {"min": s_min, "max": s_max},
        "sources": {
            "C_u": str(DENSITY_CSV.relative_to(PROJECT_ROOT)),
            "S_v": str(SUITABILITY_CSV.relative_to(PROJECT_ROOT)),
        },
    }

    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Metadata saved to:\n{OUTPUT_META}")

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    make_figure(land_mask, C_u, S_v_normalized, x_centers, y_centers)

    print("\n============================================")
    print("Finished.")
    print("============================================")


if __name__ == "__main__":
    main()
