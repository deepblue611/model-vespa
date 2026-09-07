import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from pyproj import Transformer

# Allow importing from src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.model.fisher_kpp import FisherKPP
from src.model.diffusion import terrain_diffusion
from src.solver.finite_difference import FiniteDifference2D


# ============================================================
# Configuration
# ============================================================

DEM_CSV = ROOT / "data" / "processed" / "dem_500m" / "korea_dem_500m.csv"
DEM_NPY = ROOT / "data" / "processed" / "dem_500m" / "korea_dem_500m.npy"

GRID = 500.0
INPUT_CRS = "EPSG:4326"
DEM_CRS = "EPSG:5179"

# South Korea bounding box (lon/lat), used to crop the peninsula-wide
# DEM down to a domain small enough for fast interactive playback.
SOUTH_KOREA_BBOX = {
    "lon_min": 125.0,
    "lon_max": 130.0,
    "lat_min": 33.0,
    "lat_max": 38.6,
}

# Busan port, the assumed introduction point.
BUSAN_LON = 129.0403
BUSAN_LAT = 35.1028

# Fisher-KPP parameters (placeholders -- tune to real spread data later)
# D_MAX is capped so that DT_DAYS below stays CFL-stable -- see the
# stability check in main().
D_MAX = 90000.0       # m^2/day, diffusion on flat terrain
D_MIN = 6000.0        # m^2/day, diffusion on very steep terrain
SLOPE_SCALE = 0.15   # slope (rise/run) at which D drops halfway
SLOPE_POWER = 1.5

R_GROWTH = 0.01       # per day
K_CAPACITY = 1.0      # normalized carrying capacity

INITIAL_RADIUS_M = 1000.0   # radius of the initial introduction, meters
INITIAL_DENSITY = K_CAPACITY

# Invasive-species spread is slow relative to diffusion's natural time
# scale, so we view it in coarse, human-meaningful steps.
DT_DAYS = 0.2                  # solver time step
TOTAL_TIME_DAYS = 365.0 * 15.0  # simulate 15 years, whole-country spread
CFL_SAFETY = 0.4

# Saving a frame every solver step over 15 years on a whole-country grid
# is hundreds of GB of RAM (steps * Ny * Nx * 8 bytes). Save on a coarser,
# human-meaningful cadence instead -- the viewer only needs enough frames
# for smooth playback, not one per solver step.
SAVE_INTERVAL_DAYS = 30.0


# ============================================================
# DEM loading + cropping
# ============================================================

def load_full_dem():
    """
    Load the peninsula-wide 500m DEM array plus the grid origin
    needed to convert projected (x, y) coordinates to array indices.
    """

    dem = np.load(DEM_NPY)

    df = pd.read_csv(
        DEM_CSV,
        usecols=["ix", "iy", "x", "y"],
        nrows=1
    )

    row0 = df.iloc[0]

    origin_x = row0["x"] - (row0["ix"] + 0.5) * GRID
    origin_y = row0["y"] - (row0["iy"] + 0.5) * GRID

    ix_min = int(pd.read_csv(DEM_CSV, usecols=["ix"])["ix"].min())
    iy_min = int(pd.read_csv(DEM_CSV, usecols=["iy"])["iy"].min())

    return dem, origin_x, origin_y, ix_min, iy_min


def to_grid_index(lon, lat, transformer, origin_x, origin_y):
    """Project (lon, lat) and return the (ix, iy) grid index."""

    px, py = transformer.transform(lon, lat)

    ix = int(np.floor((px - origin_x) / GRID))
    iy = int(np.floor((py - origin_y) / GRID))

    return ix, iy


def crop_to_south_korea(dem, origin_x, origin_y, ix_min, iy_min):
    """
    Crop the peninsula-wide DEM down to the South Korea bounding box.

    Returns the cropped array plus the row/col offsets needed to map
    full-grid indices into the cropped array.
    """

    transformer = Transformer.from_crs(
        INPUT_CRS, DEM_CRS, always_xy=True
    )

    lons = [
        SOUTH_KOREA_BBOX["lon_min"], SOUTH_KOREA_BBOX["lon_max"],
        SOUTH_KOREA_BBOX["lon_min"], SOUTH_KOREA_BBOX["lon_max"],
    ]
    lats = [
        SOUTH_KOREA_BBOX["lat_min"], SOUTH_KOREA_BBOX["lat_min"],
        SOUTH_KOREA_BBOX["lat_max"], SOUTH_KOREA_BBOX["lat_max"],
    ]

    corners_ix = []
    corners_iy = []

    for lon, lat in zip(lons, lats):
        ix, iy = to_grid_index(lon, lat, transformer, origin_x, origin_y)
        corners_ix.append(ix)
        corners_iy.append(iy)

    col_start = max(0, min(corners_ix) - ix_min)
    col_end = min(dem.shape[1], max(corners_ix) - ix_min + 1)

    row_start = max(0, min(corners_iy) - iy_min)
    row_end = min(dem.shape[0], max(corners_iy) - iy_min + 1)

    cropped = dem[row_start:row_end, col_start:col_end]

    return cropped, row_start, col_start, transformer


# ============================================================
# Initial condition
# ============================================================

def create_point_source(shape, center_row, center_col, radius_m, density):
    """Circular initial population centered on a single grid cell."""

    ny, nx = shape

    rows = np.arange(ny).reshape(-1, 1)
    cols = np.arange(nx).reshape(1, -1)

    distance = np.sqrt(
        ((rows - center_row) * GRID) ** 2
        + ((cols - center_col) * GRID) ** 2
    )

    u0 = np.zeros(shape)
    u0[distance <= radius_m] = density

    return u0


# ============================================================
# Visualization
# ============================================================

def run_interactive_viewer(dem, x_axis, y_axis, times, solutions, busan_xy):

    extent = [
        x_axis[0] - GRID / 2, x_axis[-1] + GRID / 2,
        y_axis[0] - GRID / 2, y_axis[-1] + GRID / 2,
    ]

    fig, ax = plt.subplots(figsize=(9, 9))
    plt.subplots_adjust(bottom=0.15)

    ax.imshow(
        dem,
        origin="lower",
        extent=extent,
        cmap="gray",
        aspect="equal",
        alpha=0.6,
    )

    population_layer = ax.imshow(
        solutions[0],
        origin="lower",
        extent=extent,
        cmap="inferno",
        aspect="equal",
        alpha=0.75,
        vmin=0.0,
        vmax=K_CAPACITY,
    )

    ax.plot(
        busan_xy[0], busan_xy[1],
        marker="x", color="cyan", markersize=10, markeredgewidth=2,
        label="Busan Port",
    )
    ax.legend(loc="upper right")

    ax.set_xlabel("X (EPSG:5179, m)")
    ax.set_ylabel("Y (EPSG:5179, m)")

    fig.colorbar(population_layer, ax=ax, label="Population density")

    title = ax.set_title(f"t = {times[0] / 365.0:.2f} years")

    slider_ax = fig.add_axes([0.2, 0.03, 0.6, 0.03])
    slider = Slider(
        slider_ax, "Frame", 0, len(times) - 1,
        valinit=0, valstep=1,
    )

    def on_change(val):
        index = int(slider.val)
        population_layer.set_data(solutions[index])
        title.set_text(f"t = {times[index] / 365.0:.2f} years")
        fig.canvas.draw_idle()

    slider.on_changed(on_change)

    plt.show()


# ============================================================
# Main
# ============================================================

def main():

    print("Loading DEM...")
    dem_full, origin_x, origin_y, ix_min, iy_min = load_full_dem()

    print("Cropping to South Korea...")
    dem, row_start, col_start, transformer = crop_to_south_korea(
        dem_full, origin_x, origin_y, ix_min, iy_min
    )
    print(f"Cropped grid shape: {dem.shape}")

    x_axis = origin_x + (
        np.arange(dem.shape[1]) + col_start + ix_min + 0.5
    ) * GRID
    y_axis = origin_y + (
        np.arange(dem.shape[0]) + row_start + iy_min + 0.5
    ) * GRID

    # Busan location within the cropped grid
    busan_ix, busan_iy = to_grid_index(
        BUSAN_LON, BUSAN_LAT, transformer, origin_x, origin_y
    )
    busan_row = busan_iy - iy_min - row_start
    busan_col = busan_ix - ix_min - col_start

    if not (0 <= busan_row < dem.shape[0] and 0 <= busan_col < dem.shape[1]):
        raise ValueError("Busan is outside the cropped domain.")

    busan_x = x_axis[busan_col]
    busan_y = y_axis[busan_row]
    print(f"Busan grid cell: row={busan_row}, col={busan_col}")

    # Diffusion field from terrain slope
    D_field = terrain_diffusion(
        dem, dx=GRID, dy=GRID,
        D_max=D_MAX, D_min=D_MIN,
        slope_scale=SLOPE_SCALE, power=SLOPE_POWER,
    )

    # Initial condition: point introduction at Busan
    u0 = create_point_source(
        dem.shape, busan_row, busan_col,
        radius_m=INITIAL_RADIUS_M, density=INITIAL_DENSITY,
    )

    # dt is fixed to the desired 15-day observation cadence; verify it
    # is still CFL-stable for the current diffusion field before running.
    dt = DT_DAYS
    dt_stability_limit = GRID ** 2 / (4.0 * D_field.max())

    if dt > CFL_SAFETY * dt_stability_limit:
        raise ValueError(
            f"dt={dt:.2f} days is not CFL-stable for D_max={D_field.max():.1f} "
            f"m^2/day (raw stability limit ~{dt_stability_limit:.2f} days, "
            f"safety-scaled limit ~{CFL_SAFETY * dt_stability_limit:.2f} days). "
            f"Lower D_MAX or DT_DAYS."
        )

    steps = int(TOTAL_TIME_DAYS / dt)

    # Save on SAVE_INTERVAL_DAYS cadence, not every solver step -- see
    # the note next to SAVE_INTERVAL_DAYS for why.
    save_every = max(1, round(SAVE_INTERVAL_DAYS / dt))
    n_frames = steps // save_every + 1
    frame_bytes = dem.shape[0] * dem.shape[1] * 8
    print(
        f"dt = {dt:.1f} days, steps = {steps}, save_every = {save_every} "
        f"({save_every * dt:.1f} days/frame), ~{n_frames} frames "
        f"(~{n_frames * frame_bytes / 1e9:.2f} GB)"
    )

    model = FisherKPP(D=D_field, r=R_GROWTH, K=K_CAPACITY)
    solver = FiniteDifference2D(dx=GRID, dy=GRID, dt=dt)

    print("Running simulation...")
    start = time.perf_counter()
    times, solutions = solver.solve(
        u0=u0, model=model, steps=steps, save_every=save_every, progress=True
    )
    elapsed = time.perf_counter() - start
    print(f"Done in {elapsed:.2f}s, saved {len(times)} frames.")

    run_interactive_viewer(
        dem, x_axis, y_axis, times, solutions, (busan_x, busan_y)
    )


if __name__ == "__main__":
    main()
