"""
M3: Vespa velutina (invasive) vs. Vespa mandarinia (native) competition
model with environmental-capacity taxis, on the real South Korea 500 m
grid produced by src/preprocessing/vespa_field_downscale.py.

See vespa_pde_model_spec.md for the governing equation. Parameter values
below are placeholders (section 7 of the spec flags them as TBD) -- tune
to real spread data once available.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.model.vespa_competition import VespaCompetition
from src.solver.keller_segel import KellerSegel2D, stable_dt
from src.experiment_utils import (
    load_vespa_fields, to_grid_index, nearest_land_cell,
    create_point_source, ensure_gpu_interpreter,
)

# ============================================================
# Configuration
# ============================================================

DATA_DIR = ROOT / "data" / "processed" / "vespa_fields_500m"
OUTPUT_DIR = ROOT / "outputs" / "m3_vespa_competition"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The literature D_U below makes the CFL-limited dt tiny at native 500m
# resolution (~452k steps). GRID_FACTOR coarsens the loaded 500m fields
# in-memory by block-averaging (cuts cell count and raises stable dt at
# once); BACKEND picks the array module the solve loop runs on. Both are
# independent knobs; benchmarked on this machine (whole-country, 20yr):
#   GRID_FACTOR=1 (500m), BACKEND="cpu": ~452k steps, ~28 hr
#   GRID_FACTOR=2 (1km),  BACKEND="cpu": ~113k steps, ~4 hr
#   GRID_FACTOR=4 (2km),  BACKEND="cpu": ~28k steps,  ~5.5 min  (sweeps)
#   GRID_FACTOR=1 (500m), BACKEND="gpu": ~452k steps, ~46 min  (~36x/step)
#
# BACKEND="gpu" requires cupy, and -- because of a CuPy/NVRTC limitation
# with non-ASCII paths on Windows -- must be run with a Python interpreter
# installed at an ASCII-only path (this project's own .venv lives under
# a Korean-named folder, which breaks CuPy's kernel compilation). Use a
# separate venv elsewhere, e.g. `C:\gpuvenv`, with cupy-cuda12x[ctk],
# matplotlib and pyproj installed, and run this script with that
# interpreter instead of the project .venv when BACKEND="gpu".
GRID_FACTOR = 2
GRID = 500.0 * GRID_FACTOR
LON_LAT_CRS = "EPSG:4326"
WORKING_CRS = "EPSG:5179"

BACKEND = "gpu"   # "cpu" or "gpu" -- see note above

# BACKEND="gpu" only works under the ASCII-path GPU venv's interpreter
# (see note above) -- if we're not already running under it, relaunch
# this same script there instead of failing on `import cupy`.
GPU_PYTHON = r"C:\gpuvenv\Scripts\python.exe"

if BACKEND == "gpu":
    ensure_gpu_interpreter(GPU_PYTHON)

if BACKEND == "gpu":
    import cupy as xp
else:
    xp = np

# Busan port, the assumed introduction point (real historical introductions
# of V. velutina in Korea trace back to Busan).
BUSAN_LON = 129.0403
BUSAN_LAT = 35.1028

# --- Model parameters ---
# D_U, ALPHA, K_V are literature-derived; CHI_U and BETA are still
# placeholders (see spec section 7 -- no literature value identified yet).
#
# Pure Fisher-KPP front speed sqrt(4*D_U*ALPHA) ~= 68 m/day (~25 km/year)
# with the values below -- taxis adds further, non-uniform speedup on
# top of this.
D_U = 1_510_190.0   # m^2/day, random-walk diffusion (literature)
CHI_U = 1e5*5    # m^2/day per unit of C_u, taxis sensitivity (placeholder)
ALPHA = 0.00077041   # per day, intrinsic logistic growth rate (literature)
BETA = 0.3 * ALPHA   # per day per unit v, competition strength (placeholder)
K_V = 1.0         # scales normalized suitability into a native density (literature)

INITIAL_RADIUS_M = 500.0   # ~1 grid cell: a founding introduction is
                            # effectively a point, not a population that
                            # already covers multiple km^2
INITIAL_DENSITY_FRACTION = 0.01   # seed density as a fraction of local
    # C_u at Busan, not an absolute constant -- C_u's absolute unit/scale
    # is unconfirmed (spec section 7), so a fixed number here would be
    # meaningless without knowing what density.csv's values represent

DT_SAFETY = 0.4
# ~18 years minimum to cross the ~450 km Busan-to-far-corner span at the
# front speed above (taxis, once CHI_U is set, should only shorten this);
# add margin since that speed ignores taxis.
TOTAL_TIME_DAYS = 365.0 * 23.0
SAVE_INTERVAL_DAYS = 90.0

# The spread boundary is drawn where u crosses this fraction of the final
# frame's own max density -- NOT a fraction of local carrying capacity
# C_u. With these literature D_U/ALPHA values, u can stay many orders of
# magnitude below C_u for a long time (the population diffuses across
# most of the country well before it saturates locally anywhere -- see
# the m3_chi_beta_sweep.py findings), so a C_u-relative "established"
# threshold stays empty for a very long transient. A max-relative
# threshold instead always traces the actual spread pattern, whatever
# its absolute scale.
SPREAD_BOUNDARY_FRACTION_OF_MAX = 0.05
HOTSPOT_TOP_N = 15   # densest land cells to mark


# ============================================================
# Visualization
# ============================================================

def draw_spread_overlay(ax, land_mask, x_centers, y_centers, u, linewidth=1.5, marker_size=30):
    """
    Draw the spread boundary contour (u crosses
    SPREAD_BOUNDARY_FRACTION_OF_MAX * max(u) for this frame) and the
    densest cells as hotspot markers onto an existing axes. Shared by
    save_snapshot_figure (one frame per panel) and
    save_spread_analysis_figure (final frame only).
    """

    land_u = np.where(land_mask, u, 0.0)
    max_density = float(np.nanmax(land_u))

    if max_density <= 0:
        return

    boundary_level = SPREAD_BOUNDARY_FRACTION_OF_MAX * max_density
    X, Y = np.meshgrid(x_centers, y_centers)
    ax.contour(X, Y, land_u, levels=[boundary_level], colors="cyan", linewidths=linewidth)

    n_hotspots = min(HOTSPOT_TOP_N, int(land_mask.sum()))
    density_land_only = np.where(land_mask, u, -np.inf)
    flat_indices = np.argpartition(density_land_only.ravel(), -n_hotspots)[-n_hotspots:]
    rows, cols = np.unravel_index(flat_indices, u.shape)

    ax.scatter(
        x_centers[cols], y_centers[rows],
        marker="^", color="lime", edgecolors="black", s=marker_size, zorder=5,
    )


def compute_spread_area_km2(land_mask, u, cell_area_km2):
    """
    Area (km^2) of cells where u crosses SPREAD_BOUNDARY_FRACTION_OF_MAX *
    that frame's own max density -- same boundary definition as
    draw_spread_overlay, so the area matches what the cyan contour encloses.
    """

    land_u = np.where(land_mask, u, 0.0)
    max_density = float(np.nanmax(land_u))

    if max_density <= 0:
        return 0.0

    boundary_level = SPREAD_BOUNDARY_FRACTION_OF_MAX * max_density
    return float((land_u >= boundary_level).sum()) * cell_area_km2


def compute_max_spread_distance_km(land_mask, u, busan_row, busan_col, grid_m):
    """
    Max straight-line distance (km) from Busan to any cell crossing the
    same SPREAD_BOUNDARY_FRACTION_OF_MAX boundary used elsewhere.

    This is the actual front reach, unlike sqrt(area/pi): Busan sits at a
    coastal corner of the domain, so a lot of the "circle" an
    area-equivalent radius assumes is ocean and gets cut off by the land
    mask -- that systematically underestimates true reach and is why this
    replaced the area-based radius for front-speed estimates.
    """

    land_u = np.where(land_mask, u, 0.0)
    max_density = float(np.nanmax(land_u))

    if max_density <= 0:
        return 0.0

    boundary_level = SPREAD_BOUNDARY_FRACTION_OF_MAX * max_density
    rows, cols = np.nonzero(land_u >= boundary_level)

    if len(rows) == 0:
        return 0.0

    distances_km = np.sqrt(
        ((rows - busan_row) * grid_m) ** 2 + ((cols - busan_col) * grid_m) ** 2
    ) / 1000.0

    return float(distances_km.max())


def save_snapshot_figure(land_mask, x_centers, y_centers, times, solutions, busan_xy):
    """Save a static multi-panel figure of the run (does not require a
    display), so the simulation output is inspectable without a GUI.
    Each panel also gets the spread boundary + density hotspot overlay
    (see draw_spread_overlay), scaled to that panel's own frame."""

    extent = [
        x_centers[0] - GRID / 2, x_centers[-1] + GRID / 2,
        y_centers[0] - GRID / 2, y_centers[-1] + GRID / 2,
    ]

    indices = sorted({round(f) for f in np.linspace(0, len(times) - 1, 4)})

    vmax = max(np.nanmax(solutions[-1]), 1e-6)

    fig, axes = plt.subplots(1, len(indices), figsize=(5 * len(indices), 6))
    if len(indices) == 1:
        axes = [axes]

    for ax, index in zip(axes, indices):
        ax.imshow(land_mask, origin="lower", extent=extent, cmap="gray", alpha=0.4)

        image = ax.imshow(
            np.where(land_mask, solutions[index], np.nan),
            origin="lower", extent=extent, cmap="inferno",
            vmin=0.0, vmax=vmax,
        )
        draw_spread_overlay(ax, land_mask, x_centers, y_centers, solutions[index])
        ax.plot(
            busan_xy[0], busan_xy[1],
            marker="x", color="white", markersize=8, markeredgewidth=2,
        )
        ax.set_title(f"t = {times[index] / 365.0:.2f} years")
        ax.set_aspect("equal")

    fig.colorbar(image, ax=axes, label="Population density (u)", shrink=0.8)
    fig.suptitle(
        f"Cyan = spread boundary (u >= {SPREAD_BOUNDARY_FRACTION_OF_MAX:.0%} of that frame's max), "
        f"green = density hotspots"
    )

    output_path = OUTPUT_DIR / "snapshots.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSnapshot figure saved to:\n{output_path}")


def save_spread_analysis_figure(land_mask, x_centers, y_centers, final_u, busan_xy):
    """
    Save a figure of the final-frame density with the spreading front
    overlaid as a contour (cells where u crosses
    SPREAD_BOUNDARY_FRACTION_OF_MAX * max(u)) and the densest cells
    marked as hotspots. See the note by SPREAD_BOUNDARY_FRACTION_OF_MAX
    for why this is relative to the run's own max rather than to C_u.
    """

    extent = [
        x_centers[0] - GRID / 2, x_centers[-1] + GRID / 2,
        y_centers[0] - GRID / 2, y_centers[-1] + GRID / 2,
    ]

    land_u = np.where(land_mask, final_u, 0.0)
    max_density = float(np.nanmax(land_u))
    vmax = max(max_density, 1e-12)

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(land_mask, origin="lower", extent=extent, cmap="gray", alpha=0.4)

    image = ax.imshow(
        np.where(land_mask, final_u, np.nan),
        origin="lower", extent=extent, cmap="inferno",
        vmin=0.0, vmax=vmax,
    )

    if max_density > 0:
        draw_spread_overlay(ax, land_mask, x_centers, y_centers, final_u, linewidth=2, marker_size=60)
    else:
        print("Final density is zero everywhere -- no boundary/hotspots to draw.")

    ax.plot(
        busan_xy[0], busan_xy[1],
        marker="x", color="white", markersize=10, markeredgewidth=2,
        label="Busan Port",
    )

    ax.set_xlabel("X (EPSG:5179, m)")
    ax.set_ylabel("Y (EPSG:5179, m)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.colorbar(image, ax=ax, label="Population density (u)")
    ax.set_title(
        f"Spread boundary (u >= {SPREAD_BOUNDARY_FRACTION_OF_MAX:.0%} of max density) "
        f"and density hotspots"
    )

    output_path = OUTPUT_DIR / "spread_analysis.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Spread analysis figure saved to:\n{output_path}")


def run_interactive_viewer(land_mask, x_centers, y_centers, times, solutions, busan_xy):

    extent = [
        x_centers[0] - GRID / 2, x_centers[-1] + GRID / 2,
        y_centers[0] - GRID / 2, y_centers[-1] + GRID / 2,
    ]

    fig, ax = plt.subplots(figsize=(8, 9))
    plt.subplots_adjust(bottom=0.15)

    ax.imshow(land_mask, origin="lower", extent=extent, cmap="gray", aspect="equal", alpha=0.4)

    vmax = max(np.nanmax(solutions[-1]), 1e-6)

    population_layer = ax.imshow(
        np.where(land_mask, solutions[0], np.nan),
        origin="lower", extent=extent, cmap="inferno", aspect="equal",
        vmin=0.0, vmax=vmax,
    )

    ax.plot(
        busan_xy[0], busan_xy[1],
        marker="x", color="cyan", markersize=10, markeredgewidth=2,
        label="Busan Port",
    )
    ax.legend(loc="upper right")

    ax.set_xlabel("X (EPSG:5179, m)")
    ax.set_ylabel("Y (EPSG:5179, m)")
    fig.colorbar(population_layer, ax=ax, label="Population density (u)")

    title = ax.set_title(f"t = {times[0] / 365.0:.2f} years")

    slider_ax = fig.add_axes([0.2, 0.03, 0.6, 0.03])
    slider = Slider(slider_ax, "Frame", 0, len(times) - 1, valinit=0, valstep=1)

    def on_change(val):
        index = int(slider.val)
        population_layer.set_data(np.where(land_mask, solutions[index], np.nan))
        title.set_text(f"t = {times[index] / 365.0:.2f} years")
        fig.canvas.draw_idle()

    slider.on_changed(on_change)

    plt.show()


# ============================================================
# Main
# ============================================================

def main():

    print("Loading preprocessed fields...")
    land_mask, C_u, S_v, x_centers, y_centers = load_vespa_fields(DATA_DIR, GRID_FACTOR)
    print(f"Grid shape: {land_mask.shape}")

    transformer = Transformer.from_crs(LON_LAT_CRS, WORKING_CRS, always_xy=True)
    busan_row, busan_col = to_grid_index(BUSAN_LON, BUSAN_LAT, transformer, x_centers, y_centers)
    busan_row, busan_col = nearest_land_cell(land_mask, busan_row, busan_col)
    busan_x, busan_y = x_centers[busan_col], y_centers[busan_row]
    print(f"Busan grid cell: row={busan_row}, col={busan_col}")

    # Fields are loaded/indexed as plain numpy above (cheap, one-off);
    # only the hot solve loop needs to live on the GPU when BACKEND="gpu".
    land_mask_xp = xp.asarray(land_mask) if BACKEND == "gpu" else land_mask
    C_u_xp = xp.asarray(C_u) if BACKEND == "gpu" else C_u
    S_v_xp = xp.asarray(S_v) if BACKEND == "gpu" else S_v

    model = VespaCompetition(
        D_u=D_U, chi_u=CHI_U, alpha=ALPHA, beta=BETA,
        C_u=C_u_xp, K_v=K_V, S_v=S_v_xp, mask=land_mask_xp, xp=xp,
    )

    # Static taxis drift field chi_u * grad(C_u), computed once.
    grid_geometry = KellerSegel2D(dx=GRID, dy=GRID, dt=1.0, xp=xp)
    velocity_x, velocity_y = grid_geometry.taxis_velocity(model.C_u, CHI_U)

    dt_limit = stable_dt(D_U, velocity_x, velocity_y, GRID, GRID, xp=xp)
    dt = DT_SAFETY * dt_limit
    solver = KellerSegel2D(dx=GRID, dy=GRID, dt=dt, xp=xp)

    steps = int(TOTAL_TIME_DAYS / dt)
    save_every = max(1, round(SAVE_INTERVAL_DAYS / dt))
    n_frames = steps // save_every + 1
    print(
        f"dt = {dt:.3f} days (CFL limit {dt_limit:.3f}), steps = {steps}, "
        f"save_every = {save_every} ({save_every * dt:.1f} days/frame), "
        f"~{n_frames} frames"
    )

    initial_density = INITIAL_DENSITY_FRACTION * C_u[busan_row, busan_col]
    u0 = create_point_source(
        land_mask.shape, busan_row, busan_col,
        radius_m=INITIAL_RADIUS_M, density=initial_density, grid=GRID,
    )
    u0 = np.where(land_mask, u0, 0.0)
    if BACKEND == "gpu":
        u0 = xp.asarray(u0)

    print("Running simulation...")
    start = time.perf_counter()
    times, solutions = solver.solve(
        u0=u0, model=model, velocity_x=velocity_x, velocity_y=velocity_y,
        steps=steps, save_every=save_every, progress=True,
    )
    elapsed = time.perf_counter() - start
    print(f"Done in {elapsed:.2f}s, saved {len(times)} frames.")

    total_population = solutions[:, land_mask].sum(axis=1)

    cell_area_km2 = (GRID / 1000.0) ** 2
    spread_area_km2 = np.array([
        compute_spread_area_km2(land_mask, solutions[i], cell_area_km2)
        for i in range(len(times))
    ])
    spread_max_distance_km = np.array([
        compute_max_spread_distance_km(land_mask, solutions[i], busan_row, busan_col, GRID)
        for i in range(len(times))
    ])

    timeseries_path = OUTPUT_DIR / "timeseries.npz"
    np.savez(
        timeseries_path, times=times, total_population=total_population,
        spread_area_km2=spread_area_km2,
        spread_max_distance_km=spread_max_distance_km,
        spread_boundary_fraction_of_max=SPREAD_BOUNDARY_FRACTION_OF_MAX,
    )
    print(f"Time series saved to:\n{timeseries_path}")

    save_snapshot_figure(land_mask, x_centers, y_centers, times, solutions, (busan_x, busan_y))
    save_spread_analysis_figure(land_mask, x_centers, y_centers, solutions[-1], (busan_x, busan_y))
    run_interactive_viewer(land_mask, x_centers, y_centers, times, solutions, (busan_x, busan_y))


if __name__ == "__main__":
    main()
