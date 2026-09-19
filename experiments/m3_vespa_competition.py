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

if BACKEND == "gpu" and Path(sys.executable).resolve() != Path(GPU_PYTHON).resolve():
    import subprocess

    if not Path(GPU_PYTHON).exists():
        raise RuntimeError(
            f"BACKEND='gpu' but the GPU venv interpreter was not found at "
            f"{GPU_PYTHON} -- see the note above for how to set it up."
        )

    print(f"BACKEND='gpu': relaunching under {GPU_PYTHON} ...")
    result = subprocess.run([GPU_PYTHON, __file__, *sys.argv[1:]])
    sys.exit(result.returncode)

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
CHI_U = 1e5    # m^2/day per unit of C_u, taxis sensitivity (placeholder)
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
TOTAL_TIME_DAYS = 365.0 * 20.0
SAVE_INTERVAL_DAYS = 90.0


# ============================================================
# Data loading
# ============================================================

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


def load_fields():
    land_mask = np.load(DATA_DIR / "land_mask_500m.npy")
    C_u = np.load(DATA_DIR / "carrying_capacity_Cu_500m.npy").astype(float)
    S_v = np.load(DATA_DIR / "suitability_Sv_normalized_500m.npy").astype(float)
    x_centers = np.load(DATA_DIR / "grid_x_500m.npy")
    y_centers = np.load(DATA_DIR / "grid_y_500m.npy")

    if GRID_FACTOR > 1:
        land_mask, C_u, S_v = block_reduce_fields(land_mask, C_u, S_v, GRID_FACTOR)
        x_centers = block_reduce_1d(x_centers, GRID_FACTOR)
        y_centers = block_reduce_1d(y_centers, GRID_FACTOR)

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

def save_snapshot_figure(land_mask, x_centers, y_centers, times, solutions, busan_xy):
    """Save a static multi-panel figure of the run (does not require a
    display), so the simulation output is inspectable without a GUI."""

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
        ax.plot(
            busan_xy[0], busan_xy[1],
            marker="x", color="cyan", markersize=8, markeredgewidth=2,
        )
        ax.set_title(f"t = {times[index] / 365.0:.2f} years")
        ax.set_aspect("equal")

    fig.colorbar(image, ax=axes, label="Population density (u)", shrink=0.8)

    output_path = OUTPUT_DIR / "snapshots.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSnapshot figure saved to:\n{output_path}")


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
    land_mask, C_u, S_v, x_centers, y_centers = load_fields()
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
        radius_m=INITIAL_RADIUS_M, density=initial_density,
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
    timeseries_path = OUTPUT_DIR / "timeseries.npz"
    np.savez(timeseries_path, times=times, total_population=total_population)
    print(f"Time series saved to:\n{timeseries_path}")

    save_snapshot_figure(land_mask, x_centers, y_centers, times, solutions, (busan_x, busan_y))
    run_interactive_viewer(land_mask, x_centers, y_centers, times, solutions, (busan_x, busan_y))


if __name__ == "__main__":
    main()
