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

GRID = 500.0
LON_LAT_CRS = "EPSG:4326"
WORKING_CRS = "EPSG:5179"

# Busan port, the assumed introduction point (real historical introductions
# of V. velutina in Korea trace back to Busan).
BUSAN_LON = 129.0403
BUSAN_LAT = 35.1028

# --- Model parameters (placeholders -- see spec section 7) ---
# D_U and ALPHA are sized so the pure Fisher-KPP front speed
# sqrt(4*D_U*ALPHA) ~= 122 m/day (~45 km/year), enough to cross South
# Korea's ~450 km Busan-to-far-corner span within the TOTAL_TIME_DAYS
# below (taxis adds further, non-uniform speedup on top of this).
D_U = 75000.0     # m^2/day, random-walk diffusion
CHI_U = 5.0e4     # m^2/day per unit of C_u, taxis sensitivity
ALPHA = 0.05      # per day, intrinsic logistic growth rate
BETA = 0.01       # per day per unit v, competition strength
K_V = 1.0         # scales normalized suitability into a native density

INITIAL_RADIUS_M = 1000.0
INITIAL_DENSITY = 0.5   # seed density at the introduction point

DT_SAFETY = 0.4
TOTAL_TIME_DAYS = 365.0 * 10.0
SAVE_INTERVAL_DAYS = 90.0


# ============================================================
# Data loading
# ============================================================

def load_fields():
    land_mask = np.load(DATA_DIR / "land_mask_500m.npy")
    C_u = np.load(DATA_DIR / "carrying_capacity_Cu_500m.npy").astype(float)
    S_v = np.load(DATA_DIR / "suitability_Sv_normalized_500m.npy").astype(float)
    x_centers = np.load(DATA_DIR / "grid_x_500m.npy")
    y_centers = np.load(DATA_DIR / "grid_y_500m.npy")

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

    model = VespaCompetition(
        D_u=D_U, chi_u=CHI_U, alpha=ALPHA, beta=BETA,
        C_u=C_u, K_v=K_V, S_v=S_v, mask=land_mask,
    )

    # Static taxis drift field chi_u * grad(C_u), computed once.
    grid_geometry = KellerSegel2D(dx=GRID, dy=GRID, dt=1.0)
    velocity_x, velocity_y = grid_geometry.taxis_velocity(model.C_u, CHI_U)

    dt_limit = stable_dt(D_U, velocity_x, velocity_y, GRID, GRID)
    dt = DT_SAFETY * dt_limit
    solver = KellerSegel2D(dx=GRID, dy=GRID, dt=dt)

    steps = int(TOTAL_TIME_DAYS / dt)
    save_every = max(1, round(SAVE_INTERVAL_DAYS / dt))
    n_frames = steps // save_every + 1
    print(
        f"dt = {dt:.3f} days (CFL limit {dt_limit:.3f}), steps = {steps}, "
        f"save_every = {save_every} ({save_every * dt:.1f} days/frame), "
        f"~{n_frames} frames"
    )

    u0 = create_point_source(
        land_mask.shape, busan_row, busan_col,
        radius_m=INITIAL_RADIUS_M, density=INITIAL_DENSITY,
    )
    u0 = np.where(land_mask, u0, 0.0)

    print("Running simulation...")
    start = time.perf_counter()
    times, solutions = solver.solve(
        u0=u0, model=model, velocity_x=velocity_x, velocity_y=velocity_y,
        steps=steps, save_every=save_every, progress=True,
    )
    elapsed = time.perf_counter() - start
    print(f"Done in {elapsed:.2f}s, saved {len(times)} frames.")

    save_snapshot_figure(land_mask, x_centers, y_centers, times, solutions, (busan_x, busan_y))
    run_interactive_viewer(land_mask, x_centers, y_centers, times, solutions, (busan_x, busan_y))


if __name__ == "__main__":
    main()
