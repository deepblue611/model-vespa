"""
M3: sweep CHI_U (taxis sensitivity) x BETA (competition strength) and map
out the qualitative behavior of the invasion across that 2D parameter
space -- both are still placeholders (spec section 7), so this is meant
to show which regimes ("dies out near Busan" / "persists locally" /
"spreads across the country") each combination falls into, not to pick
a final value.

Runs at a coarser grid (GRID_FACTOR) than the main m3_vespa_competition.py
run so the whole sweep finishes in minutes rather than hours -- see the
runtime note in that file for the GRID_FACTOR/BACKEND tradeoffs, which
apply here identically.
"""

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
OUTPUT_DIR = ROOT / "outputs" / "m3_chi_beta_sweep"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRID_FACTOR = 4   # 2km -- see m3_vespa_competition.py's runtime note
GRID = 500.0 * GRID_FACTOR
LON_LAT_CRS = "EPSG:4326"
WORKING_CRS = "EPSG:5179"

BACKEND = "gpu"   # "cpu" or "gpu" -- see m3_vespa_competition.py's note
GPU_PYTHON = r"C:\gpuvenv\Scripts\python.exe"

if BACKEND == "gpu":
    ensure_gpu_interpreter(GPU_PYTHON)

if BACKEND == "gpu":
    import cupy as xp
else:
    xp = np

BUSAN_LON = 129.0403
BUSAN_LAT = 35.1028

# D_U, ALPHA, K_V are literature-derived and held fixed (see
# m3_vespa_competition.py); this sweep is specifically for the two
# placeholders. CHI_U is log-spaced -- with D_U this large, diffusion
# alone dominates transport up to chi_u ~1e6-1e7 (see prior analysis),
# so this range covers "negligible taxis" to "taxis starting to matter".
# BETA is expressed as a multiple of ALPHA (k), since an absolute BETA
# only makes sense relative to ALPHA's scale -- k=1 means growth and
# competition are balanced at S_v=1 (max native suitability).
D_U = 1_510_190.0
ALPHA = 0.00077041
K_V = 1.0

CHI_U_VALUES = [1e3, 1e4, 1e5, 1e6]
BETA_K_VALUES = [0.0, 0.5, 1.0, 2.0, 5.0]
BETA_VALUES = [k * ALPHA for k in BETA_K_VALUES]

INITIAL_RADIUS_M = 500.0
INITIAL_DENSITY_FRACTION = 0.01

DT_SAFETY = 0.4
TOTAL_TIME_DAYS = 365.0 * 20.0

# A cell counts as "established" for the spread-radius metric once its
# population reaches this fraction of the local carrying capacity.
ESTABLISHED_THRESHOLD = 0.01


def run_one(land_mask_np, land_mask_xp, C_u_xp, S_v_xp, C_u, busan_row, busan_col, u0_seed, chi_u, beta):
    model = VespaCompetition(
        D_u=D_U, chi_u=chi_u, alpha=ALPHA, beta=beta,
        C_u=C_u_xp, K_v=K_V, S_v=S_v_xp, mask=land_mask_xp, xp=xp,
    )

    geo = KellerSegel2D(dx=GRID, dy=GRID, dt=1.0, xp=xp)
    velocity_x, velocity_y = geo.taxis_velocity(model.C_u, chi_u)

    dt_limit = stable_dt(D_U, velocity_x, velocity_y, GRID, GRID, xp=xp)
    dt = DT_SAFETY * dt_limit
    solver = KellerSegel2D(dx=GRID, dy=GRID, dt=dt, xp=xp)

    steps = int(TOTAL_TIME_DAYS / dt)

    u0 = xp.asarray(u0_seed) if BACKEND == "gpu" else u0_seed

    # Only the final state is needed for the sweep's summary metrics, so
    # save just the first and last frame (save_every >= steps).
    times, solutions = solver.solve(
        u0=u0, model=model, velocity_x=velocity_x, velocity_y=velocity_y,
        steps=steps, save_every=steps,
    )

    # solver.solve() always returns plain numpy (KellerSegel2D converts
    # GPU arrays back to host at each saved frame), regardless of BACKEND.
    final_u = solutions[-1]
    total_population = float(final_u[land_mask_np].sum())

    established = (final_u >= ESTABLISHED_THRESHOLD * C_u) & land_mask_np
    if established.any():
        rows, cols = np.nonzero(established)
        distance_km = np.sqrt(
            ((rows - busan_row) * GRID) ** 2 + ((cols - busan_col) * GRID) ** 2
        ) / 1000.0
        spread_radius_km = float(distance_km.max())
    else:
        spread_radius_km = 0.0

    return steps, dt, total_population, spread_radius_km


def main():
    print("Loading preprocessed fields...")
    land_mask_np, C_u, S_v, x_centers, y_centers = load_vespa_fields(DATA_DIR, GRID_FACTOR)
    print(f"Grid shape: {land_mask_np.shape}")

    transformer = Transformer.from_crs(LON_LAT_CRS, WORKING_CRS, always_xy=True)
    busan_row, busan_col = to_grid_index(BUSAN_LON, BUSAN_LAT, transformer, x_centers, y_centers)
    busan_row, busan_col = nearest_land_cell(land_mask_np, busan_row, busan_col)
    print(f"Busan grid cell: row={busan_row}, col={busan_col}")

    land_mask_xp = xp.asarray(land_mask_np) if BACKEND == "gpu" else land_mask_np
    C_u_xp = xp.asarray(C_u) if BACKEND == "gpu" else C_u
    S_v_xp = xp.asarray(S_v) if BACKEND == "gpu" else S_v

    initial_density = INITIAL_DENSITY_FRACTION * C_u[busan_row, busan_col]
    u0_seed = create_point_source(
        land_mask_np.shape, busan_row, busan_col,
        radius_m=INITIAL_RADIUS_M, density=initial_density, grid=GRID,
    )
    u0_seed = np.where(land_mask_np, u0_seed, 0.0)

    n_combos = len(CHI_U_VALUES) * len(BETA_VALUES)
    total_population_grid = np.zeros((len(CHI_U_VALUES), len(BETA_VALUES)))
    spread_radius_grid = np.zeros((len(CHI_U_VALUES), len(BETA_VALUES)))

    print(f"Running {n_combos} combinations (GRID_FACTOR={GRID_FACTOR}, BACKEND={BACKEND})...")
    sweep_start = time.perf_counter()

    for i, chi_u in enumerate(CHI_U_VALUES):
        for j, beta in enumerate(BETA_VALUES):
            start = time.perf_counter()
            steps, dt, total_population, spread_radius_km = run_one(
                land_mask_np, land_mask_xp, C_u_xp, S_v_xp, C_u, busan_row, busan_col,
                u0_seed, chi_u, beta,
            )
            elapsed = time.perf_counter() - start

            total_population_grid[i, j] = total_population
            spread_radius_grid[i, j] = spread_radius_km

            print(
                f"  chi_u={chi_u:.0e} beta={beta:.2e} (k={BETA_K_VALUES[j]}): "
                f"steps={steps}, dt={dt:.3f}d, final_pop={total_population:.3g}, "
                f"spread={spread_radius_km:.1f}km  [{elapsed:.1f}s]"
            )

    sweep_elapsed = time.perf_counter() - sweep_start
    print(f"Sweep done in {sweep_elapsed:.1f}s.")

    results_path = OUTPUT_DIR / "sweep_results.npz"
    np.savez(
        results_path,
        chi_u_values=np.array(CHI_U_VALUES),
        beta_values=np.array(BETA_VALUES),
        beta_k_values=np.array(BETA_K_VALUES),
        total_population_grid=total_population_grid,
        spread_radius_grid=spread_radius_grid,
    )
    print(f"Results saved to:\n{results_path}")

    plot_results(total_population_grid, spread_radius_grid)


def plot_results(total_population_grid, spread_radius_grid):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    log_pop = np.log10(np.maximum(total_population_grid, 1e-12))
    im1 = ax1.imshow(log_pop, origin="lower", aspect="auto", cmap="viridis")
    ax1.set_title("Final total population (log10)")
    fig.colorbar(im1, ax=ax1, label="log10(total population)")

    im2 = ax2.imshow(spread_radius_grid, origin="lower", aspect="auto", cmap="magma")
    ax2.set_title("Spread radius from Busan (km)")
    fig.colorbar(im2, ax=ax2, label="km")

    for ax in (ax1, ax2):
        ax.set_xticks(range(len(BETA_K_VALUES)))
        ax.set_xticklabels([f"{k:g}" for k in BETA_K_VALUES])
        ax.set_xlabel("BETA / ALPHA (k)")
        ax.set_yticks(range(len(CHI_U_VALUES)))
        ax.set_yticklabels([f"{c:.0e}" for c in CHI_U_VALUES])
        ax.set_ylabel("CHI_U")

    fig.suptitle(f"CHI_U x BETA behavioral parameter space (GRID_FACTOR={GRID_FACTOR}, {TOTAL_TIME_DAYS/365:.0f}yr)")
    fig.tight_layout()

    output_path = OUTPUT_DIR / "parameter_space.png"
    fig.savefig(output_path, dpi=150)
    print(f"Figure saved to:\n{output_path}")


if __name__ == "__main__":
    main()
