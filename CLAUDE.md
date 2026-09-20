# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

The project has no packaging (`setup.py`/`pyproject.toml`) — everything runs as plain scripts against the `.venv` interpreter.

```bash
# Activate environment (Windows)
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run a single test
pytest tests/test_fisher_kpp.py::test_solution_shape
pytest tests/test_vespa_competition.py::test_no_flux_conserves_mass_with_no_reaction

# Lint / format
ruff check .
black .

# Run an experiment (must run from repo root, see Architecture)
python experiments/m1_2d_fisher.py
python experiments/m2_busan_terrain.py
python experiments/m3_vespa_competition.py
python experiments/m3_plot_population_timeseries.py   # reads m3's saved timeseries.npz

# Run the DEM preprocessing pipeline (M1/M2)
python src/preprocessing/dem_downscale.py

# Run the Vespa field preprocessing pipeline (M3)
python src/preprocessing/vespa_field_downscale.py
```

M3's two experiment scripts can also run on GPU — see **GPU backend** below; that path uses a *different* interpreter (`C:\gpuvenv\Scripts\python.exe`), not this project's `.venv`.

## Architecture

This is a 2D reaction-diffusion simulation project, developed in milestones (see git log: M0 → M1 → M1.1 → M2 → M2.1 → M3). It has two independent model tracks sharing the same terrain-preprocessing pattern:

- **M0/M1/M2** — a plain Fisher-KPP invasion model, diffusion driven by real Korean DEM terrain.
- **M3** — a two-species (Vespa velutina invasive vs. Vespa mandarinia native) competition-taxis model on real species-distribution-model (SDM) fields, solved with a different (Keller-Segel-style) solver. See `vespa_pde_model_spec.md` for the governing equation and its still-open modeling questions.

### M1/M2 — Fisher-KPP on DEM terrain

- **`src/model/fisher_kpp.py`** — `FisherKPP` class holding the PDE parameters (`D`, `r`, `K`) and the reaction/diffusion term calculations for `du/dt = D*Laplacian(u) + r*u*(1-u/K)`.
- **`src/model/diffusion.py`** — `constant_diffusion(shape, D)` (spatially constant field, M1) and `terrain_diffusion(dem, dx, dy, D_max, slope_scale, power, D_min)` (M2): `D(x,y) = D_min + (D_max - D_min) / (1 + (slope/slope_scale)**power)`, slope from `np.gradient` on the DEM. Known simplification: sea cells are filled with elevation 0, so flat ocean gets the same `D_max` as flat land (no land/sea mask on top of this diffusion field) — see the module docstring.
- **`src/solver/finite_difference.py`** — `FiniteDifference2D`: explicit-Euler time stepping with a central-difference 5-point Laplacian and Neumann (zero-flux) boundaries via edge padding (`np.pad(..., mode="edge")`). No CFL check inside the class itself — callers must pick a stable `dt` (see `experiments/m2_busan_terrain.py`'s manual check: `dt_stability_limit = dx**2 / (4*D_max)`, scaled by a `CFL_SAFETY` factor before use). `solve()` returns `(times, solutions)` with `solutions.shape == (n_saved, Ny, Nx)`.
- **`src/visualization/plot.py`** — `plot_population` / `plot_snapshots` for rendering 2D population fields with `imshow`.
- **`src/preprocessing/dem_downscale.py`** — reads a raw lon/lat elevation CSV (`data/raw/korea-alt-complete/...`), reprojects `EPSG:4326 → EPSG:5179` with `pyproj`, aggregates 90 m points into a 500 m grid by chunked mean-aggregation (keeps memory bounded via `CHUNK_SIZE`), then emits a CSV, a 2D `.npy` array, and a PNG figure into `data/processed/dem_500m/`.
  - `dem_grid.py` duplicates the CSV→2D-array→figure half of `dem_downscale.py` as a separate step; kept for taking an already-downscaled CSV straight to array/figure without re-running the full aggregation.
  - `check.py` is a one-off diagnostic for inspecting the raw per-tile coordinate extents (older multi-tile input format, `data/raw/korea-alt/`).
- **`experiments/m0_basic.py`** — stale/broken: imports `solve_fisher_kpp`/`plot_solution`, which no longer exist under those names after the solver/viz were refactored to the current class-based API. Not a usage reference.
- **`experiments/m1_2d_fisher.py`** — constant-`D` Fisher-KPP reference run.
- **`experiments/m2_busan_terrain.py`** — the DEM-driven run: crops the peninsula-wide DEM to a South Korea bounding box, builds `terrain_diffusion`, seeds a point introduction at Busan, and runs an interactive Matplotlib slider viewer.

### M3 — Vespa competition-taxis model

Governing equation (see `vespa_pde_model_spec.md` for full derivation and boundary conditions):

```
du/dt = D_u*Laplacian(u) - chi_u*div(u*grad(C_u(x))) + alpha*u*(1 - u/C_u(x)) - beta*u*v(x)
v(x)  = K_v * S_v(x)                                          (static field)
```

`u` = invasive (V. velutina) density, evolved by the PDE. `v` = native (V. mandarinia) density, a fixed field derived from its suitability map. `C_u(x)` is used as *both* the taxis potential (organisms move up its gradient) and the logistic carrying capacity — a working assumption flagged as unconfirmed in the spec.

- **`src/model/vespa_competition.py`** — `VespaCompetition` class: holds `D_u, chi_u, alpha, beta, C_u, v, mask` and computes `reaction()`/`diffusion()`/`rhs()`. Takes an `xp` array-module parameter (default `numpy`) so the same class runs on CPU or GPU — see **GPU backend**. Every output term is masked to zero outside `mask` (sea cells).
- **`src/solver/keller_segel.py`** — `KellerSegel2D`: a finite-volume solver, distinct from `FiniteDifference2D`. See **Numerical methods** below for the scheme; also takes `xp`.
- **`src/experiment_utils.py`** — helpers shared by the two M3 experiment scripts (factored out to avoid duplicating them): `load_vespa_fields`/`block_reduce_fields`/`block_reduce_1d` (load + optionally coarsen the 500m fields), `to_grid_index`/`nearest_land_cell` (lon/lat → grid cell), `create_point_source` (circular seed), `ensure_gpu_interpreter` (the GPU-venv auto-relaunch trick — see **GPU backend**).
- **`src/preprocessing/vespa_field_downscale.py`** — builds the two static input fields on a common 500m EPSG:5179 grid from irregular-resolution raw SDM rasters (`data/raw/SAM/density.csv` ~900m, `data/raw/jangsu/suitability.csv` ~9km) via `scipy.interpolate.griddata` linear interpolation (upsampling, not mean-aggregation like the DEM pipeline) plus a land mask derived from `density.csv`'s own point coverage (`LAND_SUPPORT_RADIUS_M`). Writes `land_mask_500m.npy`, `carrying_capacity_Cu_500m.npy`, `suitability_Sv_raw_500m.npy`, `suitability_Sv_normalized_500m.npy`, `grid_x_500m.npy`, `grid_y_500m.npy` into `data/processed/vespa_fields_500m/`.
- **`experiments/m3_vespa_competition.py`** — the main run: loads the fields (optionally coarsened via `GRID_FACTOR`), seeds Busan, runs `KellerSegel2D.solve()`, and saves `outputs/m3_vespa_competition/{timeseries.npz, snapshots.png, spread_analysis.png}`, then opens an interactive slider viewer.
- **`experiments/m3_plot_population_timeseries.py`** — standalone: just reads `outputs/m3_vespa_competition/timeseries.npz` (written by the main run) and plots total population vs. time. No solving, so it works from any interpreter.
- **`tests/test_vespa_competition.py`** — covers `VespaCompetition` validation, `KellerSegel2D`'s mass conservation (pure diffusion+taxis, no reaction, closed domain), sea-cell masking, and `stable_dt`'s monotonicity in `chi_u`.

### Numerical methods

**M1/M2 (`FiniteDifference2D`)**: explicit Euler, 5-point central-difference Laplacian, Neumann BC via edge-padding (so boundary cells see a zero-gradient mirror rather than a ghost value). No masking — the whole rectangular grid is solved, terrain enters only through the diffusion coefficient. `dt` stability is the caller's responsibility (no `stable_dt`-equivalent helper in this solver); the informal rule used in `experiments/m2_busan_terrain.py` is the standard 2D explicit-diffusion CFL bound `dt <= dx**2 / (4*D_max)`, taken with a safety factor.

**M3 (`KellerSegel2D`)**: a masked finite-volume scheme, needed because South Korea is an irregular (non-rectangular) landmass and the taxis (advection) term is prone to creating negative densities under naive central differencing:

- **Diffusion term** (`laplacian()`): face-centered central differences, assembled as `d/dx(flux_x) + d/dy(flux_y)` rather than a single 5-point stencil, so that any face bordering a masked-out (sea) cell or the array boundary can be forced to zero flux directly — this realizes the Neumann boundary condition *exactly* at the coastline, without ghost cells.
- **Taxis term** (`taxis_divergence()`): first-order upwind finite-volume flux of `u * velocity`, per the spec's recommendation (section 5) to avoid the negative-density problem a centered advection scheme would create. Same masked-face-flux construction as the diffusion term.
- **Time stepping**: explicit Euler (`step()`), with `u_next = max(u + dt*du_dt, 0)` as a numerical safety net against negative population from truncation error.
- **Stability**: `stable_dt(D_u, velocity_x, velocity_y, dx, dy)` returns the CFL limit for the *combined* diffusion+upwind-advection scheme: `dt <= 1 / (2*D_u*(1/dx**2 + 1/dy**2) + max|vx|/dx + max|vy|/dy)`. Every experiment script applies a `DT_SAFETY` factor (0.4 in the current scripts) on top of this raw limit. With the current literature `D_U`, the diffusion term dominates this bound heavily (by roughly 40x over the taxis term at the placeholder `CHI_U` scale) — see **Parameters** for why this matters for runtime.
- **Backend**: both `VespaCompetition` and `KellerSegel2D` take an `xp` parameter (`numpy` or `cupy`) so the identical code path runs on CPU or GPU; `KellerSegel2D.solve()` always converts each saved frame back to host `numpy` before returning (`_to_numpy`), so callers never need to branch on backend. Verified bit-for-bit identical output between the two backends on real data.

### GPU backend

`BACKEND = "gpu"` in `m3_vespa_competition.py` / `m3_chi_beta_sweep.py` switches the solve loop to CuPy for a large (~36x per-step, benchmarked) speedup. Two things to know:

1. **CuPy cannot run from this project's own `.venv`.** Its NVRTC kernel compiler fails on the non-ASCII (Korean) characters in this repo's OneDrive path (`...\바탕 화면\...`) with `cannot open source file "cupy/complex.cuh"`. The fix in use is a second, ASCII-path-only venv at `C:\gpuvenv`, with `cupy-cuda12x[ctk]`, `numpy`, `matplotlib`, and `pyproj` installed (CUDA 12.x wheels; this machine's driver — GTX 1660 Ti — supports CUDA 13.1, and 12.x wheels run fine under it).
2. **You normally don't need to think about this.** `src/experiment_utils.py`'s `ensure_gpu_interpreter(GPU_PYTHON)` runs at the top of both M3 experiment scripts when `BACKEND == "gpu"`: if the running interpreter isn't already `C:\gpuvenv\Scripts\python.exe`, it relaunches the same script under that interpreter and exits. So `python experiments/m3_vespa_competition.py` from *any* interpreter does the right thing automatically.

Benchmarked wall-clock time (whole-country grid, 20yr horizon, this machine):

| `GRID_FACTOR` | resolution | steps | CPU | GPU |
|---|---|---|---|---|
| 1 | 500m | ~452k | ~28 hr | ~46 min |
| 2 | 1km | ~113–138k | ~4 hr | ~10 min |
| 4 | 2km | ~28–41k | ~5.5 min | ~1 min |

Use `GRID_FACTOR=4`/GPU for fast exploratory runs, `GRID_FACTOR=1–2`/GPU for a single high-fidelity confirmation run.

### Parameters (M3)

Per `vespa_pde_model_spec.md` section 7, not everything in the model is settled. Current status, all in `experiments/m3_vespa_competition.py`:

**Literature-derived (fixed):**
- `D_U = 1,510,190` m²/day — random-walk diffusion coefficient.
- `ALPHA = 0.00077041` /day — intrinsic logistic growth rate.
- `K_V = 1.0` — scales normalized `S_v` into a native-species density.

**Still placeholders (no literature value identified):**
- `CHI_U` — taxis sensitivity. With the literature `D_U`, diffusion dominates transport heavily, so `CHI_U` needs to be quite large before it visibly affects outcomes — and once it's large enough to matter, it also starts shrinking the CFL-limited `dt`, eating back GPU speed gains. Treat as unresolved.
- `BETA` — competition strength from the native species. **Must be set relative to `ALPHA`, not as an independent absolute constant** — the breakeven native-suitability level above which the invasive species has net-negative local growth is `S_v_breakeven = ALPHA/BETA`. The original placeholder `BETA=0.01` was calibrated against an old `ALPHA=0.05`; once `ALPHA` became the literature value above (65x smaller), the same `BETA` made breakeven `S_v ≈ 0.077`, which **95.9% of South Korea's land area exceeds** — this silently drove the introduced population to near-total extinction within months in an earlier run, looking like "no diffusion happened" in the output figure until diagnosed. Current convention: `BETA = k * ALPHA` for a dimensionless `k`.

**Introduction / initial condition (assumption, not data-derived):**
- `INITIAL_RADIUS_M = 500` (~1 grid cell) and `INITIAL_DENSITY_FRACTION = 0.01` — seeded as a small fraction of *local* `C_u` at Busan, not an absolute density, since `C_u`'s absolute unit/scale is itself unconfirmed (spec section 7: is `density.csv` already "carrying capacity" in the sense the model assumes?).

**Open modeling question, not a parameter to tune:** with the literature `D_U`/`ALPHA` and a single-grid-cell point-source initial condition, the local diffusion timescale (`dx**2/(4*D_u)`, well under a day at any tested resolution) is roughly 3 orders of magnitude faster than the logistic growth timescale (`1/ALPHA`, ~1,300 days). Population dilutes across the domain far faster than it can locally rebuild toward carrying capacity, so a genuine Fisher-KPP traveling front may not lock in within any humanly-relevant simulated horizon from this IC. `spread_analysis.png`'s boundary/hotspot overlay is therefore defined relative to *that run's own max density* (`SPREAD_BOUNDARY_FRACTION_OF_MAX`, default 5%) rather than as a fraction of `C_u`, so it stays informative regardless of how far the run is from local saturation.

## Data

`data/` and `outputs/` are gitignored — raw source rasters and generated arrays/figures/results are never committed. Coordinate systems: all raw inputs are `EPSG:4326` (lon/lat), the working/output projection is `EPSG:5179` (Korea 2000 / Unified CS), with a 500 m target grid resolution (M3 experiments may further coarsen this in-memory via `GRID_FACTOR`, without regenerating the underlying 500m files).
