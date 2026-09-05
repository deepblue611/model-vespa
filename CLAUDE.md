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

# Lint / format
ruff check .
black .

# Run an experiment (must run from repo root, see Architecture)
python experiments/m1_2d_fisher.py

# Run the DEM preprocessing pipeline
python src/preprocessing/dem_downscale.py
```

## Architecture

This is a 2D reaction-diffusion (Fisher-KPP) simulation project, developed in milestones (see git log: M0 → M1 → M1.1). The end goal is to drive the diffusion term with real Korean terrain (DEM) data rather than a constant coefficient.

- **`src/model/fisher_kpp.py`** — `FisherKPP` class holding the PDE parameters (`D`, `r`, `K`) and the reaction/diffusion term calculations for `du/dt = D*Laplacian(u) + r*u*(1-u/K)`.
- **`src/model/diffusion.py`** — currently only builds a spatially constant diffusion field (`constant_diffusion`). Its docstring flags this as the extension point for a future terrain-based `terrain_diffusion(dem, ...)` once the DEM pipeline output is wired in.
- **`src/solver/finite_difference.py`** — `FiniteDifference2D`: explicit-Euler time stepping with a central-difference Laplacian and Neumann (zero-flux) boundaries via edge padding. `solve()` returns `(times, solutions)` with `solutions.shape == (n_saved, Ny, Nx)`.
- **`src/visualization/plot.py`** — `plot_population` / `plot_snapshots` for rendering 2D population fields with `imshow`.
- **`src/preprocessing/`** — standalone DEM data pipeline, independent of `src/model`/`src/solver`:
  - `dem_downscale.py` is the main entry point: reads a raw lon/lat elevation CSV (`data/raw/korea-alt-complete/...`), reprojects `EPSG:4326 → EPSG:5179` with `pyproj`, aggregates 90 m points into a 500 m grid by chunked mean-aggregation (keeps memory bounded via `CHUNK_SIZE`), then emits a CSV, a 2D `.npy` array, and a PNG figure into `data/processed/dem_500m/`.
  - `dem_grid.py` duplicates the CSV→2D-array→figure half of `dem_downscale.py` as a separate step; kept for taking an already-downscaled CSV straight to array/figure without re-running the full aggregation.
  - `check.py` is a one-off diagnostic for inspecting the raw per-tile coordinate extents (older multi-tile input format, `data/raw/korea-alt/`).
- **`experiments/`** — runnable milestone scripts wiring model + solver + visualization together with concrete parameters. They are not a package: each one does `sys.path.append(str(ROOT))` before importing `src.*`, so they only work when run with the repo root as the working directory. `m0_basic.py` imports `solve_fisher_kpp`/`plot_solution`, which no longer exist under those names after the solver/viz were refactored to the current class-based API — treat it as stale/broken, not a usage reference.
- **`tests/`** use the same manual `sys.path.append` pattern (no installed package), so tests must be run from the repo root as well.

## Data

`data/` and `outputs/` are gitignored — raw DEM tiles and generated arrays/figures are never committed. Coordinate systems: input DEM data is `EPSG:4326` (lon/lat), the working/output projection is `EPSG:5179` (Korea 2000 / Unified CS), with a 500 m target grid resolution.
