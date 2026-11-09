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
pytest tests/test_vespa_competition.py::test_no_flux_conserves_mass_with_no_reaction

# Lint / format
ruff check .
black .

# Field preprocessing (writes data/processed/vespa_fields_500m/)
python src/preprocessing/vespa_field_downscale.py

# Main runs (must run from repo root; GPU runs auto-relaunch under C:\gpuvenv)
python experiments/m3_vespa_competition.py       # with native competition
python experiments/m4_vespa_no_competition.py    # BETA=0

# Post-processing (read a finished run's outputs; no solving)
python experiments/m3_plot_population_timeseries.py   # likewise m4_plot_population_timeseries.py
python experiments/m3_export_u_csv.py                 # u_fields.npz -> u_fields.csv (needs pandas -> project .venv)

# Sensitivity analysis (GPU, GRID_FACTOR=4; still on the old 5%-of-max front)
python experiments/m3_sensitivity.py morris   # ~60 runs
python experiments/m3_sensitivity.py sobol    # ~320 runs, on Morris's top factors
python experiments/m4_sensitivity.py morris|sobol   # same, BETA=0, without the BETA_K factor
```

## Architecture

A two-species reaction-diffusion-taxis model of *Vespa velutina* (invasive) spreading from Busan, competing with *Vespa mandarinia* (native), on real South Korea species-distribution fields. See `vespa_pde_model_spec.md` for the governing equation and its open modeling questions.

```
du/dt = D_u*Laplacian(u) - chi_u*div(u*grad(C_u(x))) + alpha*u*(1 - u/C_u(x)) - beta*u*v(x)
v(x)  = K_v * S_v(x)                                          (static field)
```

`u` = invasive density, evolved by the PDE. `v` = native density, a fixed field from its suitability map. `C_u` is both the taxis potential and the logistic carrying capacity. That is a working assumption, flagged as unconfirmed in the spec.

Legacy Fisher-KPP files from earlier milestones (M0–M2: `m0_basic.py`, `m1_2d_fisher.py`, `m2_busan_terrain.py`, `src/model/fisher_kpp.py`, `src/model/diffusion.py`, `src/solver/finite_difference.py`, `src/preprocessing/dem_*.py`) are still in the repo but are not documented here and are not part of the current model.

### Model variants

- `m3_vespa_competition.py` uses `BETA = 0.3*ALPHA` and writes to `outputs/m3_vespa_competition/`.
- `m4_vespa_no_competition.py` is the same model with `BETA = 0` and writes to `outputs/m4_vespa_no_competition/`. M3 − M4 isolates the effect of *V. mandarinia*.
- Both use C_u as stored, i.e. on the SAM's ln(catch+1) scale (see **Data and units**).
- **Keep `GRID_FACTOR`, `INITIAL_*`, `CHI_U` and `TOTAL_TIME_DAYS` in sync between M3 and M4**, or the difference no longer isolates competition.

### Files

- **`src/model/vespa_competition.py`** — `VespaCompetition`: holds `D_u, chi_u, alpha, beta, C_u, v, mask`; computes `reaction()`/`diffusion()`/`rhs()`. Takes an `xp` array module (`numpy` or `cupy`). Every term is masked to zero on sea cells. Requires `C_u > 0` on land.
- **`src/solver/keller_segel.py`** — `KellerSegel2D` masked finite-volume solver (see **Numerical methods**), plus `taxis_velocity(potential, chi_u)` and `stable_dt(...)`.
- **`src/experiment_utils.py`** — shared helpers:
  - `load_vespa_fields(data_dir, grid_factor, back_transform_log1p=False)` loads the 500m fields and optionally coarsens them.
  - `back_transform_log1p=True` applies `expm1` to C_u at 500m, *before* block-averaging, so a coarse cell holds a mean catch rather than the expm1 of a mean log. It defaults to off; M3/M4 don't use it.
  - Also: `block_reduce_fields`/`block_reduce_1d`, `to_grid_index`/`nearest_land_cell`, `create_point_source`, and `ensure_gpu_interpreter`.
- **`src/preprocessing/vespa_field_downscale.py`** — builds the static fields on a common 500m EPSG:5179 grid with `scipy.interpolate.griddata` linear interpolation:
  - C_u from `data/raw/SAM/density.csv` (~0.01°), stored unchanged, i.e. still on the log scale.
  - S_v from `data/raw/jangsu/suitability.csv` (5 arcmin, ~9 km), min-max normalized over land.
  - Land mask from `density.csv`'s own point coverage.
  - Writes `land_mask_500m.npy`, `carrying_capacity_Cu_500m.npy`, `suitability_Sv_raw_500m.npy`, `suitability_Sv_normalized_500m.npy`, `grid_x_500m.npy`, `grid_y_500m.npy`.
- **Main run scripts (M3/M4)** — each one:
  - loads the fields and seeds Busan;
  - runs `solve()`, saving a frame every `SAVE_INTERVAL_DAYS` (90). Saving every `dt` would be ~150 GB at `GRID_FACTOR=2` and would not fit in RAM;
  - writes `timeseries.npz` (times, total population, spread area, max spread distance, `spread_boundary_absolute`), `u_fields.npz`, `snapshots.png` and `spread_analysis.png`;
  - opens an interactive slider viewer.

  `u_fields.npz` holds per-cell u for land cells only (`u_land`, shape `(n_frames, n_land)`, row-major `land_mask` order) plus `times`, `land_mask`, `x_centers`, `y_centers`.
- **`m{3,4}_plot_population_timeseries.py`** — read the matching `timeseries.npz` and plot population, spread area, max distance from Busan, and front speed.
  - The front-speed fit uses only frames with `0 < distance < FIT_MAX_DISTANCE_KM` (380 km). The front reaches the far edge of the landmass at ~397 km, and including that plateau biases a linear fit toward whichever run started slower.
  - Older `timeseries.npz` files (with `spread_boundary_fraction_of_max`) still load.
- **`m3_export_u_csv.py`** — `u_fields.npz` → `u_fields.csv`, one row per land cell (lon/lat, 6 decimals), one `t_<day>d` column per frame. About 100 MB at `GRID_FACTOR=2`. Needs pandas, so run it from the project `.venv`.
- **`m3_sensitivity.py` / `m4_sensitivity.py`** — Morris screening then Sobol over the top `SOBOL_TOP_N` factors.
  - Factors: `D_U`, `ALPHA` (log, 0.5–2×), `CHI_U` (log, 1e3–10^6.5), `BETA_K` (0–2, M3 only), `INITIAL_DENSITY_FRACTION` (log, 1e-3–1e-1).
  - Outputs: `log10_total_population`, `max_spread_km`, `log10_peak_saturation`.
  - Runs are cached to `outputs/m{3,4}_sensitivity/<stage>_runs.npz`, so an interrupted stage resumes.
  - **Not yet updated** to the absolute spread threshold: they still use `SPREAD_BOUNDARY_FRACTION_OF_MAX = 0.05`.
- **`m3_chi_beta_sweep.py`** — coarse CHI_U × BETA regime map (dies out / persists / spreads).
- **`src/sensitivity.py`** — numpy-only Morris and Saltelli/Sobol, hand-rolled because the GPU venv has neither scipy nor SALib. Tested in `tests/test_sensitivity.py` against a linear function and Ishigami's analytic indices.
- **`tests/test_vespa_competition.py`** — covers model validation, mass conservation (diffusion+taxis, no reaction), sea masking, and `stable_dt` monotonicity in `chi_u`.

### Numerical methods (`KellerSegel2D`)

A masked finite-volume scheme, needed because the landmass is irregular and the naive centered taxis term creates negative densities.

- **Diffusion** — face-centered differences assembled as `d/dx(flux_x) + d/dy(flux_y)`. Faces bordering sea or the array edge are forced to zero flux, which gives an exact Neumann boundary at the coast with no ghost cells.
- **Taxis** — first-order upwind finite-volume flux of `u*velocity`, with the same masked faces (spec section 5).
- **Time stepping** — explicit Euler, `u_next = max(u + dt*du_dt, 0)`.
- **Stability** — `stable_dt = 1 / (2*D_u*(1/dx**2 + 1/dy**2) + max|vx|/dx + max|vy|/dy)`. Scripts multiply it by `DT_SAFETY = 0.4`. The diffusion term dominates this bound.
- **Backend** — `xp` = numpy or cupy, using the identical code path. `solve()` always returns host numpy. Verified bit-for-bit identical results across the two backends.

### GPU backend

`BACKEND = "gpu"` uses CuPy (~36× per-step speedup).

1. **CuPy cannot run from the project `.venv`.** NVRTC fails on the non-ASCII OneDrive path (`...\바탕 화면\...`) with `cannot open source file "cupy/complex.cuh"`. A second, ASCII-path venv at `C:\gpuvenv` has `cupy-cuda12x[ctk]`, `numpy`, `matplotlib` and `pyproj` installed (GTX 1660 Ti, driver supports CUDA 13.1, 12.x wheels work). That venv has no pandas, scipy or SALib.
2. **`ensure_gpu_interpreter(GPU_PYTHON)` relaunches the script under `C:\gpuvenv`**, so running the script from any interpreter works.

Wall-clock time (whole country, 20 yr):

| `GRID_FACTOR` | resolution | steps | CPU | GPU |
|---|---|---|---|---|
| 1 | 500m | ~452k | ~28 hr | ~46 min |
| 2 | 1km | ~113–138k | ~4 hr | ~10 min |
| 4 | 2km | ~28–41k | ~5.5 min | ~1 min |

## Data and units

`data/` and `outputs/` are gitignored. Raw inputs are EPSG:4326; the working grid is EPSG:5179 at 500 m. Scripts may coarsen in memory via `GRID_FACTOR`.

### C_u — V. velutina SAM (`data/raw/SAM/density.csv`)

Source: Kim, Bak & Jung (2023), *Sci. Rep.* 13:13616, https://doi.org/10.1038/s41598-023-40016-9. It is the RF species-abundance-model prediction (`prediction/density_rf.asc` in the authors' data package, same 0.01° grid).

- **The stored values are `ln(mean trap catch + 1)`, not a density.**
  - The response was R's `log()`, which is the natural log.
  - The raw data are *V. velutina* caught per bait trap (fermented oak sap) per month, in September or October, averaged over 2018–2021 per site. That is a relative foraging-abundance index of workers, not individuals or nests per km².
  - Raw `density` has mean 2.45 and max 4.90, i.e. about 10.6 and 133 catches.
  - On the 1 km grid, `expm1` C_u on land ranges 0.19–91 (block averaging lowers the max).
- **Trap catches cannot be converted to absolute density.** No conversion factor is known, so results must be reported as "expected trap catch equivalent" or as relative abundance.
- **Using C_u as carrying capacity is defensible.** The survey (2018–2021) was taken 15+ years after the 2003 Busan introduction, so the field is a near-equilibrium distribution. The north, invaded later, may not have been saturated yet.
- **Caveats:**
  - Modest predictive skill: held-out r = 0.47, slope 0.867.
  - RF shrinks extremes: the predicted max is ~133 against an observed site max of 338.5.
  - `expm1` of a predicted log underestimates the mean (Jensen).
  - Jeju was excluded from training, so Jeju values are extrapolated.
- **What this means for M3/M4:** u has the same "unit" as ln(catch+1). Log values are not additive, so absolute u and summed "total population" there are not interpretable.
  - Still valid: u/C_u and M3-vs-M4 comparisons. For small values ln(1+x) ≈ x, so tiny u ≈ catches anyway.
  - The log scale also compressed C_u's spatial contrast: max/min ~25× instead of ~470× on the linear scale.
- The authors' package also has `probability_rf.asc` (RF presence probability) and raw trap records. Neither is used yet; they are candidates for validation or an alternative taxis potential.

### S_v — V. mandarinia suitability (`data/raw/jangsu/suitability.csv`)

- 1,465 points on a 5 arcmin (~9 km) grid. Values 19.8–480.4 (median 138.8), one decimal place.
- **Unit and source model are unknown, and no metadata has been found.** It is not a probability and not a log value.
- Currently min-max normalized over land to [0, 1].
- If it is a ratio scale, `S/max` would be more defensible: min-max zeroes out competition at the least-suitable cell, whose raw value is 19.8, not 0.
- The C_u back-transform does *not* require changing S_v. `beta*u*v` is linear in u, so `beta*v` is a per-capita mortality rate map that is independent of u's unit.

## Parameters

Literature-derived (fixed):
- `D_U = 1,510,190` m²/day — random-walk diffusion.
- `ALPHA = 0.00077041` /day — logistic growth (~0.28/yr, doubling ~2.5 yr).
- `K_V = 1.0`.

Placeholders:
- **`CHI_U`** (5e5 in the current scripts) — taxis sensitivity, multiplying `grad(C_u)` with C_u on the ln(catch+1) scale. Max taxis velocity is ~780 m/day at 1 km.
- **`BETA`** — always set as `BETA = k * ALPHA` (currently k = 0.3), never as an independent absolute constant.
  - The breakeven native suitability for net-negative invasive growth is `S_v = ALPHA/BETA`.
  - An old `BETA = 0.01`, calibrated against an earlier `ALPHA = 0.05`, put breakeven at S_v ≈ 0.077. 95.9% of land exceeds that, and the population went extinct within months.

Initial condition (assumption):
- A point source at Busan of radius `INITIAL_RADIUS_M = 500` (one cell), with density `INITIAL_DENSITY_FRACTION × C_u(Busan)`.
- Being relative to C_u, the IC makes the dynamics invariant to C_u's overall scale (except through taxis).

## Spread metric

- **The spread boundary is a fixed absolute density: `SPREAD_BOUNDARY_ABSOLUTE = 1e-5`.** It is used for the contour, area and max distance in the M3/M4 scripts.
- **It replaced "5% of each frame's own max u".** That definition made M3 look as fast as M4 (16.2 vs 16.2 km/yr), because competition lowers M3's max and so lowers its threshold too.
  - On identical thresholds, M4 ≥ M3 in every cell, as expected. Example: at 1e-5 with the 380 km fit window, M3 ≈ 17.9 and M4 ≈ 19.3 km/yr.
- **Measured front speed depends strongly on the threshold** (~20–35 km/yr between 1e-8 and 1e-6). Compare runs at the same threshold, and don't read the speed as an absolute invasion speed.
  - At 1e-5, M3's max u dips just below the threshold around year ~3.7, so a few frames report zero spread.
  - `u ≥ 1` (one wasp per trap per month, the paper's "Low" class lower bound) would be the biologically meaningful detection boundary. It is empty under the current IC because max u stays ≪ 1.

## Open modeling problem: the population never saturates (atto-fox)

**With literature `D_U`/`ALPHA` and a one-cell point-source IC, u stays orders of magnitude below C_u for the whole 20 yr run.**

- **Observed in M3/M4:** the max u/C_u stays around 1e-3 or below for the whole run.
- **Rescaling C_u doesn't help.** Because the IC is a fraction of C_u, rescaling C_u (e.g. back-transforming it to the linear catch scale) just rescales u by the same factor.
- **Cause:** diffusive reach √(4Dt) is ~47 km at 1 yr and ~210 km at 20 yr, while growth is only e^(αT) ≈ 275× over 20 yr. M4 grows by exactly e^(αT), so the logistic term never engages.
- **Estimate:** central density ≈ M·e^(αt)/(4πDt), where M is the initial mass.
  - It matches the simulated max within a factor of ~2; taxis concentrates slightly.
  - Starting from a single cell filled to its full carrying capacity, it predicts saturation at Busan only after ~39 yr, even without competition.
- **The model therefore spreads a thin, exponentially growing cloud, not a Fisher-KPP front.** "Spread" at 1e-5 tracks the diffusive tail (deterministic-PDE "atto-fox" problem, Mollison 1991).
- **This contradicts observation.** Real *V. velutina* went from Busan (2003) to nationwide in ~15 years, with mean trap catches ≥10 by 2018–2021.
- **Candidate fixes**, not yet tried:
  - re-examine ALPHA (0.28/yr is very slow for a social wasp whose nests produce hundreds of gynes);
  - a larger founding area (`INITIAL_RADIUS_M` of 10–20 km; limited effect by the estimate above);
  - an Allee effect or a low-density cutoff;
  - calibrating α/D against the observed ~15-yr nationwide establishment.
