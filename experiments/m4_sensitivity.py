"""
M4: global sensitivity analysis of the no-competition (BETA = 0) Vespa
taxis model -- see m4_vespa_no_competition.py.

Two stages, run separately (each is hours of GPU time at GRID_FACTOR=4):

    python experiments/m4_sensitivity.py morris   # screening, all factors
    python experiments/m4_sensitivity.py sobol    # variance decomposition

1. Morris elementary effects screen every factor in PARAMETERS cheaply
   (n_trajectories*(k+1) runs) and rank them by mu* (mean |EE|) -- sigma
   flags nonlinearity / interactions.
2. Sobol (Saltelli design) then decomposes output variance into first-
   order S1 and total-order ST indices, but only over the SOBOL_TOP_N
   most influential factors from the Morris stage (by mu* on
   SCREENING_OUTPUT); screened-out factors are held at their nominal
   values. Without a Morris result on disk, all factors go into Sobol.

Every run is cached to outputs/m4_sensitivity/<method>_runs.npz as soon
as it finishes, so an interrupted stage resumes where it stopped (the
sample design is regenerated from a fixed seed and checked against the
cache).

Factors are the same as m3_sensitivity.py minus BETA_K: the placeholder
CHI_U, the initial-condition assumption, and the literature D_U/ALPHA
within an uncertainty band. BETA is fixed at 0 (that defines M4), which
also makes K_V and S_v irrelevant to the dynamics.
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
from src.sensitivity import (
    morris_trajectories, morris_indices, saltelli_samples, sobol_indices,
    scale_to_bounds,
)

# ============================================================
# Configuration
# ============================================================

DATA_DIR = ROOT / "data" / "processed" / "vespa_fields_500m"
OUTPUT_DIR = ROOT / "outputs" / "m4_sensitivity"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRID_FACTOR = 4   # 2km -- ~1 min/run on GPU at 20yr, see CLAUDE.md table
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

D_U_NOMINAL = 1_510_190.0
ALPHA_NOMINAL = 0.00077041
K_V = 1.0    # unused with BETA = 0, kept for the VespaCompetition signature
BETA = 0.0   # no native competition (M4)

# (name, low, high, log10-scaled, nominal). For log-scaled factors the
# bounds are log10 values and sampling is uniform in log space -- used for
# factors whose uncertainty is "order of magnitude", not "+-x%". Nominal
# values match m4_vespa_no_competition.py and are used for factors that the
# Morris stage screens out of the Sobol stage.
PARAMETERS = [
    ("D_U",                      np.log10(0.5 * D_U_NOMINAL),   np.log10(2.0 * D_U_NOMINAL),   True,  D_U_NOMINAL),
    ("ALPHA",                    np.log10(0.5 * ALPHA_NOMINAL), np.log10(2.0 * ALPHA_NOMINAL), True,  ALPHA_NOMINAL),
    ("CHI_U",                    3.0,                           6.5,                           True,  5e5),
    ("INITIAL_DENSITY_FRACTION", -3.0,                          -1.0,                          True,  0.01),
]
PARAM_NAMES = [p[0] for p in PARAMETERS]

INITIAL_RADIUS_M = 500.0
DT_SAFETY = 0.4
TOTAL_TIME_DAYS = 365.0 * 20.0

# Same spread-front definition as m4_vespa_no_competition.py. There is no
# C_u-relative "established area" output on purpose: with the literature
# D_U/ALPHA it stays empty for most of the parameter space (see CLAUDE.md),
# so it would carry no variance to decompose. log10 of the peak u/C_u
# measures how close the run gets to local saturation instead.
SPREAD_BOUNDARY_FRACTION_OF_MAX = 0.05

OUTPUT_NAMES = ["log10_total_population", "max_spread_km", "log10_peak_saturation"]

MORRIS_TRAJECTORIES = 10   # 10*(4+1) = 50 runs
MORRIS_LEVELS = 4
MORRIS_SEED = 20261101

SCREENING_OUTPUT = "max_spread_km"   # which output's mu* picks Sobol factors
SOBOL_TOP_N = 3
SOBOL_BASE_SAMPLES = 64   # n_base*(k+2) runs: 64*5 = 320 at k=3
SOBOL_SEED = 20261102
SOBOL_BOOTSTRAP = 1000


# ============================================================
# Model evaluation
# ============================================================

class Evaluator:
    """Loads the fields once and evaluates the model at one parameter set."""

    def __init__(self):
        land_mask, C_u, S_v, x_centers, y_centers = load_vespa_fields(DATA_DIR, GRID_FACTOR)

        transformer = Transformer.from_crs(LON_LAT_CRS, WORKING_CRS, always_xy=True)
        row, col = to_grid_index(BUSAN_LON, BUSAN_LAT, transformer, x_centers, y_centers)
        self.busan_row, self.busan_col = nearest_land_cell(land_mask, row, col)

        self.land_mask = land_mask
        self.C_u = C_u
        self.land_mask_xp = xp.asarray(land_mask) if BACKEND == "gpu" else land_mask
        self.C_u_xp = xp.asarray(C_u) if BACKEND == "gpu" else C_u
        self.S_v_xp = xp.asarray(S_v) if BACKEND == "gpu" else S_v

        print(f"Grid shape: {land_mask.shape}, Busan cell: ({self.busan_row}, {self.busan_col})")

    def __call__(self, params):
        D_u = params["D_U"]
        alpha = params["ALPHA"]
        chi_u = params["CHI_U"]

        model = VespaCompetition(
            D_u=D_u, chi_u=chi_u, alpha=alpha, beta=BETA,
            C_u=self.C_u_xp, K_v=K_V, S_v=self.S_v_xp, mask=self.land_mask_xp, xp=xp,
        )

        geo = KellerSegel2D(dx=GRID, dy=GRID, dt=1.0, xp=xp)
        velocity_x, velocity_y = geo.taxis_velocity(model.C_u, chi_u)
        dt = DT_SAFETY * stable_dt(D_u, velocity_x, velocity_y, GRID, GRID, xp=xp)
        solver = KellerSegel2D(dx=GRID, dy=GRID, dt=dt, xp=xp)
        steps = int(TOTAL_TIME_DAYS / dt)

        initial_density = params["INITIAL_DENSITY_FRACTION"] * self.C_u[self.busan_row, self.busan_col]
        u0 = create_point_source(
            self.land_mask.shape, self.busan_row, self.busan_col,
            radius_m=INITIAL_RADIUS_M, density=initial_density, grid=GRID,
        )
        u0 = np.where(self.land_mask, u0, 0.0)
        if BACKEND == "gpu":
            u0 = xp.asarray(u0)

        _, solutions = solver.solve(
            u0=u0, model=model, velocity_x=velocity_x, velocity_y=velocity_y,
            steps=steps, save_every=steps,
        )

        return self.summarize(solutions[-1]), steps

    def summarize(self, final_u):
        land_u = np.where(self.land_mask, final_u, 0.0)
        total_population = float(land_u.sum())
        max_density = float(land_u.max())

        if max_density > 0:
            rows, cols = np.nonzero(land_u >= SPREAD_BOUNDARY_FRACTION_OF_MAX * max_density)
            distance_km = np.sqrt(
                ((rows - self.busan_row) * GRID) ** 2 + ((cols - self.busan_col) * GRID) ** 2
            ) / 1000.0
            max_spread_km = float(distance_km.max())
        else:
            max_spread_km = 0.0

        peak_saturation = float((land_u / self.C_u)[self.land_mask].max())

        return np.array([
            np.log10(max(total_population, 1e-300)),
            max_spread_km,
            np.log10(max(peak_saturation, 1e-300)),
        ])


def to_physical(unit_row, factor_indices):
    """
    Unit-hypercube sample over `factor_indices` -> full parameter dict,
    with every other factor at its nominal value.
    """

    params = {name: nominal for name, _, _, _, nominal in PARAMETERS}

    bounds = [(PARAMETERS[i][1], PARAMETERS[i][2]) for i in factor_indices]
    scaled = scale_to_bounds(unit_row[None, :], bounds)[0]

    for value, i in zip(scaled, factor_indices):
        name, _, _, is_log, _ = PARAMETERS[i]
        params[name] = 10.0 ** value if is_log else value

    return params


def run_design(method, unit_samples, factor_indices):
    """Evaluate every row of the design, resuming from the on-disk cache."""

    cache_path = OUTPUT_DIR / f"{method}_runs.npz"
    factor_names = np.array([PARAM_NAMES[i] for i in factor_indices])

    Y = np.full((len(unit_samples), len(OUTPUT_NAMES)), np.nan)

    if cache_path.exists():
        cache = np.load(cache_path)
        if (
            cache["unit_samples"].shape == unit_samples.shape
            and np.allclose(cache["unit_samples"], unit_samples)
            and list(cache["factor_names"]) == list(factor_names)
        ):
            Y = cache["Y"].copy()
            print(f"Resuming from cache: {int((~np.isnan(Y[:, 0])).sum())}/{len(Y)} runs done.")
        else:
            print(f"Cache at {cache_path} is for a different design -- starting over.")

    evaluate = Evaluator()
    pending = np.nonzero(np.isnan(Y[:, 0]))[0]
    print(f"{method}: {len(pending)} runs to go (GRID_FACTOR={GRID_FACTOR}, BACKEND={BACKEND}).")

    stage_start = time.perf_counter()
    for count, i in enumerate(pending, start=1):
        params = to_physical(unit_samples[i], factor_indices)
        start = time.perf_counter()
        Y[i], steps = evaluate(params)
        elapsed = time.perf_counter() - start

        remaining = (time.perf_counter() - stage_start) / count * (len(pending) - count)
        print(
            f"  [{count}/{len(pending)}] run {i}: "
            + ", ".join(f"{name}={params[name]:.3g}" for name in factor_names)
            + f" -> logpop={Y[i, 0]:.3f}, spread={Y[i, 1]:.0f}km, logsat={Y[i, 2]:.3f}"
            + f"  [{steps} steps, {elapsed:.0f}s, ~{remaining / 60:.0f} min left]"
        )

        np.savez(
            cache_path, unit_samples=unit_samples, factor_names=factor_names,
            output_names=np.array(OUTPUT_NAMES), Y=Y,
        )

    return Y


# ============================================================
# Stages
# ============================================================

def run_morris():
    k = len(PARAMETERS)
    factor_indices = list(range(k))
    X = morris_trajectories(k, MORRIS_TRAJECTORIES, MORRIS_LEVELS, rng=MORRIS_SEED)
    Y = run_design("morris", X, factor_indices)

    results = {}
    print("\nMorris results (mu* / sigma, in unit-hypercube scale):")
    for j, output in enumerate(OUTPUT_NAMES):
        r = morris_indices(X, Y[:, j], k)
        results[output] = r
        print(f"  {output}:")
        for i in np.argsort(-r["mu_star"]):
            print(f"    {PARAM_NAMES[i]:<26s} mu*={r['mu_star'][i]:10.4g}  sigma={r['sigma'][i]:10.4g}")

    np.savez(
        OUTPUT_DIR / "morris_results.npz",
        param_names=np.array(PARAM_NAMES), output_names=np.array(OUTPUT_NAMES),
        mu=np.array([results[o]["mu"] for o in OUTPUT_NAMES]),
        mu_star=np.array([results[o]["mu_star"] for o in OUTPUT_NAMES]),
        sigma=np.array([results[o]["sigma"] for o in OUTPUT_NAMES]),
    )
    plot_morris(results)


def select_sobol_factors():
    morris_path = OUTPUT_DIR / "morris_results.npz"
    if not morris_path.exists():
        print("No morris_results.npz found -- running Sobol over all factors.")
        return list(range(len(PARAMETERS)))

    morris = np.load(morris_path)
    j = list(morris["output_names"]).index(SCREENING_OUTPUT)
    ranked = np.argsort(-morris["mu_star"][j])[:SOBOL_TOP_N]
    names = [str(n) for n in morris["param_names"][ranked]]
    print(f"Sobol factors (top {SOBOL_TOP_N} by Morris mu* on {SCREENING_OUTPUT}): {names}")

    return sorted(PARAM_NAMES.index(n) for n in names)


def run_sobol():
    factor_indices = select_sobol_factors()
    k = len(factor_indices)
    X = saltelli_samples(SOBOL_BASE_SAMPLES, k, rng=SOBOL_SEED)
    Y = run_design("sobol", X, factor_indices)

    factor_names = [PARAM_NAMES[i] for i in factor_indices]
    results = {}
    print("\nSobol results (S1 / ST, +- 95% bootstrap half-width):")
    for j, output in enumerate(OUTPUT_NAMES):
        r = sobol_indices(Y[:, j], SOBOL_BASE_SAMPLES, k, n_bootstrap=SOBOL_BOOTSTRAP, rng=SOBOL_SEED)
        results[output] = r
        print(f"  {output} (n_valid={r['n_valid']}):")
        for i, name in enumerate(factor_names):
            print(
                f"    {name:<26s} S1={r['S1'][i]:6.3f} +- {r['S1_conf'][i]:.3f}   "
                f"ST={r['ST'][i]:6.3f} +- {r['ST_conf'][i]:.3f}"
            )

    np.savez(
        OUTPUT_DIR / "sobol_results.npz",
        param_names=np.array(factor_names), output_names=np.array(OUTPUT_NAMES),
        **{f"{key}": np.array([results[o][key] for o in OUTPUT_NAMES])
           for key in ("S1", "S1_conf", "ST", "ST_conf")},
    )
    plot_sobol(results, factor_names)


# ============================================================
# Visualization
# ============================================================

def plot_morris(results):
    fig, axes = plt.subplots(1, len(OUTPUT_NAMES), figsize=(5.5 * len(OUTPUT_NAMES), 5))

    for ax, output in zip(axes, OUTPUT_NAMES):
        r = results[output]
        ax.scatter(r["mu_star"], r["sigma"], color="tab:blue", zorder=3)
        for i, name in enumerate(PARAM_NAMES):
            ax.annotate(name, (r["mu_star"][i], r["sigma"][i]), fontsize=8,
                        xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("mu* (mean |elementary effect|)")
        ax.set_ylabel("sigma (std of elementary effect)")
        ax.set_title(output)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Morris screening ({MORRIS_TRAJECTORIES} trajectories, GRID_FACTOR={GRID_FACTOR})")
    fig.tight_layout()
    output_path = OUTPUT_DIR / "morris_screening.png"
    fig.savefig(output_path, dpi=150)
    print(f"Figure saved to:\n{output_path}")


def plot_sobol(results, factor_names):
    fig, axes = plt.subplots(1, len(OUTPUT_NAMES), figsize=(5.5 * len(OUTPUT_NAMES), 5))
    positions = np.arange(len(factor_names))
    width = 0.38

    for ax, output in zip(axes, OUTPUT_NAMES):
        r = results[output]
        ax.bar(positions - width / 2, r["S1"], width, yerr=r["S1_conf"], capsize=3, label="S1 (first order)")
        ax.bar(positions + width / 2, r["ST"], width, yerr=r["ST_conf"], capsize=3, label="ST (total order)")
        ax.set_xticks(positions)
        ax.set_xticklabels(factor_names, rotation=20, ha="right")
        ax.set_ylim(min(0.0, ax.get_ylim()[0]), max(1.05, ax.get_ylim()[1]))
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_title(output)
        ax.legend(fontsize=8)

    fig.suptitle(f"Sobol indices (N={SOBOL_BASE_SAMPLES}, GRID_FACTOR={GRID_FACTOR})")
    fig.tight_layout()
    output_path = OUTPUT_DIR / "sobol_indices.png"
    fig.savefig(output_path, dpi=150)
    print(f"Figure saved to:\n{output_path}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "morris"
    if stage == "morris":
        run_morris()
    elif stage == "sobol":
        run_sobol()
    else:
        raise SystemExit("usage: python experiments/m4_sensitivity.py [morris|sobol]")
