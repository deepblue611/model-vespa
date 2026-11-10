"""
Map the first-arrival time of V. velutina at every land cell, from a
finished run's u_fields.npz (no solving).

Arrival = first time u >= SPREAD_BOUNDARY_ABSOLUTE (the same absolute
threshold the M3/M4 scripts use). Frames are only every 90 days, so the
crossing time is interpolated linearly in log10(u) between the last frame
below and the first frame at/above the threshold. Cells never reached are
drawn grey.

Usage (from repo root):
    python experiments/plot_arrival_time.py m3            # one run
    python experiments/plot_arrival_time.py m3 m4         # side by side + delay map (m3 - m4)
    python experiments/plot_arrival_time.py m3 m4 --threshold 1e-6

Writes outputs/<run>/arrival_time.png (one run) or
outputs/arrival_time_<a>_vs_<b>.png (two runs), plus arrival_time_years.npy
per run (full grid, NaN = sea or never reached).
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

RUN_DIRS = {
    "m3": "m3_vespa_competition",
    "m4": "m4_vespa_no_competition",
    "m5": "m5_vespa_competition",
    "m6": "m6_vespa_no_competition",
}
# Display names in figures (data/script names unchanged).
LABELS = {"m4": "m1"}
SPREAD_BOUNDARY_ABSOLUTE = 1e-5
BUSAN_LON, BUSAN_LAT = 129.0403, 35.1028
CONTOUR_YEARS = [1, 2, 5, 10, 15, 20]


def arrival_time_years(u_land, times_days, threshold):
    """Per-land-cell first crossing time (years), NaN if never reached."""
    above = u_land >= threshold                      # (n_frames, n_land)
    reached = above.any(axis=0)
    k = np.argmax(above, axis=0)                     # first frame at/above
    t = times_days[k].astype(float)

    # Interpolate in log10(u) between frame k-1 and k (k == 0: seeded cell).
    interp = reached & (k > 0)
    idx = np.nonzero(interp)[0]
    u0 = u_land[k[idx] - 1, idx]
    u1 = u_land[k[idx], idx]
    t0, t1 = times_days[k[idx] - 1], times_days[k[idx]]
    with np.errstate(divide="ignore", invalid="ignore"):
        l0 = np.log10(np.maximum(u0, 1e-300))
        l1 = np.log10(u1)
        lt = np.log10(threshold)
        frac = np.clip((lt - l0) / (l1 - l0), 0.0, 1.0)
    t[idx] = t0 + frac * (t1 - t0)

    t[~reached] = np.nan
    return t / 365.25


def load_run(name, threshold):
    path = ROOT / "outputs" / RUN_DIRS.get(name, name) / "u_fields.npz"
    d = np.load(path)
    mask = d["land_mask"]
    grid = np.full(mask.shape, np.nan)
    grid[mask] = arrival_time_years(d["u_land"], d["times"], threshold)
    np.save(path.parent / "arrival_time_years.npy", grid)
    t_end = d["times"][-1] / 365.25
    n_reached = np.isfinite(grid).sum()
    print(f"{name}: reached {n_reached}/{mask.sum()} land cells "
          f"({100 * n_reached / mask.sum():.1f}%) by {t_end:.1f} yr")
    return grid, mask, d["x_centers"] / 1e3, d["y_centers"] / 1e3, t_end, path.parent


def busan_xy_km():
    try:
        from pyproj import Transformer
    except ImportError:
        return None
    tr = Transformer.from_crs("EPSG:4326", "EPSG:5179", always_xy=True)
    x, y = tr.transform(BUSAN_LON, BUSAN_LAT)
    return x / 1e3, y / 1e3


def draw_arrival(ax, grid, mask, x, y, vmax, title, busan):
    ax.imshow(np.where(mask, 1.0, np.nan), origin="lower", cmap="Greys", vmin=0, vmax=3,
              extent=(x[0], x[-1], y[0], y[-1]))  # never-reached land = light grey
    im = ax.imshow(grid, origin="lower", cmap="viridis_r", vmin=0, vmax=vmax,
                   extent=(x[0], x[-1], y[0], y[-1]))
    levels = [lv for lv in CONTOUR_YEARS if lv < vmax]
    if levels and np.isfinite(grid).sum() > 0:
        cs = ax.contour(x, y, grid, levels=levels, colors="white", linewidths=0.6)
        ax.clabel(cs, fmt="%d yr", fontsize=7)
    if busan:
        ax.plot(*busan, marker="*", color="red", markersize=10, markeredgecolor="k")
    ax.set_title(title)
    ax.set_xlabel("x (km, EPSG:5179)")
    ax.set_ylabel("y (km)")
    ax.set_aspect("equal")
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="m3, m4, m5, m6 or an outputs/ subdir name")
    p.add_argument("--threshold", type=float, default=SPREAD_BOUNDARY_ABSOLUTE)
    args = p.parse_args()
    if len(args.runs) > 2:
        p.error("give one or two runs")

    runs = [load_run(r, args.threshold) for r in args.runs]
    labels = [LABELS.get(r, r) for r in args.runs]
    busan = busan_xy_km()
    vmax = max(r[4] for r in runs)
    thr = f"u ≥ {args.threshold:.0e}"

    ncols = len(runs) + (1 if len(runs) == 2 else 0)
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 5.8), squeeze=False)
    axes = axes[0]

    for ax, name, (grid, mask, x, y, _, _) in zip(axes, labels, runs):
        im = draw_arrival(ax, grid, mask, x, y, vmax, f"{name}: 도달 시간 ({thr})", busan)
        if ax is not axes[0]:
            ax.set_ylabel("")
    fig.colorbar(im, ax=list(axes[:len(runs)]), shrink=0.8, label="첫 도달 시간 (년)")

    if len(runs) == 2:
        (ga, mask, x, y, _, _), (gb, *_rest) = runs
        delay = ga - gb
        lim = np.nanpercentile(np.abs(delay), 99) if np.isfinite(delay).any() else 1.0
        ax = axes[2]
        ax.imshow(np.where(mask, 1.0, np.nan), origin="lower", cmap="Greys", vmin=0, vmax=3,
                  extent=(x[0], x[-1], y[0], y[-1]))
        im2 = ax.imshow(delay, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim,
                        extent=(x[0], x[-1], y[0], y[-1]))
        if busan:
            ax.plot(*busan, marker="*", color="red", markersize=10, markeredgecolor="k")
        ax.set_title(f"지연: {labels[0]} - {labels[1]} (년)")
        ax.set_xlabel("x (km, EPSG:5179)")
        ax.set_aspect("equal")
        fig.colorbar(im2, ax=ax, shrink=0.8, label="도달 지연 (년, +: 늦게 도달)")
        finite = np.isfinite(delay)
        if finite.any():
            print(f"delay {args.runs[0]}-{args.runs[1]}: median {np.nanmedian(delay):.2f} yr, "
                  f"max {np.nanmax(delay):.2f} yr over {finite.sum()} cells reached by both")
        out = ROOT / "outputs" / f"arrival_time_{labels[0]}_vs_{labels[1]}.png"
    else:
        out = runs[0][5] / "arrival_time.png"

    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to:\n{out}")
    plt.show()


if __name__ == "__main__":
    main()
