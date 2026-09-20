"""
Plot total population and spread area vs. time from a completed
m3_vespa_competition.py run.

Reads outputs/m3_vespa_competition/timeseries.npz, written by that script
at the end of its solve() call.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

TIMESERIES_PATH = ROOT / "outputs" / "m3_vespa_competition" / "timeseries.npz"
POPULATION_OUTPUT_PATH = ROOT / "outputs" / "m3_vespa_competition" / "population_timeseries.png"
AREA_OUTPUT_PATH = ROOT / "outputs" / "m3_vespa_competition" / "spread_area_timeseries.png"
DISTANCE_OUTPUT_PATH = ROOT / "outputs" / "m3_vespa_competition" / "spread_distance_timeseries.png"
SPEED_OUTPUT_PATH = ROOT / "outputs" / "m3_vespa_competition" / "spread_speed_timeseries.png"


def plot_population(times_years, total_population):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times_years, total_population, marker="o", markersize=3)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Total population (sum of u over land)")
    ax.set_title("V. velutina total population vs. time")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(POPULATION_OUTPUT_PATH, dpi=150)
    print(f"Saved to:\n{POPULATION_OUTPUT_PATH}")


def plot_spread_area(times_years, spread_area_km2, boundary_fraction):
    """Plot spread area vs. time with a linear trend line (km^2/yr)."""

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times_years, spread_area_km2, marker="o", markersize=3, label="Spread area")

    valid = spread_area_km2 > 0
    if valid.sum() >= 2:
        area_slope, area_intercept = np.polyfit(times_years[valid], spread_area_km2[valid], 1)
        ax.plot(
            times_years, area_slope * times_years + area_intercept,
            linestyle="--", color="gray",
            label=f"Linear trend: {area_slope:.2f} km$^2$/yr",
        )
        print(f"Spread area trend: {area_slope:.3f} km^2/yr")
    else:
        print("Not enough nonzero-area frames to fit a trend.")

    ax.set_xlabel("Time (years)")
    ax.set_ylabel(f"Spread area (km$^2$, u >= {boundary_fraction:.0%} of that frame's max)")
    ax.set_title("Spread area vs. time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(AREA_OUTPUT_PATH, dpi=150)
    print(f"Saved to:\n{AREA_OUTPUT_PATH}")


def plot_spread_distance(times_years, spread_max_distance_km, boundary_fraction):
    """
    Plot max distance from Busan (among cells crossing the spread
    boundary) vs. time, with a linear trend line -- this is the actual
    front reach, unlike sqrt(area/pi): Busan sits at a coastal corner of
    the domain, so an area-equivalent radius assumes a full circle around
    it when much of that circle is ocean, systematically underestimating
    true reach. The trend's slope is the front-speed estimate (km/yr).
    """

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times_years, spread_max_distance_km, marker="o", markersize=3, label="Max distance from Busan")

    valid = spread_max_distance_km > 0
    if valid.sum() >= 2:
        speed_km_per_year, intercept = np.polyfit(times_years[valid], spread_max_distance_km[valid], 1)
        ax.plot(
            times_years, speed_km_per_year * times_years + intercept,
            linestyle="--", color="gray",
            label=f"Linear trend (front speed): {speed_km_per_year:.2f} km/yr",
        )
        print(f"Estimated front speed (max-distance-from-Busan linear fit): {speed_km_per_year:.3f} km/yr")
    else:
        print("Not enough nonzero-distance frames to fit a trend.")

    ax.set_xlabel("Time (years)")
    ax.set_ylabel(f"Max distance from Busan (km, u >= {boundary_fraction:.0%} of that frame's max)")
    ax.set_title("Spread distance from Busan vs. time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(DISTANCE_OUTPUT_PATH, dpi=150)
    print(f"Saved to:\n{DISTANCE_OUTPUT_PATH}")


def plot_spread_speed(times_years, spread_max_distance_km):
    """
    Plot instantaneous front speed vs. time: d(max distance from Busan)/dt
    via np.gradient (central differences), so the speed curve itself can
    be inspected for acceleration/deceleration rather than collapsing the
    whole run into one overall trend number.
    """

    valid = spread_max_distance_km > 0
    if valid.sum() < 2:
        print("Not enough nonzero-distance frames to compute a speed curve.")
        return

    t = times_years[valid]
    distance_km = spread_max_distance_km[valid]
    speed_km_per_year = np.gradient(distance_km, t)

    overall_speed, _ = np.polyfit(t, distance_km, 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, speed_km_per_year, marker="o", markersize=3, label="Instantaneous front speed")
    ax.axhline(
        overall_speed, color="gray", linestyle="--",
        label=f"Overall trend: {overall_speed:.2f} km/yr",
    )

    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Front speed (km/yr, d(max distance from Busan)/dt)")
    ax.set_title("Spread speed vs. time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(SPEED_OUTPUT_PATH, dpi=150)
    print(f"Saved to:\n{SPEED_OUTPUT_PATH}")


def main():
    if not TIMESERIES_PATH.exists():
        raise FileNotFoundError(
            f"{TIMESERIES_PATH} not found -- run experiments/m3_vespa_competition.py "
            f"first (it saves this file at the end of the simulation)."
        )

    data = np.load(TIMESERIES_PATH)
    times_years = data["times"] / 365.0
    total_population = data["total_population"]

    plot_population(times_years, total_population)

    if "spread_area_km2" in data:
        boundary_fraction = float(data["spread_boundary_fraction_of_max"])
        plot_spread_area(times_years, data["spread_area_km2"], boundary_fraction)
    else:
        print(
            "timeseries.npz has no 'spread_area_km2' -- it was saved by an older "
            "version of m3_vespa_competition.py; rerun the simulation to get the "
            "spread-area plot too."
        )

    if "spread_max_distance_km" in data:
        boundary_fraction = float(data["spread_boundary_fraction_of_max"])
        plot_spread_distance(times_years, data["spread_max_distance_km"], boundary_fraction)
        plot_spread_speed(times_years, data["spread_max_distance_km"])
    else:
        print(
            "timeseries.npz has no 'spread_max_distance_km' -- it was saved by an "
            "older version of m3_vespa_competition.py; rerun the simulation to get "
            "the distance/speed plots too."
        )

    plt.show()


if __name__ == "__main__":
    main()
