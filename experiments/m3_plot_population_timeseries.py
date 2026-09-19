"""
Plot total population (sum of u over land cells) vs. time from a completed
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
OUTPUT_PATH = ROOT / "outputs" / "m3_vespa_competition" / "population_timeseries.png"


def main():
    if not TIMESERIES_PATH.exists():
        raise FileNotFoundError(
            f"{TIMESERIES_PATH} not found -- run experiments/m3_vespa_competition.py "
            f"first (it saves this file at the end of the simulation)."
        )

    data = np.load(TIMESERIES_PATH)
    times_years = data["times"] / 365.0
    total_population = data["total_population"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times_years, total_population, marker="o", markersize=3)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Total population (sum of u over land)")
    ax.set_title("V. velutina total population vs. time")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"Saved to:\n{OUTPUT_PATH}")

    plt.show()


if __name__ == "__main__":
    main()
