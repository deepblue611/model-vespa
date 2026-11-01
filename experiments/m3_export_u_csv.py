"""
Export per-cell u(t) from a completed m3_vespa_competition.py run to CSV.

Reads outputs/m3_vespa_competition/u_fields.npz (written by that script at
the end of its solve() call) and writes u_fields.csv next to it: one row
per land cell, indexed by the cell center's (lon, lat) in EPSG:4326, and
one column per saved frame (t_<day>d). Sea cells are omitted (u is
always 0 there).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

OUTPUT_DIR = ROOT / "outputs" / "m3_vespa_competition"
U_FIELDS_PATH = OUTPUT_DIR / "u_fields.npz"
CSV_OUTPUT_PATH = OUTPUT_DIR / "u_fields.csv"

LON_LAT_CRS = "EPSG:4326"
WORKING_CRS = "EPSG:5179"


def main():
    if not U_FIELDS_PATH.exists():
        raise FileNotFoundError(
            f"{U_FIELDS_PATH} not found -- run experiments/m3_vespa_competition.py "
            f"first (it saves this file at the end of the simulation)."
        )

    data = np.load(U_FIELDS_PATH)
    times = data["times"]
    land_mask = data["land_mask"]
    u_land = data["u_land"]   # (n_frames, n_land_cells)

    # Land cell centers, in the same row-major order as solutions[:, land_mask].
    rows, cols = np.nonzero(land_mask)
    x = data["x_centers"][cols]
    y = data["y_centers"][rows]

    transformer = Transformer.from_crs(WORKING_CRS, LON_LAT_CRS, always_xy=True)
    lon, lat = transformer.transform(x, y)

    time_columns = [f"t_{round(t)}d" for t in times]
    # lon/lat pre-formatted as strings so to_csv's %.6g (meant for u) doesn't
    # truncate them to ~100 m; 6 decimals ~= 0.1 m, well below grid spacing.
    table = pd.DataFrame(u_land.T, columns=time_columns)
    table.insert(0, "lat", [f"{v:.6f}" for v in lat])
    table.insert(0, "lon", [f"{v:.6f}" for v in lon])

    print(f"Writing {len(table)} cells x {len(time_columns)} frames...")
    table.to_csv(CSV_OUTPUT_PATH, index=False, float_format="%.6g")
    print(f"Saved to:\n{CSV_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
