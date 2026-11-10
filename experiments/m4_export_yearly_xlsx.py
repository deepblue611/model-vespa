"""
Export a completed m4_vespa_no_competition.py run to Excel at 1-year intervals.

Reads outputs/m4_vespa_no_competition/{u_fields,timeseries}.npz and writes
m4_yearly.xlsx next to them. Frames are saved every SAVE_INTERVAL_DAYS (90),
so for each year k the frame nearest to day 365*k is used; its actual day is
recorded in the "summary" sheet.

Sheets:
  summary -- one row per year: year, actual day, total population,
             spread area, max spread distance from Busan.
  u       -- one row per land cell (lon, lat), one column per year (u_<k>yr).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

OUTPUT_DIR = ROOT / "outputs" / "m4_vespa_no_competition"
U_FIELDS_PATH = OUTPUT_DIR / "u_fields.npz"
TIMESERIES_PATH = OUTPUT_DIR / "timeseries.npz"
XLSX_OUTPUT_PATH = OUTPUT_DIR / "m4_yearly.xlsx"

DAYS_PER_YEAR = 365.0
LON_LAT_CRS = "EPSG:4326"
WORKING_CRS = "EPSG:5179"


def yearly_frame_indices(times):
    """Index of the saved frame nearest to each whole year within the run."""
    n_years = int(round(times[-1] / DAYS_PER_YEAR))
    years = np.arange(n_years + 1)
    idx = np.abs(times[None, :] - years[:, None] * DAYS_PER_YEAR).argmin(axis=1)
    return years, idx


def main():
    for path in (U_FIELDS_PATH, TIMESERIES_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found -- run experiments/m4_vespa_no_competition.py first."
            )

    fields = np.load(U_FIELDS_PATH)
    ts = np.load(TIMESERIES_PATH)
    times = fields["times"]
    years, idx = yearly_frame_indices(times)

    summary = pd.DataFrame(
        {
            "year": years,
            "day": np.round(times[idx], 2),
            "total_population": ts["total_population"][idx],
            "spread_area_km2": ts["spread_area_km2"][idx],
            "spread_max_distance_km": ts["spread_max_distance_km"][idx],
        }
    )

    land_mask = fields["land_mask"]
    rows, cols = np.nonzero(land_mask)
    transformer = Transformer.from_crs(WORKING_CRS, LON_LAT_CRS, always_xy=True)
    lon, lat = transformer.transform(fields["x_centers"][cols], fields["y_centers"][rows])

    u = pd.DataFrame(fields["u_land"][idx].T, columns=[f"u_{k}yr" for k in years])
    u.insert(0, "lat", np.round(lat, 6))
    u.insert(0, "lon", np.round(lon, 6))

    print(f"Writing {len(years)} yearly frames x {len(u)} land cells...")
    with pd.ExcelWriter(XLSX_OUTPUT_PATH, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        u.to_excel(writer, sheet_name="u", index=False)
    print(f"Saved to:\n{XLSX_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
