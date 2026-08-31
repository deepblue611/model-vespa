from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "korea-alt"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "korea_dem_500m.csv"
)

INPUT_CRS = "EPSG:4326"   # longitude / latitude
OUTPUT_CRS = "EPSG:5179"  # Korea 2000 / Unified CS

GRID_SIZE = 500.0         # 500 m × 500 m
CHUNK_SIZE = 200_000


# ============================================================
# File discovery
# ============================================================

def get_input_files():
    """
    Find korea-alt 1~45 CSV files.

    Supports both:
        korea-alt (1).csv
        korea-alt 1.csv
    """

    files = []

    for i in range(1, 46):
        candidates = [
            RAW_DIR / f"korea-alt ({i}).csv",
            #RAW_DIR / f"korea-alt {i}.csv",
        ]

        found = next((p for p in candidates if p.exists()), None)

        if found is None:
            raise FileNotFoundError(
                f"Could not find DEM file #{i}. "
                f"Expected one of: {candidates}"
            )

        files.append(found)

    return files


# ============================================================
# Coordinate transformation
# ============================================================

def transform_coordinates(x, y, transformer):
    """
    Transform longitude/latitude to projected coordinates.
    """

    x_projected, y_projected = transformer.transform(x, y)

    return (
        np.asarray(x_projected),
        np.asarray(y_projected),
    )


# ============================================================
# Find global spatial extent
# ============================================================

def find_global_extent(files, transformer):
    """
    Find the global projected-coordinate extent of all 45 DEM tiles.
    """

    min_x = np.inf
    max_x = -np.inf
    min_y = np.inf
    max_y = -np.inf

    for file in files:
        print(f"[Extent] {file.name}")

        df = pd.read_csv(
            file,
            usecols=["X", "Y"],
        )

        x, y = transform_coordinates(
            df["X"].to_numpy(),
            df["Y"].to_numpy(),
            transformer,
        )

        min_x = min(min_x, np.min(x))
        max_x = max(max_x, np.max(x))
        min_y = min(min_y, np.min(y))
        max_y = max(max_y, np.max(y))

    return min_x, max_x, min_y, max_y


# ============================================================
# Downscale
# ============================================================

def downscale_dem(files, transformer):
    """
    Aggregate 90 m DEM points into 500 m × 500 m cells.

    For each 500 m cell:

        elevation = mean(elevation of all 90 m points)

    Sea-level values (elevation == 0) are preserved.
    """

    print("\nFinding global extent...")

    min_x, max_x, min_y, max_y = find_global_extent(
        files,
        transformer,
    )

    # Align the grid to 500 m boundaries.
    origin_x = np.floor(min_x / GRID_SIZE) * GRID_SIZE
    origin_y = np.floor(min_y / GRID_SIZE) * GRID_SIZE

    print("\nGlobal extent:")
    print(f"  X: {min_x:.2f} ~ {max_x:.2f}")
    print(f"  Y: {min_y:.2f} ~ {max_y:.2f}")

    print("\n500 m grid origin:")
    print(f"  X0 = {origin_x:.2f}")
    print(f"  Y0 = {origin_y:.2f}")

    # --------------------------------------------------------
    # Accumulators
    #
    # key = (grid_x, grid_y)
    # value = [sum of elevation, number of observations]
    #
    # This allows us to process the 45 files chunk-by-chunk.
    # --------------------------------------------------------

    sums = {}
    counts = {}

    total_rows = 0

    for file_index, file in enumerate(files, start=1):

        print(
            f"\n[{file_index:02d}/45] "
            f"Processing {file.name}"
        )

        for chunk in pd.read_csv(
            file,
            usecols=["X", "Y", "elevation"],
            chunksize=CHUNK_SIZE,
        ):

            total_rows += len(chunk)

            x, y = transform_coordinates(
                chunk["X"].to_numpy(),
                chunk["Y"].to_numpy(),
                transformer,
            )

            elevation = chunk["elevation"].to_numpy(
                dtype=np.float64
            )

            # ------------------------------------------------
            # Convert projected coordinates to 500 m grid index
            # ------------------------------------------------

            ix = np.floor(
                (x - origin_x) / GRID_SIZE
            ).astype(np.int64)

            iy = np.floor(
                (y - origin_y) / GRID_SIZE
            ).astype(np.int64)

            # ------------------------------------------------
            # Aggregate within this chunk
            # ------------------------------------------------

            temp = pd.DataFrame({
                "ix": ix,
                "iy": iy,
                "elevation": elevation,
            })

            grouped = temp.groupby(
                ["ix", "iy"],
                sort=False,
            )["elevation"].agg(
                ["sum", "count"]
            )

            # ------------------------------------------------
            # Merge chunk result into global accumulator
            # ------------------------------------------------

            for (gx, gy), row in grouped.iterrows():

                key = (int(gx), int(gy))

                if key not in sums:
                    sums[key] = 0.0
                    counts[key] = 0

                sums[key] += row["sum"]
                counts[key] += int(row["count"])

        print(f"  Processed rows: {total_rows:,}")

    # ========================================================
    # Construct final DataFrame
    # ========================================================

    print("\nConstructing 500 m DEM...")

    rows = []

    for (ix, iy), total_elevation in sums.items():

        count = counts[(ix, iy)]

        mean_elevation = total_elevation / count

        # Center of the 500 m cell
        center_x = (
            origin_x
            + (ix + 0.5) * GRID_SIZE
        )

        center_y = (
            origin_y
            + (iy + 0.5) * GRID_SIZE
        )

        rows.append({
            "ix": ix,
            "iy": iy,
            "x": center_x,
            "y": center_y,
            "elevation": mean_elevation,
            "n_points": count,
        })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        ["iy", "ix"]
    ).reset_index(drop=True)

    return result


# ============================================================
# Main
# ============================================================

def main():

    print("=== DEM 90m -> 500m preprocessing ===")

    files = get_input_files()

    print(f"\nFound {len(files)} DEM files.")

    transformer = Transformer.from_crs(
        INPUT_CRS,
        OUTPUT_CRS,
        always_xy=True,
    )

    result = downscale_dem(
        files,
        transformer,
    )

    # Create output directory
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n========================================")
    print("Finished!")
    print("========================================")

    print(f"Output: {OUTPUT_FILE}")
    print(f"Number of 500m cells: {len(result):,}")

    print("\nColumns:")
    print(result.columns.tolist())

    print("\nElevation statistics:")
    print(result["elevation"].describe())


if __name__ == "__main__":
    main()