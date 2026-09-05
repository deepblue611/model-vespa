from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Complete DEM
INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "korea-alt-complete"
    / "korea_altitude_complete.csv"
)

# Output directory
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dem_500m"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_CSV = OUTPUT_DIR / "korea_dem_500m.csv"
OUTPUT_NPY = OUTPUT_DIR / "korea_dem_500m.npy"
OUTPUT_FIGURE = OUTPUT_DIR / "korea_dem_500m.png"


# Coordinate systems
INPUT_CRS = "EPSG:4326"
OUTPUT_CRS = "EPSG:5179"

# Target grid resolution
GRID_SIZE = 500.0

# Number of source rows read at once
CHUNK_SIZE = 500_000


# ============================================================
# Find columns
# ============================================================

def find_columns():

    sample = pd.read_csv(
        INPUT_FILE,
        nrows=5
    )

    print("Input columns:")
    print(sample.columns.tolist())

    columns = {}

    for col in sample.columns:

        name = col.strip().lower()

        if name in {
            "x",
            "lon",
            "longitude"
        }:
            columns["x"] = col

        elif name in {
            "y",
            "lat",
            "latitude"
        }:
            columns["y"] = col

        elif name in {
            "elevation",
            "altitude",
            "elev",
            "height",
            "z"
        }:
            columns["elevation"] = col

    required = {
        "x",
        "y",
        "elevation"
    }

    missing = required - set(columns)

    if missing:
        raise ValueError(
            f"Could not identify columns: {missing}"
        )

    return columns


# ============================================================
# Find projected extent
# ============================================================

def find_extent(columns, transformer):

    print("\nFinding spatial extent...")

    min_x = np.inf
    max_x = -np.inf

    min_y = np.inf
    max_y = -np.inf

    total_rows = 0

    for chunk in pd.read_csv(
        INPUT_FILE,
        usecols=[
            columns["x"],
            columns["y"]
        ],
        chunksize=CHUNK_SIZE
    ):

        total_rows += len(chunk)

        x = chunk[
            columns["x"]
        ].to_numpy()

        y = chunk[
            columns["y"]
        ].to_numpy()

        px, py = transformer.transform(
            x,
            y
        )

        min_x = min(
            min_x,
            np.min(px)
        )

        max_x = max(
            max_x,
            np.max(px)
        )

        min_y = min(
            min_y,
            np.min(py)
        )

        max_y = max(
            max_y,
            np.max(py)
        )

    return (
        min_x,
        max_x,
        min_y,
        max_y,
        total_rows
    )


# ============================================================
# Downscale
# ============================================================

def downscale(
    columns,
    transformer,
    origin_x,
    origin_y
):
    """
    90 m DEM -> 500 m DEM

    Each 500m cell receives the mean elevation
    of all source points inside that cell.

    elevation = 0 is preserved.
    """

    print("\nDownscaling...")

    partial_results = []

    total_rows = 0

    for chunk_id, chunk in enumerate(
        pd.read_csv(
            INPUT_FILE,
            usecols=[
                columns["x"],
                columns["y"],
                columns["elevation"]
            ],
            chunksize=CHUNK_SIZE
        ),
        start=1
    ):

        total_rows += len(chunk)

        x = chunk[
            columns["x"]
        ].to_numpy()

        y = chunk[
            columns["y"]
        ].to_numpy()

        elevation = pd.to_numeric(
            chunk[
                columns["elevation"]
            ],
            errors="coerce"
        ).to_numpy(
            dtype=np.float64
        )

        # Coordinate transformation
        px, py = transformer.transform(
            x,
            y
        )

        # Remove invalid rows only
        valid = (
            np.isfinite(px)
            & np.isfinite(py)
            & np.isfinite(elevation)
        )

        px = px[valid]
        py = py[valid]
        elevation = elevation[valid]

        # 500m grid indices
        ix = np.floor(
            (px - origin_x)
            / GRID_SIZE
        ).astype(np.int64)

        iy = np.floor(
            (py - origin_y)
            / GRID_SIZE
        ).astype(np.int64)

        temp = pd.DataFrame({
            "iy": iy,
            "ix": ix,
            "elevation": elevation
        })

        # Aggregate this chunk
        grouped = (
            temp
            .groupby(
                ["iy", "ix"],
                sort=False
            )["elevation"]
            .agg(
                elevation_sum="sum",
                n_points="count"
            )
            .reset_index()
        )

        partial_results.append(grouped)

        print(
            f"Chunk {chunk_id:>4}: "
            f"{total_rows:,} rows"
        )

    # --------------------------------------------------------
    # Merge partial results
    # --------------------------------------------------------

    print("\nMerging aggregated chunks...")

    combined = pd.concat(
        partial_results,
        ignore_index=True
    )

    # Same 500m cells can occur in different chunks.
    result = (
        combined
        .groupby(
            ["iy", "ix"],
            sort=False
        )
        .agg(
            elevation_sum=(
                "elevation_sum",
                "sum"
            ),
            n_points=(
                "n_points",
                "sum"
            )
        )
        .reset_index()
    )

    # Mean elevation
    result["elevation"] = (
        result["elevation_sum"]
        / result["n_points"]
    )

    # Cell center coordinates
    result["x"] = (
        origin_x
        + (result["ix"] + 0.5)
        * GRID_SIZE
    )

    result["y"] = (
        origin_y
        + (result["iy"] + 0.5)
        * GRID_SIZE
    )

    result = result[
        [
            "ix",
            "iy",
            "x",
            "y",
            "elevation",
            "n_points"
        ]
    ]

    result = result.sort_values(
        ["iy", "ix"]
    ).reset_index(drop=True)

    return result


# ============================================================
# Convert to 2D array
# ============================================================

def make_2d_dem(
    df,
    origin_x,
    origin_y
):

    ix_min = int(
        df["ix"].min()
    )

    ix_max = int(
        df["ix"].max()
    )

    iy_min = int(
        df["iy"].min()
    )

    iy_max = int(
        df["iy"].max()
    )

    nx = ix_max - ix_min + 1
    ny = iy_max - iy_min + 1

    print("\n2D grid")
    print("--------------------------------")
    print(
        f"X cells : {nx:,}"
    )
    print(
        f"Y cells : {ny:,}"
    )
    print(
        f"Shape   : ({ny}, {nx})"
    )
    print(
        f"Total cells : {nx * ny:,}"
    )
    print(
        f"Observed cells : {len(df):,}"
    )

    # NaN = no source data
    dem = np.full(
        (ny, nx),
        np.nan,
        dtype=np.float32
    )

    rows = (
        df["iy"].to_numpy(
            dtype=np.int64
        )
        - iy_min
    )

    cols = (
        df["ix"].to_numpy(
            dtype=np.int64
        )
        - ix_min
    )

    dem[
        rows,
        cols
    ] = df[
        "elevation"
    ].to_numpy(
        dtype=np.float32
    )

    # Coordinate axes
    x = (
        origin_x
        + (
            np.arange(
                ix_min,
                ix_max + 1
            )
            + 0.5
        )
        * GRID_SIZE
    )

    y = (
        origin_y
        + (
            np.arange(
                iy_min,
                iy_max + 1
            )
            + 0.5
        )
        * GRID_SIZE
    )

    return dem, x, y


# ============================================================
# Plot
# ============================================================

def make_figure(
    dem,
    x,
    y
):

    print("\nCreating DEM figure...")

    extent = [
        x[0] - GRID_SIZE / 2,
        x[-1] + GRID_SIZE / 2,
        y[0] - GRID_SIZE / 2,
        y[-1] + GRID_SIZE / 2
    ]

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    image = ax.imshow(
        dem,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest"
    )

    ax.set_xlabel(
        "X coordinate (m)"
    )

    ax.set_ylabel(
        "Y coordinate (m)"
    )

    ax.set_title(
        "Korea Digital Elevation Model\n"
        "500 m × 500 m Grid"
    )

    colorbar = fig.colorbar(
        image,
        ax=ax
    )

    colorbar.set_label(
        "Elevation (m)"
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_FIGURE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"Figure saved to:\n"
        f"{OUTPUT_FIGURE}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print(
        "============================================"
    )
    print(
        "Korea DEM 90m -> 500m"
    )
    print(
        "============================================"
    )

    # Check input
    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nInput DEM not found:\n"
            f"{INPUT_FILE}\n\n"
            f"Expected location:\n"
            f"data/raw/korea-altitude/"
        )

    print(
        f"\nInput:\n{INPUT_FILE}"
    )

    # Coordinate transformer
    transformer = Transformer.from_crs(
        INPUT_CRS,
        OUTPUT_CRS,
        always_xy=True
    )

    # --------------------------------------------------------
    # 1. Columns
    # --------------------------------------------------------

    columns = find_columns()

    print(
        "\nDetected columns:"
    )

    print(
        f"X         = {columns['x']}"
    )

    print(
        f"Y         = {columns['y']}"
    )

    print(
        f"Elevation = {columns['elevation']}"
    )

    # --------------------------------------------------------
    # 2. Extent
    # --------------------------------------------------------

    (
        min_x,
        max_x,
        min_y,
        max_y,
        total_rows
    ) = find_extent(
        columns,
        transformer
    )

    # Align grid to 500m boundaries
    origin_x = (
        np.floor(
            min_x / GRID_SIZE
        )
        * GRID_SIZE
    )

    origin_y = (
        np.floor(
            min_y / GRID_SIZE
        )
        * GRID_SIZE
    )

    print(
        f"\nSource rows: {total_rows:,}"
    )

    print(
        f"Projected X: "
        f"{min_x:.2f} ~ {max_x:.2f}"
    )

    print(
        f"Projected Y: "
        f"{min_y:.2f} ~ {max_y:.2f}"
    )

    print(
        f"500m grid origin: "
        f"({origin_x:.2f}, {origin_y:.2f})"
    )

    # --------------------------------------------------------
    # 3. Downscale
    # --------------------------------------------------------

    dem_500 = downscale(
        columns,
        transformer,
        origin_x,
        origin_y
    )

    dem_500.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print(
        f"\n500m DEM saved:\n"
        f"{OUTPUT_CSV}"
    )

    # --------------------------------------------------------
    # 4. 2D array
    # --------------------------------------------------------

    dem, x, y = make_2d_dem(
        dem_500,
        origin_x,
        origin_y
    )

    np.save(
        OUTPUT_NPY,
        dem
    )

    print(
        f"\n2D DEM saved:\n"
        f"{OUTPUT_NPY}"
    )

    # --------------------------------------------------------
    # 5. Statistics
    # --------------------------------------------------------

    valid = dem[
        np.isfinite(dem)
    ]

    print(
        "\nElevation statistics"
    )
    print("--------------------------------")

    print(
        f"Minimum : "
        f"{np.min(valid):.2f} m"
    )

    print(
        f"Maximum : "
        f"{np.max(valid):.2f} m"
    )

    print(
        f"Mean    : "
        f"{np.mean(valid):.2f} m"
    )

    print(
        f"Median  : "
        f"{np.median(valid):.2f} m"
    )

    print(
        f"NaN cells : "
        f"{np.isnan(dem).sum():,}"
    )

    # --------------------------------------------------------
    # 6. Figure
    # --------------------------------------------------------

    make_figure(
        dem,
        x,
        y
    )

    print(
        "\n============================================"
    )

    print(
        "Finished."
    )

    print(
        "============================================"
    )


if __name__ == "__main__":
    main()