from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "korea_dem_500m.csv"
)

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

ARRAY_FILE = OUTPUT_DIR / "korea_dem_500m.npy"
FIGURE_FILE = OUTPUT_DIR / "korea_dem_500m.png"


# ============================================================
# Load DEM
# ============================================================

def load_dem():

    print(f"Loading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "ix",
        "iy",
        "x",
        "y",
        "elevation",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    return df


# ============================================================
# Convert CSV -> 2D DEM array
# ============================================================

def make_dem_array(df):

    ix_min = int(df["ix"].min())
    ix_max = int(df["ix"].max())

    iy_min = int(df["iy"].min())
    iy_max = int(df["iy"].max())

    nx = ix_max - ix_min + 1
    ny = iy_max - iy_min + 1

    print()
    print("DEM grid")
    print("-----------------------------")
    print(f"ix range : {ix_min} ~ {ix_max}")
    print(f"iy range : {iy_min} ~ {iy_max}")
    print(f"nx       : {nx}")
    print(f"ny       : {ny}")
    print(f"shape    : ({ny}, {nx})")

    # NaN = no DEM data
    dem = np.full(
        (ny, nx),
        np.nan,
        dtype=np.float32,
    )

    rows = (
        df["iy"].to_numpy(dtype=np.int64)
        - iy_min
    )

    cols = (
        df["ix"].to_numpy(dtype=np.int64)
        - ix_min
    )

    dem[rows, cols] = (
        df["elevation"]
        .to_numpy(dtype=np.float32)
    )

    return dem, ix_min, iy_min


# ============================================================
# Get coordinate axes
# ============================================================

def get_coordinates(df, ix_min, iy_min, dem):

    nx = dem.shape[1]
    ny = dem.shape[0]

    x = (
        df.groupby("ix")["x"]
        .first()
        .reindex(
            range(
                ix_min,
                ix_min + nx
            )
        )
        .to_numpy()
    )

    y = (
        df.groupby("iy")["y"]
        .first()
        .reindex(
            range(
                iy_min,
                iy_min + ny
            )
        )
        .to_numpy()
    )

    return x, y


# ============================================================
# Plot DEM
# ============================================================

def plot_dem(dem, x, y):

    fig, ax = plt.subplots(
        figsize=(12, 9)
    )

    # 500 m grid
    dx = 500.0
    dy = 500.0

    extent = [
        x[0] - dx / 2,
        x[-1] + dx / 2,
        y[0] - dy / 2,
        y[-1] + dy / 2,
    ]

    image = ax.imshow(
        dem,
        origin="lower",
        extent=extent,
        aspect="equal",
        interpolation="nearest",
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
        FIGURE_FILE,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print()
    print(f"Figure saved to:")
    print(FIGURE_FILE)


# ============================================================
# Main
# ============================================================

def main():

    print(
        "========================================"
    )
    print(
        "500 m DEM Grid + Figure"
    )
    print(
        "========================================"
    )

    # 1. Load CSV
    df = load_dem()

    # 2. CSV -> 2D array
    dem, ix_min, iy_min = make_dem_array(df)

    # 3. Coordinates
    x, y = get_coordinates(
        df,
        ix_min,
        iy_min,
        dem,
    )

    # 4. Save numpy array
    np.save(
        ARRAY_FILE,
        dem,
    )

    print()
    print(f"2D DEM saved to:")
    print(ARRAY_FILE)

    # 5. Statistics
    valid = dem[
        ~np.isnan(dem)
    ]

    print()
    print("Elevation statistics")
    print("-----------------------------")
    print(f"Min    : {valid.min():.2f} m")
    print(f"Max    : {valid.max():.2f} m")
    print(f"Mean   : {valid.mean():.2f} m")
    print(f"Median : {np.median(valid):.2f} m")

    print()
    print(
        f"Valid cells : {len(valid):,}"
    )

    print(
        f"NaN cells   : "
        f"{np.isnan(dem).sum():,}"
    )

    # 6. Figure
    plot_dem(
        dem,
        x,
        y,
    )


if __name__ == "__main__":
    main()