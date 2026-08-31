from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "korea-alt"


def main():

    files = sorted(RAW_DIR.glob("*.csv"))

    print(f"Found {len(files)} CSV files\n")

    global_min_x = float("inf")
    global_max_x = float("-inf")
    global_min_y = float("inf")
    global_max_y = float("-inf")

    for file in files:

        df = pd.read_csv(
            file,
            usecols=["X", "Y"],
        )

        min_x = df["X"].min()
        max_x = df["X"].max()
        min_y = df["Y"].min()
        max_y = df["Y"].max()

        global_min_x = min(global_min_x, min_x)
        global_max_x = max(global_max_x, max_x)

        global_min_y = min(global_min_y, min_y)
        global_max_y = max(global_max_y, max_y)

        print(
            f"{file.name:25s} "
            f"X: {min_x:.4f} ~ {max_x:.4f}  "
            f"Y: {min_y:.4f} ~ {max_y:.4f}"
        )

    print("\n========================================")
    print("GLOBAL EXTENT")
    print("========================================")

    print(
        f"X: {global_min_x:.6f} ~ "
        f"{global_max_x:.6f}"
    )

    print(
        f"Y: {global_min_y:.6f} ~ "
        f"{global_max_y:.6f}"
    )


if __name__ == "__main__":
    main()