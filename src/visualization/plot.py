import numpy as np
import matplotlib.pyplot as plt


def plot_population(
    u: np.ndarray,
    time: float | None = None,
    title: str = "Population density"
):
    """
    Plot a 2D population density field.
    """

    plt.figure(figsize=(7, 6))

    plt.imshow(
        u,
        origin="lower",
        interpolation="nearest",
        aspect="equal"
    )

    plt.colorbar(
        label="Population density"
    )

    if time is not None:
        title = f"{title} (t = {time:.2f})"

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.tight_layout()
    plt.show()


def plot_snapshots(
    times: np.ndarray,
    solutions: np.ndarray,
    indices: list[int]
):
    """
    Plot several snapshots of the simulation.

    Parameters
    ----------
    times :
        Saved simulation times.

    solutions :
        Array with shape (T, Ny, Nx).

    indices :
        Indices of snapshots to display.
    """

    for index in indices:

        if index < 0 or index >= len(solutions):
            raise IndexError(
                f"Snapshot index {index} is out of range."
            )

        plot_population(
            solutions[index],
            time=times[index]
        )