import matplotlib.pyplot as plt
import numpy as np


def plot_solution(
    x: np.ndarray,
    times: np.ndarray,
    solutions: np.ndarray,
    selected_times: list[float] | None = None,
) -> None:
    """
    Plot population density at selected times.
    """

    if selected_times is None:
        selected_times = [
            times[0],
            times[len(times) // 3],
            times[2 * len(times) // 3],
            times[-1],
        ]

    plt.figure(figsize=(10, 6))

    for target_time in selected_times:
        index = np.argmin(np.abs(times - target_time))

        plt.plot(
            x,
            solutions[index],
            label=f"t = {times[index]:.2f}",
        )

    plt.xlabel("x")
    plt.ylabel("Population density")
    plt.title("1D Fisher-KPP model")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()