import numpy as np


class FiniteDifference2D:
    """
    Explicit finite-difference solver for a 2D reaction-diffusion PDE.
    """

    def __init__(
        self,
        dx: float,
        dy: float,
        dt: float
    ):
        if dx <= 0 or dy <= 0:
            raise ValueError("dx and dy must be positive.")

        if dt <= 0:
            raise ValueError("dt must be positive.")

        self.dx = dx
        self.dy = dy
        self.dt = dt

    def laplacian(self, u: np.ndarray) -> np.ndarray:
        """
        Calculate the 2D Laplacian using central finite differences.

        ∇²u =
            d²u/dx² + d²u/dy²

        Boundary condition:
            zero-flux (Neumann)
        """

        # Neumann boundary condition:
        # replicate boundary values
        padded = np.pad(
            u,
            pad_width=1,
            mode="edge"
        )

        d2x = (
            padded[1:-1, 2:]
            - 2.0 * u
            + padded[1:-1, :-2]
        ) / (self.dx ** 2)

        d2y = (
            padded[2:, 1:-1]
            - 2.0 * u
            + padded[:-2, 1:-1]
        ) / (self.dy ** 2)

        return d2x + d2y

    def step(self, u: np.ndarray, model) -> np.ndarray:
        """
        Perform one explicit Euler time step.
        """

        laplacian_u = self.laplacian(u)

        du_dt = model.rhs(
            u,
            laplacian_u
        )

        u_next = u + self.dt * du_dt

        # Numerical errors should not create negative population.
        u_next = np.maximum(u_next, 0.0)

        return u_next

    def solve(
        self,
        u0: np.ndarray,
        model,
        steps: int,
        save_every: int = 1
    ):
        """
        Run the simulation.

        Returns
        -------
        times : np.ndarray
        solutions : np.ndarray
            Shape:
                (number_of_saved_steps, Ny, Nx)
        """

        if steps <= 0:
            raise ValueError("steps must be positive.")

        u = np.array(u0, dtype=float, copy=True)

        solutions = [u.copy()]
        times = [0.0]

        for step in range(1, steps + 1):

            u = self.step(u, model)

            if step % save_every == 0:
                solutions.append(u.copy())
                times.append(step * self.dt)

        return (
            np.array(times),
            np.array(solutions)
        )