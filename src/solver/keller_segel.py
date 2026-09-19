import numpy as np


class KellerSegel2D:
    """
    Explicit finite-volume solver for a 2D reaction-diffusion-taxis PDE
    on an irregular, land/sea-masked domain (see vespa_pde_model_spec.md).

    Diffusion uses a central-difference Laplacian and the taxis term
    uses a first-order upwind flux (Patankar-type positivity-preserving
    scheme, per the spec's recommendation in section 5). Both are built
    from face fluxes that are forced to zero at any face bordering a
    masked-out (sea) cell or the array boundary -- this realizes the
    no-flux (Neumann) condition at the coastline exactly, without
    needing ghost cells.
    """

    def __init__(self, dx: float, dy: float, dt: float, xp=np):
        if dx <= 0 or dy <= 0:
            raise ValueError("dx and dy must be positive.")

        if dt <= 0:
            raise ValueError("dt must be positive.")

        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.xp = xp

    def gradient(self, field: np.ndarray):
        """
        Central-difference gradient (edge-padded at the array boundary).

        `field` is expected to already be gap-filled (no NaNs) outside
        the land mask -- see vespa_field_downscale.py -- so no masking
        is needed here; it is only used to build a static drift field.
        """

        padded = self.xp.pad(field, pad_width=1, mode="edge")

        grad_x = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2.0 * self.dx)
        grad_y = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2.0 * self.dy)

        return grad_x, grad_y

    def taxis_velocity(self, C_u: np.ndarray, chi_u: float):
        """
        Static drift velocity chi_u * grad(C_u) driving the taxis term.

        Precompute once (C_u and chi_u don't change over time) and reuse
        across every solver step.
        """

        grad_x, grad_y = self.gradient(C_u)

        return chi_u * grad_x, chi_u * grad_y

    def laplacian(self, u: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Finite-volume Laplacian: sum of face-centered gradients, with
        zero flux enforced at masked (sea) faces and the array boundary.
        """

        xp = self.xp
        ny, nx = u.shape

        diff_x = (u[:, 1:] - u[:, :-1]) / self.dx
        face_open_x = mask[:, :-1] & mask[:, 1:]
        diff_x = xp.where(face_open_x, diff_x, 0.0)

        flux_x = xp.zeros((ny, nx + 1))
        flux_x[:, 1:-1] = diff_x
        d2x = (flux_x[:, 1:] - flux_x[:, :-1]) / self.dx

        diff_y = (u[1:, :] - u[:-1, :]) / self.dy
        face_open_y = mask[:-1, :] & mask[1:, :]
        diff_y = xp.where(face_open_y, diff_y, 0.0)

        flux_y = xp.zeros((ny + 1, nx))
        flux_y[1:-1, :] = diff_y
        d2y = (flux_y[1:, :] - flux_y[:-1, :]) / self.dy

        return d2x + d2y

    def taxis_divergence(
        self,
        u: np.ndarray,
        velocity_x: np.ndarray,
        velocity_y: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        First-order upwind finite-volume approximation of
        div(u * velocity), with zero flux enforced at masked (sea)
        faces and the array boundary.
        """

        xp = self.xp
        ny, nx = u.shape

        vx_face = 0.5 * (velocity_x[:, :-1] + velocity_x[:, 1:])
        u_left = u[:, :-1]
        u_right = u[:, 1:]
        flux_x = xp.where(vx_face >= 0.0, vx_face * u_left, vx_face * u_right)

        face_open_x = mask[:, :-1] & mask[:, 1:]
        flux_x = xp.where(face_open_x, flux_x, 0.0)

        flux_x_padded = xp.zeros((ny, nx + 1))
        flux_x_padded[:, 1:-1] = flux_x
        div_x = (flux_x_padded[:, 1:] - flux_x_padded[:, :-1]) / self.dx

        vy_face = 0.5 * (velocity_y[:-1, :] + velocity_y[1:, :])
        u_bottom = u[:-1, :]
        u_top = u[1:, :]
        flux_y = xp.where(vy_face >= 0.0, vy_face * u_bottom, vy_face * u_top)

        face_open_y = mask[:-1, :] & mask[1:, :]
        flux_y = xp.where(face_open_y, flux_y, 0.0)

        flux_y_padded = xp.zeros((ny + 1, nx))
        flux_y_padded[1:-1, :] = flux_y
        div_y = (flux_y_padded[1:, :] - flux_y_padded[:-1, :]) / self.dy

        return div_x + div_y

    def step(
        self,
        u: np.ndarray,
        model,
        velocity_x: np.ndarray,
        velocity_y: np.ndarray,
    ) -> np.ndarray:
        """Perform one explicit Euler time step."""

        laplacian_u = self.laplacian(u, model.mask)
        taxis_div = self.taxis_divergence(u, velocity_x, velocity_y, model.mask)

        du_dt = model.rhs(u, laplacian_u, taxis_div)

        u_next = u + self.dt * du_dt

        # Numerical errors should not create negative population.
        u_next = self.xp.maximum(u_next, 0.0)

        return u_next

    def _to_numpy(self, arr):
        """Bring a solver-backend array (numpy or cupy) back to host numpy."""

        if self.xp is np:
            return arr.copy()

        return self.xp.asnumpy(arr)

    def solve(
        self,
        u0: np.ndarray,
        model,
        velocity_x: np.ndarray,
        velocity_y: np.ndarray,
        steps: int,
        save_every: int = 1,
        progress: bool = False,
    ):
        """
        Run the simulation.

        Returns
        -------
        times : np.ndarray
        solutions : np.ndarray
            Shape: (number_of_saved_steps, Ny, Nx)
        """

        if steps <= 0:
            raise ValueError("steps must be positive.")

        xp = self.xp
        u = xp.array(u0, dtype=float, copy=True)

        solutions = [self._to_numpy(u)]
        times = [0.0]

        progress_every = max(1, steps // 100)

        for step in range(1, steps + 1):

            u = self.step(u, model, velocity_x, velocity_y)

            if step % save_every == 0:
                solutions.append(self._to_numpy(u))
                times.append(step * self.dt)

            if progress and (step % progress_every == 0 or step == steps):
                pct = 100.0 * step / steps
                print(f"\r  step {step}/{steps} ({pct:5.1f}%)", end="", flush=True)

        if progress:
            print()

        return (
            np.array(times),
            np.array(solutions)
        )


def stable_dt(
    D_u: float,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    dx: float,
    dy: float,
    xp=np,
) -> float:
    """
    Explicit-Euler CFL limit for the combined diffusion + upwind-advection
    scheme used by KellerSegel2D:

        dt <= 1 / (2*D_u*(1/dx^2 + 1/dy^2) + max|vx|/dx + max|vy|/dy)

    Callers should apply a safety factor (e.g. 0.4-0.5) to the result.
    `velocity_x`/`velocity_y` may be numpy or cupy arrays -- pass the
    matching `xp` module; the result is always a plain Python float.
    """

    diffusion_rate = 2.0 * D_u * (1.0 / dx ** 2 + 1.0 / dy ** 2)
    advection_rate = (
        float(xp.max(xp.abs(velocity_x))) / dx
        + float(xp.max(xp.abs(velocity_y))) / dy
    )

    return 1.0 / (diffusion_rate + advection_rate)
