"""
Global sensitivity analysis helpers: Morris elementary-effects screening
and Sobol variance-based indices (Saltelli sampling, Saltelli-2010 first-
order / Jansen total-order estimators).

Pure numpy on purpose -- the GPU venv (C:\\gpuvenv) has neither scipy nor
SALib, and the M3 sensitivity script must run there. All sampling is done
in the unit hypercube [0, 1]^k; `scale_to_bounds` maps it onto the real
parameter ranges (callers handle log-scaled parameters by putting log10
bounds in and exponentiating afterwards).
"""

import numpy as np


def scale_to_bounds(unit_samples, bounds):
    """Map samples in [0, 1]^k onto per-column [low, high] bounds."""

    bounds = np.asarray(bounds, dtype=float)
    return bounds[:, 0] + unit_samples * (bounds[:, 1] - bounds[:, 0])


# ============================================================
# Morris screening
# ============================================================

def morris_trajectories(k, n_trajectories, n_levels=4, rng=None):
    """
    Morris (1991) one-at-a-time trajectories in [0, 1]^k.

    Each trajectory has k+1 points; consecutive points differ in exactly
    one factor by +-delta, delta = n_levels / (2*(n_levels-1)). Returns
    shape (n_trajectories*(k+1), k).
    """

    if n_levels % 2 != 0:
        raise ValueError("n_levels should be even so delta lands on the level grid.")

    rng = np.random.default_rng(rng)
    delta = n_levels / (2.0 * (n_levels - 1))

    # Base points are drawn from the levels that leave room for +delta.
    base_levels = np.arange(n_levels // 2) / (n_levels - 1)

    B = np.tril(np.ones((k + 1, k)), -1)
    J = np.ones((k + 1, k))

    trajectories = []
    for _ in range(n_trajectories):
        x_star = rng.choice(base_levels, size=k)
        D_star = np.diag(rng.choice([-1.0, 1.0], size=k))
        P_star = np.eye(k)[rng.permutation(k)]

        B_star = (J * x_star + (delta / 2.0) * ((2.0 * B - J) @ D_star + J)) @ P_star
        trajectories.append(B_star)

    return np.vstack(trajectories)


def morris_indices(unit_samples, y, k):
    """
    Elementary effects from morris_trajectories() samples and outputs.

    Returns dict with mu, mu_star (mean |EE|, the screening measure),
    sigma (std of EE: nonlinearity/interaction), and the raw EE array of
    shape (n_trajectories, k). Trajectories containing a NaN output are
    skipped.
    """

    y = np.asarray(y, dtype=float)
    n_points = k + 1
    n_trajectories = len(y) // n_points

    effects = []
    for t in range(n_trajectories):
        X_t = unit_samples[t * n_points:(t + 1) * n_points]
        y_t = y[t * n_points:(t + 1) * n_points]

        if np.isnan(y_t).any():
            continue

        ee = np.empty(k)
        for step in range(k):
            dx = X_t[step + 1] - X_t[step]
            factor = int(np.argmax(np.abs(dx)))
            ee[factor] = (y_t[step + 1] - y_t[step]) / dx[factor]
        effects.append(ee)

    effects = np.array(effects)

    return {
        "mu": effects.mean(axis=0),
        "mu_star": np.abs(effects).mean(axis=0),
        "sigma": effects.std(axis=0, ddof=1) if len(effects) > 1 else np.zeros(k),
        "effects": effects,
    }


# ============================================================
# Sobol indices
# ============================================================

def latin_hypercube(n, k, rng=None):
    """Plain Latin hypercube sample in [0, 1]^k, shape (n, k)."""

    rng = np.random.default_rng(rng)
    strata = np.array([rng.permutation(n) for _ in range(k)]).T
    return (strata + rng.random((n, k))) / n


def saltelli_samples(n_base, k, rng=None):
    """
    Saltelli design: independent base matrices A, B (n_base x k each,
    Latin hypercube) plus k matrices AB_i = A with column i taken from B.

    Returns shape (n_base*(k+2), k), stacked as [A; B; AB_1; ...; AB_k] --
    the order sobol_indices() expects.
    """

    rng = np.random.default_rng(rng)
    A = latin_hypercube(n_base, k, rng)
    B = latin_hypercube(n_base, k, rng)

    blocks = [A, B]
    for i in range(k):
        AB_i = A.copy()
        AB_i[:, i] = B[:, i]
        blocks.append(AB_i)

    return np.vstack(blocks)


def _sobol_from_blocks(f_A, f_B, f_AB):
    variance = np.var(np.concatenate([f_A, f_B]), ddof=1)

    # Saltelli et al. (2010) first order, Jansen (1999) total order.
    S1 = np.mean(f_B[:, None] * (f_AB - f_A[:, None]), axis=0) / variance
    ST = 0.5 * np.mean((f_A[:, None] - f_AB) ** 2, axis=0) / variance

    return S1, ST


def sobol_indices(y, n_base, k, n_bootstrap=500, confidence=0.95, rng=None):
    """
    First-order (S1) and total-order (ST) Sobol indices from outputs of
    saltelli_samples(), with bootstrap confidence half-widths.

    Rows of the base sample with a NaN in any of their k+2 evaluations
    are dropped before estimating.
    """

    y = np.asarray(y, dtype=float)
    f_A = y[:n_base]
    f_B = y[n_base:2 * n_base]
    f_AB = y[2 * n_base:].reshape(k, n_base).T

    valid = ~(np.isnan(f_A) | np.isnan(f_B) | np.isnan(f_AB).any(axis=1))
    f_A, f_B, f_AB = f_A[valid], f_B[valid], f_AB[valid]
    n_valid = len(f_A)

    # An (almost) constant output has no variance to decompose; the
    # estimators would just amplify round-off. Report NaN instead.
    pooled = np.concatenate([f_A, f_B])
    if n_valid < 2 or np.var(pooled) <= 1e-12 * max(np.mean(pooled) ** 2, 1e-300):
        nan = np.full(k, np.nan)
        return {"S1": nan, "S1_conf": nan, "ST": nan, "ST_conf": nan, "n_valid": n_valid}

    S1, ST = _sobol_from_blocks(f_A, f_B, f_AB)

    rng = np.random.default_rng(rng)
    S1_boot = np.empty((n_bootstrap, k))
    ST_boot = np.empty((n_bootstrap, k))
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_valid, n_valid)
        S1_boot[b], ST_boot[b] = _sobol_from_blocks(f_A[idx], f_B[idx], f_AB[idx])

    # Normal-approximation half-width, as SALib reports it.
    z = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}[confidence]

    return {
        "S1": S1,
        "S1_conf": z * S1_boot.std(axis=0, ddof=1),
        "ST": ST,
        "ST_conf": z * ST_boot.std(axis=0, ddof=1),
        "n_valid": n_valid,
    }
