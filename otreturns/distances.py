"""
Wasserstein and Sinkhorn distances between empirical distributions.

For one-dimensional distributions the p-Wasserstein distance admits
the closed-form quantile representation:

    W_p(mu, nu)^p = integral_0^1 |F_mu^{-1}(u) - F_nu^{-1}(u)|^p du

making exact computation O(n log n).

For multi-dimensional distributions we use the POT library (Flamary et al.)
which implements the exact EMD via network simplex and the regularised
Sinkhorn-Knopp algorithm.
"""

from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
import ot

from otreturns.distributions import EmpiricalDistribution, DistributionPanel


# ---------------------------------------------------------------------------
# 1-d exact Wasserstein
# ---------------------------------------------------------------------------

def wasserstein_1d(
    mu: EmpiricalDistribution,
    nu: EmpiricalDistribution,
    p: int = 2,
) -> float:
    """
    p-Wasserstein distance between two 1-d empirical distributions.

    Uses the closed-form quantile coupling:
        W_p^p = integral_0^1 |F_mu^{-1}(u) - F_nu^{-1}(u)|^p du

    For uniform weights of equal size n this reduces to:
        W_p^p = (1/n) * sum_i |x_(i) - y_(i)|^p   (sorted order statistics)

    For unequal sizes or non-uniform weights we merge all support points
    onto a joint sorted grid and integrate the piecewise-constant
    quantile difference exactly — no approximation error.

    Complexity: O((n+m) log(n+m)).

    Args:
        mu: source distribution
        nu: target distribution
        p:  Wasserstein order (any positive integer)

    Returns:
        W_p(mu, nu)  (not raised to the p-th power)
    """
    # Fast path: equal-size uniform weights — sort and pair directly.
    if mu.n == nu.n and np.allclose(mu.weights, 1.0 / mu.n):
        diff = np.sort(mu.samples) - np.sort(nu.samples)
        return float(np.mean(np.abs(diff) ** p) ** (1.0 / p))

    # General path: merge support points onto a joint CDF grid.
    # Build the combined sorted support and integrate |Q_mu - Q_nu|^p
    # over each sub-interval using the quantile functions.
    all_points = np.concatenate([mu._sorted_samples, nu._sorted_samples])
    all_points = np.unique(all_points)

    # CDF values just to the right of each point give the quantile level
    # at the right endpoint of each interval.
    cdf_mu = mu.cdf(all_points)   # shape (k,)
    cdf_nu = nu.cdf(all_points)   # shape (k,)

    # Build a fine uniform quantile grid covering (0, 1) — include both
    # CDF break-points and a dense uniform grid to capture all intervals.
    n_grid = max(1000, 4 * (mu.n + nu.n))
    u_uniform = np.linspace(0.0, 1.0, n_grid + 2)[1:-1]  # (0,1) exclusive
    u_grid = np.union1d(u_uniform, np.concatenate([cdf_mu, cdf_nu]))
    u_grid = np.clip(u_grid, 0.0, 1.0)

    q_mu = mu.quantile_function(u_grid)
    q_nu = nu.quantile_function(u_grid)

    # Trapezoidal integration of |q_mu - q_nu|^p over u in [0,1]
    integrand = np.abs(q_mu - q_nu) ** p
    cost = float(np.trapezoid(integrand, u_grid))
    return float(cost ** (1.0 / p))


# ---------------------------------------------------------------------------
# n-d exact Wasserstein (EMD via POT)
# ---------------------------------------------------------------------------

def wasserstein_nd(
    mu_samples: np.ndarray,
    nu_samples: np.ndarray,
    mu_weights: np.ndarray = None,
    nu_weights: np.ndarray = None,
    p: int = 2,
    method: str = "emd",
    reg: float = 0.01,
    n_projections: int = 100,
    seed: int = None,
) -> float:
    """
    p-Wasserstein distance between d-dimensional empirical distributions.

    Solves the Kantorovich problem:
        W_p^p = inf_{gamma in Pi(mu,nu)} integral ||x-y||^p d_gamma(x,y)

    Methods:
        'emd'      — exact Earth Mover's Distance via network simplex (POT).
                     O(n^3 log n). Exact but slow for n > 5000.
        'sinkhorn' — entropy-regularised OT (Sinkhorn-Knopp).
                     O(n^2 / eps). Approximate.
        'sliced'   — Sliced Wasserstein distance over random projections.
                     O(L * n log n). Approximate.

    Args:
        mu_samples: (n, d) or (n,) array
        nu_samples: (m, d) or (m,) array
        mu_weights: length-n weight vector (default uniform)
        nu_weights: length-m weight vector (default uniform)
        p:          Wasserstein order
        method:     'emd' | 'sinkhorn' | 'sliced'
        reg:        Sinkhorn regularisation (used when method='sinkhorn')
        n_projections: projections (used when method='sliced')
        seed:       RNG seed for sliced method

    Returns:
        W_p(mu, nu)
    """
    mu_samples = np.asarray(mu_samples, dtype=float)
    nu_samples = np.asarray(nu_samples, dtype=float)
    if mu_samples.ndim == 1:
        mu_samples = mu_samples[:, None]
    if nu_samples.ndim == 1:
        nu_samples = nu_samples[:, None]

    n, d = mu_samples.shape
    m = nu_samples.shape[0]

    a = _uniform_weights(n) if mu_weights is None else _check_weights(mu_weights, n)
    b = _uniform_weights(m) if nu_weights is None else _check_weights(nu_weights, m)

    if method == "emd":
        M = ot.dist(mu_samples, nu_samples, metric="sqeuclidean")
        if p != 2:
            M = np.abs(M) ** (p / 2.0)
        cost = float(ot.emd2(a, b, M))
        return float(cost ** (1.0 / p))

    if method == "sinkhorn":
        result = sinkhorn_distance(
            mu_samples.squeeze() if d == 1 else mu_samples,
            nu_samples.squeeze() if d == 1 else nu_samples,
            mu_weights=a,
            nu_weights=b,
            reg=reg,
            p=p,
        )
        return result["distance"]

    if method == "sliced":
        return sliced_wasserstein(
            mu_samples, nu_samples,
            p=p, n_projections=n_projections, seed=seed,
        )

    raise ValueError(f"Unknown method '{method}'. Choose 'emd', 'sinkhorn', or 'sliced'.")


# ---------------------------------------------------------------------------
# Sinkhorn-Knopp (log-domain for numerical stability)
# ---------------------------------------------------------------------------

def sinkhorn_distance(
    mu_samples: np.ndarray,
    nu_samples: np.ndarray,
    mu_weights: np.ndarray = None,
    nu_weights: np.ndarray = None,
    reg: float = 0.01,
    p: int = 2,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> dict:
    """
    Sinkhorn-regularised optimal transport distance (log-domain).

    Solves:
        W^eps_p = inf_{gamma in Pi(mu,nu)} <C, gamma> + eps * KL(gamma || mu x nu)

    Algorithm (log-domain Sinkhorn):
        log_f = log_a - logsumexp(log_K + log_g, axis=1)
        log_g = log_b - logsumexp(log_K^T + log_f, axis=1)
    where log_K_{ij} = -C_{ij} / eps.

    Args:
        mu_samples: (n,) or (n, d) source samples
        nu_samples: (m,) or (m, d) target samples
        mu_weights: length-n weights (default uniform)
        nu_weights: length-m weights (default uniform)
        reg:        regularisation epsilon > 0
        p:          ground cost exponent (C = ||x-y||^p)
        max_iter:   maximum Sinkhorn iterations
        tol:        convergence tolerance on marginal error

    Returns:
        dict with keys:
            'distance'       — Sinkhorn distance W^eps
            'transport_plan' — (n, m) coupling matrix gamma
            'n_iterations'   — iterations to convergence
            'dual_potentials'— (log_f, log_g)
            'marginal_error' — max marginal constraint violation at convergence
    """
    mu_samples = np.asarray(mu_samples, dtype=float)
    nu_samples = np.asarray(nu_samples, dtype=float)

    if mu_samples.ndim == 1:
        mu_samples = mu_samples[:, None]
    if nu_samples.ndim == 1:
        nu_samples = nu_samples[:, None]

    n = mu_samples.shape[0]
    m = nu_samples.shape[0]

    a = _uniform_weights(n) if mu_weights is None else _check_weights(mu_weights, n)
    b = _uniform_weights(m) if nu_weights is None else _check_weights(nu_weights, m)

    # Cost matrix
    M = ot.dist(mu_samples, nu_samples, metric="sqeuclidean")
    if p != 2:
        M = np.abs(M) ** (p / 2.0)

    # Auto-scale reg relative to median cost
    if reg <= 0:
        raise ValueError("reg must be positive")

    log_K = -M / reg          # (n, m)
    log_a = np.log(a + 1e-300)
    log_b = np.log(b + 1e-300)

    log_f = np.zeros(n)       # dual potential f (source)
    log_g = np.zeros(m)       # dual potential g (target)

    marginal_error = float("inf")
    it = 0
    for it in range(1, max_iter + 1):
        # f update: log_f = log_a - logsumexp(log_K + log_g, axis=1)
        log_f = log_a - logsumexp(log_K + log_g[None, :], axis=1)
        # g update: log_g = log_b - logsumexp(log_K^T + log_f, axis=1)
        log_g = log_b - logsumexp(log_K.T + log_f[None, :], axis=1)

        # Check marginal error every 20 iterations
        if it % 20 == 0 or it == max_iter:
            # gamma = diag(f) K diag(g)
            log_gamma = log_f[:, None] + log_K + log_g[None, :]
            gamma = np.exp(log_gamma)
            err_a = np.max(np.abs(gamma.sum(axis=1) - a))
            err_b = np.max(np.abs(gamma.sum(axis=0) - b))
            marginal_error = float(max(err_a, err_b))
            if marginal_error < tol:
                break

    # Final transport plan
    log_gamma = log_f[:, None] + log_K + log_g[None, :]
    gamma = np.exp(log_gamma)
    distance = float(np.sum(M * gamma)) ** (1.0 / p)

    return {
        "distance": distance,
        "transport_plan": gamma,
        "n_iterations": it,
        "dual_potentials": (log_f, log_g),
        "marginal_error": marginal_error,
    }


# ---------------------------------------------------------------------------
# Sliced Wasserstein
# ---------------------------------------------------------------------------

def sliced_wasserstein(
    mu_samples: np.ndarray,
    nu_samples: np.ndarray,
    p: int = 2,
    n_projections: int = 100,
    seed: int = None,
) -> float:
    """
    Sliced Wasserstein distance.

    SW_p^p = (1/L) * sum_{l=1}^L W_p(theta_l # mu, theta_l # nu)^p

    where theta_l ~ Uniform(S^{d-1}) and theta # mu is the 1-d pushforward.

    Each 1-d Wasserstein uses the exact sorted-order-statistics formula,
    so the total complexity is O(L * (n+m) * log(n+m)).

    Args:
        mu_samples:    (n,) or (n, d) array
        nu_samples:    (m,) or (m, d) array
        p:             Wasserstein order
        n_projections: number of random projection directions L
        seed:          RNG seed

    Returns:
        SW_p(mu, nu)
    """
    mu_samples = np.asarray(mu_samples, dtype=float)
    nu_samples = np.asarray(nu_samples, dtype=float)

    if mu_samples.ndim == 1:
        mu_samples = mu_samples[:, None]
    if nu_samples.ndim == 1:
        nu_samples = nu_samples[:, None]

    d = mu_samples.shape[1]
    n = mu_samples.shape[0]
    m = nu_samples.shape[0]

    rng = np.random.default_rng(seed)

    # For 1-d, a single direction suffices (sign doesn't matter for W_p)
    if d == 1:
        # Use fast equal-n path when possible
        mu_proj = mu_samples[:, 0]
        nu_proj = nu_samples[:, 0]
        mu_d = EmpiricalDistribution(mu_proj)
        nu_d = EmpiricalDistribution(nu_proj)
        return wasserstein_1d(mu_d, nu_d, p=p)

    # Sample random unit vectors on S^{d-1}
    directions = rng.standard_normal((n_projections, d))
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / norms  # (L, d)

    # Project both samples onto each direction: shape (L, n) and (L, m)
    proj_mu = mu_samples @ directions.T   # (n, L) -> transpose to (L, n)
    proj_nu = nu_samples @ directions.T   # (m, L)
    proj_mu = proj_mu.T  # (L, n)
    proj_nu = proj_nu.T  # (L, m)

    # Sort projections for each direction
    proj_mu_sorted = np.sort(proj_mu, axis=1)  # (L, n)
    proj_nu_sorted = np.sort(proj_nu, axis=1)  # (L, m)

    # For equal sample sizes, pair directly
    if n == m:
        costs = np.mean(np.abs(proj_mu_sorted - proj_nu_sorted) ** p, axis=1)  # (L,)
    else:
        # Interpolate quantiles to a common grid
        n_q = max(n, m)
        u = np.linspace(0.0, 1.0, n_q)
        q_mu = np.array([np.interp(u, np.linspace(0, 1, n), proj_mu_sorted[l])
                          for l in range(n_projections)])
        q_nu = np.array([np.interp(u, np.linspace(0, 1, m), proj_nu_sorted[l])
                          for l in range(n_projections)])
        costs = np.mean(np.abs(q_mu - q_nu) ** p, axis=1)  # (L,)

    sw_p = float(np.mean(costs))
    return float(sw_p ** (1.0 / p))


# ---------------------------------------------------------------------------
# Pairwise distance matrix
# ---------------------------------------------------------------------------

def wasserstein_distance_matrix(
    panel: "DistributionPanel | list",
    dates: list = None,
    p: int = 2,
    method: str = "1d",
) -> np.ndarray:
    """
    Pairwise W_p distance matrix between cross-sectional distributions.

    D_{st} = W_p(mu_s, mu_t)

    This matrix is the foundation for regime detection: dates with small
    D_{st} have similar cross-sectional structure; clusters correspond to
    market regimes.

    Args:
        panel:  DistributionPanel or list of EmpiricalDistribution
        dates:  subset of dates to include (default: all)
        p:      Wasserstein order
        method: '1d' (exact quantile), 'sinkhorn', or 'sliced'

    Returns:
        D: symmetric (T, T) float array with zero diagonal
    """
    if isinstance(panel, list):
        dists = panel
    else:
        dates = dates if dates is not None else panel.dates
        dists = [panel[d] for d in dates]

    T = len(dists)
    D = np.zeros((T, T))

    for i in range(T):
        for j in range(i + 1, T):
            if method == "1d":
                d = wasserstein_1d(dists[i], dists[j], p=p)
            elif method == "sinkhorn":
                d = sinkhorn_distance(
                    dists[i].samples, dists[j].samples,
                    mu_weights=dists[i].weights,
                    nu_weights=dists[j].weights,
                    p=p,
                )["distance"]
            elif method == "sliced":
                d = sliced_wasserstein(
                    dists[i].samples, dists[j].samples, p=p
                )
            else:
                raise ValueError(f"Unknown method '{method}'")
            D[i, j] = d
            D[j, i] = d

    return D


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_weights(n: int) -> np.ndarray:
    return np.full(n, 1.0 / n)


def _check_weights(w: np.ndarray, n: int) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"weights must have shape ({n},), got {w.shape}")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    s = w.sum()
    if s <= 0:
        raise ValueError("weights must sum to a positive value")
    return w / s
