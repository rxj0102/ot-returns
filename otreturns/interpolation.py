"""
Displacement interpolation (McCann interpolation) between distributions.

Given two measures mu_0 and mu_1, the McCann interpolant at t in [0,1]:
    mu_t = ((1-t)*Id + t*T)_# mu_0

where T = F_nu^{-1} o F_mu is the optimal transport map.

In 1-d this simplifies to:
    F_{mu_t}^{-1}(u) = (1-t)*F_{mu_0}^{-1}(u) + t*F_{mu_1}^{-1}(u)

This traces the W_2-geodesic (shortest path) between mu_0 and mu_1, and
the path has constant speed:
    W_2(mu_0, mu_t) = t * W_2(mu_0, mu_1)
"""

from __future__ import annotations

import numpy as np

from otreturns.distributions import EmpiricalDistribution


# ---------------------------------------------------------------------------
# McCann interpolation (single step)
# ---------------------------------------------------------------------------

def mccann_interpolation(
    source: EmpiricalDistribution,
    target: EmpiricalDistribution,
    t: float,
    n_support: int = None,
) -> EmpiricalDistribution:
    """
    McCann (1997) displacement interpolant at time t in [0, 1].

    Geodesic in (P_2, W_2) space:
        F_{mu_t}^{-1}(u) = (1-t)*F_{source}^{-1}(u) + t*F_{target}^{-1}(u)

    t=0: source distribution (returned as-is, same object)
    t=1: target distribution (returned as-is, same object)
    t=0.5: Wasserstein midpoint

    The path has constant speed: W_2(source, mu_t) = t * W_2(source, target).

    Args:
        source:    source distribution (t=0 endpoint)
        target:    target distribution (t=1 endpoint)
        t:         interpolation parameter in [0, 1]
        n_support: number of support points (default: max(source.n, target.n))

    Returns:
        Interpolated EmpiricalDistribution
    """
    t = float(t)
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"t must be in [0, 1], got {t}")
    if t == 0.0:
        return source
    if t == 1.0:
        return target

    n = n_support if n_support is not None else max(source.n, target.n)

    # Midpoint quantile grid: u[i] = (i + 0.5) / n
    # This convention ensures quantile_function(u[i]) == sorted_samples[i]
    # for a distribution built with the same n, giving exact constant-speed.
    u = (np.arange(n) + 0.5) / n

    Q_s = source.quantile_function(u)
    Q_t = target.quantile_function(u)

    Q_interp = (1.0 - t) * Q_s + t * Q_t

    return EmpiricalDistribution(Q_interp)


# ---------------------------------------------------------------------------
# Full interpolation path
# ---------------------------------------------------------------------------

def interpolation_path(
    source: EmpiricalDistribution,
    target: EmpiricalDistribution,
    n_steps: int = 10,
    n_support: int = None,
) -> list:
    """
    Compute the full W_2-geodesic path from source to target.

    Args:
        source:    source distribution
        target:    target distribution
        n_steps:   number of interpolation steps (path has n_steps+1 elements)
        n_support: support size for intermediate distributions

    Returns:
        List of n_steps+1 EmpiricalDistribution objects at
        t = 0, 1/n_steps, 2/n_steps, ..., 1.
        Endpoints are the original source and target objects.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    n = n_support if n_support is not None else max(source.n, target.n)
    u = (np.arange(n) + 0.5) / n

    # Evaluate quantile functions once, reuse for all steps
    Q_s = source.quantile_function(u)
    Q_t = target.quantile_function(u)

    path = []
    for k in range(n_steps + 1):
        t = k / n_steps
        if t == 0.0:
            path.append(source)
        elif t == 1.0:
            path.append(target)
        else:
            Q_interp = (1.0 - t) * Q_s + t * Q_t
            path.append(EmpiricalDistribution(Q_interp))

    return path


# ---------------------------------------------------------------------------
# Multi-marginal piecewise geodesic interpolation
# ---------------------------------------------------------------------------

def multi_marginal_interpolation(
    distributions: list,
    times: np.ndarray,
    query_time: float,
    n_support: int = None,
) -> EmpiricalDistribution:
    """
    Piecewise geodesic interpolation over multiple distributions.

    Given K distributions at observation times t_1 < t_2 < ... < t_K,
    interpolate at an arbitrary query_time using the McCann geodesic on the
    enclosing interval [t_k, t_{k+1}]:

        s = (query_time - t_k) / (t_{k+1} - t_k)
        result = mccann_interpolation(mu_k, mu_{k+1}, s)

    Queries at or before t_1 return the first distribution; queries at or
    after t_K return the last distribution.

    Args:
        distributions: list of K EmpiricalDistribution objects
        times:         length-K array of observation times (need not be sorted)
        query_time:    the time at which to evaluate the interpolant
        n_support:     passed to mccann_interpolation (default: max sample count)

    Returns:
        Interpolated EmpiricalDistribution at query_time
    """
    times = np.asarray(times, dtype=float)
    K = len(distributions)

    if K != len(times):
        raise ValueError(
            f"distributions ({K}) and times ({len(times)}) must have the same length"
        )
    if K < 2:
        raise ValueError("need at least 2 distributions for interpolation")

    # Sort by time
    order = np.argsort(times)
    t_sorted = times[order]
    d_sorted = [distributions[int(i)] for i in order]

    query_time = float(query_time)

    if query_time <= t_sorted[0]:
        return d_sorted[0]
    if query_time >= t_sorted[-1]:
        return d_sorted[-1]

    # Find enclosing interval [t_k, t_{k+1}]
    k = int(np.searchsorted(t_sorted, query_time, side="right")) - 1
    k = min(k, K - 2)   # guard against floating-point edge case

    t0, t1 = float(t_sorted[k]), float(t_sorted[k + 1])
    s = (query_time - t0) / (t1 - t0)

    return mccann_interpolation(d_sorted[k], d_sorted[k + 1], float(s), n_support)
