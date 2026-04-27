"""
Wasserstein barycenters of empirical distributions.

The Wasserstein-2 barycenter of {mu_1, ..., mu_K} with weights {lambda_k} is:
    mu* = argmin_nu sum_k lambda_k W_2^2(nu, mu_k)

In 1-d this reduces to averaging quantile functions (Agueh & Carlier 2011):
    F_{mu*}^{-1}(u) = sum_k lambda_k F_{mu_k}^{-1}(u)

making exact computation O(K * n_support).

For d >= 1 distributions we use the iterative Bregman projection algorithm
(Benamou, Carlier, Cuturi, Nenna & Peyré 2015) on a fixed support grid.
"""

from __future__ import annotations

import numpy as np
import ot

from otreturns.distributions import EmpiricalDistribution, DistributionPanel


# ---------------------------------------------------------------------------
# 1-d exact barycenter (Agueh-Carlier closed form)
# ---------------------------------------------------------------------------

def wasserstein_barycenter_1d(
    distributions: list,
    weights: np.ndarray = None,
    n_support: int = 500,
) -> EmpiricalDistribution:
    """
    Wasserstein-2 barycenter of 1-d distributions via quantile averaging.

    Closed-form result (Agueh & Carlier 2011):
        F_{mu*}^{-1}(u) = sum_k w_k * F_{mu_k}^{-1}(u)

    The barycenter quantile function is the weighted average of the input
    quantile functions — a remarkable consequence of the 1-d OT structure.

    Notably, this averages standard deviations (not variances):
        W_2-barycenter of N(0, s1^2) and N(0, s2^2) has std = (s1 + s2) / 2

    Args:
        distributions: list of EmpiricalDistribution objects (length K >= 1)
        weights:       barycentric weights summing to 1 (default: uniform 1/K)
        n_support:     number of support points for the returned distribution

    Returns:
        EmpiricalDistribution representing the barycenter
    """
    K = len(distributions)
    if K == 0:
        raise ValueError("distributions list must be non-empty")

    weights = _check_barycenter_weights(weights, K)

    # Midpoint quantile grid — matches EmpiricalDistribution's internal convention
    # so that quantile_function(u[i]) == sorted_samples[i] for same-size dists.
    u = (np.arange(n_support) + 0.5) / n_support

    # Stack quantile functions: shape (K, n_support)
    Q = np.stack([d.quantile_function(u) for d in distributions], axis=0)

    # Weighted average of quantile functions
    Q_bar = weights @ Q  # (n_support,)

    return EmpiricalDistribution(Q_bar)


# ---------------------------------------------------------------------------
# Sinkhorn barycenter (fixed-support, iterative Bregman projections)
# ---------------------------------------------------------------------------

def wasserstein_barycenter_sinkhorn(
    distributions: list,
    weights: np.ndarray = None,
    reg: float = 0.01,
    n_support: int = 200,
    support_range: tuple = None,
    max_iter_outer: int = 50,
    max_iter_sinkhorn: int = 50,
    tol: float = 1e-6,
) -> EmpiricalDistribution:
    """
    Wasserstein-2 barycenter via Sinkhorn iterations (POT backend).

    Procedure:
        1. Build a shared fixed support grid z_1, ..., z_M.
        2. Project each input distribution onto the grid as a histogram.
        3. Compute the barycenter histogram via ot.bregman.barycenter
           (iterative Bregman projections, Benamou et al. 2015).
        4. Return as EmpiricalDistribution(z, barycenter_weights).

    This works for any d >= 1. For d=1 the exact wasserstein_barycenter_1d
    is computationally cheaper and has no discretisation error.

    Args:
        distributions:      list of EmpiricalDistribution objects
        weights:            barycentric weights (default: uniform)
        reg:                Sinkhorn regularisation epsilon > 0.
                            Auto-scaled to the median squared grid spacing
                            if the raw value seems too small/large.
        n_support:          number of support points in the barycenter grid
        support_range:      (lo, hi) for the support grid (default: data range)
        max_iter_outer:     maximum Sinkhorn iterations (passed to POT)
        max_iter_sinkhorn:  unused (kept for API compatibility)
        tol:                convergence tolerance (passed to POT)

    Returns:
        EmpiricalDistribution on the fixed support grid
    """
    K = len(distributions)
    if K == 0:
        raise ValueError("distributions list must be non-empty")
    if reg <= 0:
        raise ValueError(f"reg must be positive, got {reg}")

    weights = _check_barycenter_weights(weights, K)

    # Build fixed support grid
    if support_range is None:
        lo = float(min(d.samples.min() for d in distributions))
        hi = float(max(d.samples.max() for d in distributions))
        margin = max(0.1 * (hi - lo), 1e-6)
        lo, hi = lo - margin, hi + margin
    else:
        lo, hi = float(support_range[0]), float(support_range[1])

    z = np.linspace(lo, hi, n_support)  # (M,)

    # Project each distribution onto the shared grid as a histogram.
    # Shape: (n_support, K) — each column is one input histogram.
    A = np.zeros((n_support, K))
    for k, d in enumerate(distributions):
        _, w = d.to_histogram(n_bins=n_support, range=(lo, hi))
        # Ensure strictly positive (Sinkhorn needs non-zero entries)
        w = w + 1e-10
        A[:, k] = w / w.sum()

    # Ground cost matrix on the shared grid
    M_cost = (z[:, None] - z[None, :]) ** 2   # (M, M)

    # POT's Sinkhorn barycenter
    b = ot.bregman.barycenter(
        A, M_cost, reg,
        weights=weights,
        numItermax=max_iter_outer,
        stopThr=tol,
        warn=False,
    )
    b = np.asarray(b, dtype=float)
    b = b / b.sum()

    return EmpiricalDistribution(z, b)


# ---------------------------------------------------------------------------
# Regime barycenters
# ---------------------------------------------------------------------------

def regime_barycenters(
    panel: DistributionPanel,
    regime_labels: np.ndarray,
    method: str = "1d",
    **kwargs,
) -> dict:
    """
    Compute the Wasserstein barycenter for each regime.

    Args:
        panel:          DistributionPanel
        regime_labels:  integer array of length len(panel.dates) assigning
                        each date to a regime in {0, 1, ..., K-1}
        method:         '1d' (exact) or 'sinkhorn' (iterative)
        **kwargs:       forwarded to the chosen barycenter function

    Returns:
        dict mapping regime_label (int) -> EmpiricalDistribution (barycenter)
    """
    regime_labels = np.asarray(regime_labels, dtype=int)
    dates = panel.dates

    if len(regime_labels) != len(dates):
        raise ValueError(
            f"regime_labels length {len(regime_labels)} != "
            f"panel date count {len(dates)}"
        )

    unique_labels = np.unique(regime_labels)
    result: dict = {}

    for label in unique_labels:
        mask = regime_labels == label
        dists = [panel[d] for d, m in zip(dates, mask) if m]

        if not dists:
            continue

        if method == "1d":
            bary = wasserstein_barycenter_1d(dists, **kwargs)
        elif method == "sinkhorn":
            bary = wasserstein_barycenter_sinkhorn(dists, **kwargs)
        else:
            raise ValueError(f"Unknown method '{method}'. Use '1d' or 'sinkhorn'.")

        result[int(label)] = bary

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_barycenter_weights(weights, K: int) -> np.ndarray:
    if weights is None:
        return np.ones(K) / K
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (K,):
        raise ValueError(f"weights must have shape ({K},), got {weights.shape}")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative")
    s = weights.sum()
    if s <= 0:
        raise ValueError("weights must sum to a positive value")
    return weights / s
