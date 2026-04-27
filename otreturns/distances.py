"""
Wasserstein and Sinkhorn distances between empirical distributions.

For one-dimensional distributions the p-Wasserstein distance admits
the closed-form quantile representation:

    W_p(mu, nu)^p = integral_0^1 |F_mu^{-1}(u) - F_nu^{-1}(u)|^p du

making exact computation O(n log n).

For multi-dimensional distributions we use the POT library (Flamary et al.)
which implements the exact EMD (Earth Mover's Distance) via network simplex
and the regularised Sinkhorn-Knopp algorithm.
"""

from __future__ import annotations

import numpy as np
import ot

from otreturns.distributions import EmpiricalDistribution


def wasserstein_1d(mu: EmpiricalDistribution, nu: EmpiricalDistribution,
                   p: int = 2, n_quantiles: int = 1000) -> float:
    """
    Exact W_p distance between two 1-d empirical distributions via
    the quantile (rearrangement) formula.

    Args:
        mu: source distribution
        nu: target distribution
        p: order (1 or 2 recommended)
        n_quantiles: number of quadrature points in [0, 1]

    Returns:
        W_p(mu, nu)
    """
    raise NotImplementedError("Implemented in Prompt 2")


def wasserstein_sliced(mu: EmpiricalDistribution, nu: EmpiricalDistribution,
                       n_projections: int = 200, p: int = 2,
                       seed: int = None) -> float:
    """
    Sliced Wasserstein distance (Monte Carlo over random projections).
    Scales to multi-dimensional distributions.
    """
    raise NotImplementedError("Implemented in Prompt 2")


def sinkhorn_distance(mu: EmpiricalDistribution, nu: EmpiricalDistribution,
                      reg: float = 0.01, p: int = 2,
                      n_bins: int = 200) -> float:
    """
    Regularised Sinkhorn distance on a common histogram grid.

    Args:
        mu: source distribution
        nu: target distribution
        reg: entropic regularisation parameter epsilon
        p: ground metric exponent (cost = |x-y|^p)
        n_bins: number of histogram bins

    Returns:
        S_epsilon(mu, nu) — regularised OT cost
    """
    raise NotImplementedError("Implemented in Prompt 2")


def wasserstein_distance_matrix(distributions: list,
                                p: int = 2,
                                n_quantiles: int = 1000) -> np.ndarray:
    """
    Compute the full pairwise W_p distance matrix for a list of distributions.

    Returns:
        D: symmetric (T x T) distance matrix
    """
    raise NotImplementedError("Implemented in Prompt 2")
