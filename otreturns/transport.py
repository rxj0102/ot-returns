"""
Optimal transport maps and plans.

The optimal transport plan gamma between mu and nu solves:
    min_{gamma in Pi(mu, nu)} integral |x - y|^2 d_gamma(x, y)

In 1-d the plan is fully determined by the quantile coupling:
    gamma = (Id, T)_# mu,  T = F_nu^{-1} o F_mu

For multi-dimensional distributions we use POT's emd() solver
(network simplex, exact) or regularised sinkhorn for large n.
"""

from __future__ import annotations

import numpy as np
from otreturns.distributions import EmpiricalDistribution


def transport_map_1d(mu: EmpiricalDistribution,
                     nu: EmpiricalDistribution,
                     n_quantiles: int = 1000) -> tuple:
    """
    Compute the optimal transport map T: supp(mu) -> supp(nu) in 1-d.

    T is evaluated on a uniform quantile grid and returned as
    (source_points, target_points) suitable for interpolation.

    Returns:
        (x, Tx): arrays of shape (n_quantiles,)
    """
    raise NotImplementedError("Implemented in Prompt 2")


def transport_plan(mu: EmpiricalDistribution,
                   nu: EmpiricalDistribution) -> np.ndarray:
    """
    Compute the exact discrete optimal transport plan (joint distribution matrix).

    Uses POT's EMD solver (network simplex). Memory is O(n*m) so this is
    best suited for histogram representations with n_bins <= 500.

    Returns:
        gamma: (n, m) coupling matrix with row sums = mu.weights, col sums = nu.weights
    """
    raise NotImplementedError("Implemented in Prompt 2")


def barycentric_projection(mu: EmpiricalDistribution,
                           gamma: np.ndarray,
                           nu_samples: np.ndarray) -> np.ndarray:
    """
    Compute the barycentric projection of the transport plan onto source atoms.

    T_bar(x_i) = sum_j gamma_{ij} y_j / mu_i

    Returns:
        projected: array of shape (n,) giving the barycentric target for each source atom
    """
    raise NotImplementedError("Implemented in Prompt 2")
