"""
Wasserstein barycenters of empirical distributions.

The Wasserstein barycenter of {mu_1, ..., mu_T} with weights {lambda_t} is:
    mu* = argmin_nu sum_t lambda_t W_2^2(nu, mu_t)

In 1-d this reduces to the barycenter of quantile functions:
    F_{mu*}^{-1}(u) = sum_t lambda_t F_{mu_t}^{-1}(u)

making computation O(T * n_quantiles).
"""

from __future__ import annotations

import numpy as np
from otreturns.distributions import EmpiricalDistribution


def wasserstein_barycenter_1d(distributions: list,
                               weights: np.ndarray = None,
                               n_quantiles: int = 1000) -> EmpiricalDistribution:
    """
    Compute the W_2 barycenter of a list of 1-d distributions.

    In 1-d the barycenter quantile function is the weighted average of
    individual quantile functions (Agueh & Carlier 2011).

    Args:
        distributions: list of EmpiricalDistribution objects
        weights: barycenter weights (default: uniform)
        n_quantiles: number of quantile grid points

    Returns:
        EmpiricalDistribution representing the barycenter
    """
    raise NotImplementedError("Implemented in Prompt 3")


def free_support_barycenter(distributions: list,
                             weights: np.ndarray = None,
                             n_support: int = 500,
                             reg: float = 0.01,
                             max_iter: int = 100) -> EmpiricalDistribution:
    """
    Free-support Wasserstein barycenter via alternating minimisation (POT).

    Allows the barycenter to have arbitrary support locations (not fixed to
    a grid), appropriate for multi-dimensional distributions.

    Args:
        distributions: list of EmpiricalDistribution objects
        weights: barycenter weights
        n_support: number of support points in the barycenter
        reg: Sinkhorn regularisation (0 = exact EMD)
        max_iter: maximum alternating-minimisation iterations

    Returns:
        EmpiricalDistribution representing the barycenter
    """
    raise NotImplementedError("Implemented in Prompt 3")


def regime_barycenter(panel, regime_dates: list,
                      weights: np.ndarray = None) -> EmpiricalDistribution:
    """
    Compute the barycenter across all dates belonging to a regime.

    Args:
        panel: DistributionPanel
        regime_dates: list of dates in this regime
        weights: per-date weights (default: uniform)

    Returns:
        EmpiricalDistribution representing the regime's canonical distribution
    """
    raise NotImplementedError("Implemented in Prompt 3")
