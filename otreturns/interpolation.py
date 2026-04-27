"""
Displacement interpolation (McCann interpolation) between distributions.

Given two measures mu_0 and mu_1 with optimal transport map T,
the McCann interpolant at time t in [0, 1] is:
    mu_t = ((1-t) Id + t T)_# mu_0

In 1-d this is simply:
    F_{mu_t}^{-1}(u) = (1-t) F_{mu_0}^{-1}(u) + t F_{mu_1}^{-1}(u)

This traces the geodesic in Wasserstein-2 space between the two distributions.
"""

from __future__ import annotations

import numpy as np
from otreturns.distributions import EmpiricalDistribution


def mccann_interpolation(mu0: EmpiricalDistribution,
                         mu1: EmpiricalDistribution,
                         t: float,
                         n_quantiles: int = 1000) -> EmpiricalDistribution:
    """
    McCann displacement interpolant at time t in [0, 1].

    mu_t = ((1-t)*Id + t*T)_# mu_0

    Args:
        mu0: initial distribution (t=0)
        mu1: target distribution (t=1)
        t: interpolation parameter in [0, 1]
        n_quantiles: quantile grid resolution

    Returns:
        EmpiricalDistribution at interpolation time t
    """
    raise NotImplementedError("Implemented in Prompt 4")


def interpolation_path(mu0: EmpiricalDistribution,
                       mu1: EmpiricalDistribution,
                       n_steps: int = 20,
                       n_quantiles: int = 1000) -> list:
    """
    Compute the full geodesic path from mu_0 to mu_1.

    Returns:
        list of EmpiricalDistribution objects at t = 0, 1/(n-1), ..., 1
    """
    raise NotImplementedError("Implemented in Prompt 4")


def regime_transition_path(barycenter_start: EmpiricalDistribution,
                            barycenter_end: EmpiricalDistribution,
                            n_steps: int = 20) -> list:
    """
    Geodesic from one regime barycenter to another.
    Visualises how the market distribution transforms between regimes.
    """
    raise NotImplementedError("Implemented in Prompt 4")
