"""
Two-sample distributional tests based on Wasserstein distances.

The null hypothesis H_0: mu = nu is tested via a permutation test on
the Wasserstein distance W_p(mu_hat, nu_hat).

Under H_0, pooling and randomly splitting the combined sample yields
a null distribution for W_p. The p-value is:
    p = #{permutations with W_p >= W_p_observed} / n_permutations

Bootstrap confidence intervals for W_p follow the same logic with
resampling within each group.
"""

from __future__ import annotations

import numpy as np
from otreturns.distributions import EmpiricalDistribution


def wasserstein_two_sample_test(mu: EmpiricalDistribution,
                                 nu: EmpiricalDistribution,
                                 p: int = 2,
                                 n_permutations: int = 999,
                                 seed: int = None) -> dict:
    """
    Permutation test for H_0: mu = nu using W_p as test statistic.

    Args:
        mu: first sample distribution
        nu: second sample distribution
        p: Wasserstein order
        n_permutations: number of permutation replicates
        seed: random seed

    Returns:
        dict with keys: statistic, p_value, null_distribution
    """
    raise NotImplementedError("Implemented in Prompt 5")


def bootstrap_wasserstein_ci(mu: EmpiricalDistribution,
                              nu: EmpiricalDistribution,
                              p: int = 2,
                              n_bootstrap: int = 999,
                              confidence: float = 0.95,
                              seed: int = None) -> dict:
    """
    Bootstrap confidence interval for W_p(mu, nu).

    Returns:
        dict with keys: estimate, ci_lower, ci_upper, bootstrap_distribution
    """
    raise NotImplementedError("Implemented in Prompt 5")


def energy_distance_test(mu: EmpiricalDistribution,
                         nu: EmpiricalDistribution,
                         n_permutations: int = 999,
                         seed: int = None) -> dict:
    """
    Two-sample test using the energy distance (related to W_1).
    Computationally cheaper than exact W_2 for large samples.
    """
    raise NotImplementedError("Implemented in Prompt 5")
