"""Placeholder tests for otreturns.distances — implemented in Prompt 2."""

import pytest
from otreturns.distances import (
    wasserstein_1d,
    wasserstein_sliced,
    sinkhorn_distance,
    wasserstein_distance_matrix,
)
from otreturns.distributions import EmpiricalDistribution
import numpy as np


@pytest.fixture
def two_dists():
    rng = np.random.default_rng(0)
    mu = EmpiricalDistribution(rng.normal(0, 1, 200))
    nu = EmpiricalDistribution(rng.normal(1, 1, 200))
    return mu, nu


def test_wasserstein_1d_not_implemented(two_dists):
    mu, nu = two_dists
    with pytest.raises(NotImplementedError):
        wasserstein_1d(mu, nu)


def test_sinkhorn_distance_not_implemented(two_dists):
    mu, nu = two_dists
    with pytest.raises(NotImplementedError):
        sinkhorn_distance(mu, nu)


def test_distance_matrix_not_implemented(two_dists):
    mu, nu = two_dists
    with pytest.raises(NotImplementedError):
        wasserstein_distance_matrix([mu, nu])
