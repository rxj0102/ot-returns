"""Placeholder tests for otreturns.transport — implemented in Prompt 2."""

import pytest
import numpy as np
from otreturns.transport import transport_map_1d, transport_plan
from otreturns.distributions import EmpiricalDistribution


@pytest.fixture
def two_dists():
    rng = np.random.default_rng(1)
    mu = EmpiricalDistribution(rng.normal(0, 1, 100))
    nu = EmpiricalDistribution(rng.normal(1, 1, 100))
    return mu, nu


def test_transport_map_1d_not_implemented(two_dists):
    mu, nu = two_dists
    with pytest.raises(NotImplementedError):
        transport_map_1d(mu, nu)


def test_transport_plan_not_implemented(two_dists):
    mu, nu = two_dists
    with pytest.raises(NotImplementedError):
        transport_plan(mu, nu)
