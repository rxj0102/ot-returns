"""Placeholder tests for otreturns.barycenters — implemented in Prompt 3."""

import pytest
import numpy as np
from otreturns.barycenters import wasserstein_barycenter_1d
from otreturns.distributions import EmpiricalDistribution


def test_barycenter_not_implemented():
    rng = np.random.default_rng(0)
    dists = [EmpiricalDistribution(rng.normal(i, 1, 100)) for i in range(3)]
    with pytest.raises(NotImplementedError):
        wasserstein_barycenter_1d(dists)
