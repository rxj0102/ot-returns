"""Placeholder tests for otreturns.testing — implemented in Prompt 5."""

import pytest
import numpy as np
from otreturns.testing import wasserstein_two_sample_test
from otreturns.distributions import EmpiricalDistribution


def test_two_sample_test_not_implemented():
    rng = np.random.default_rng(0)
    mu = EmpiricalDistribution(rng.normal(0, 1, 100))
    nu = EmpiricalDistribution(rng.normal(0, 1, 100))
    with pytest.raises(NotImplementedError):
        wasserstein_two_sample_test(mu, nu)
