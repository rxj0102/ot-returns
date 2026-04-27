"""Placeholder tests for otreturns.interpolation — implemented in Prompt 4."""

import pytest
import numpy as np
from otreturns.interpolation import mccann_interpolation
from otreturns.distributions import EmpiricalDistribution


def test_mccann_not_implemented():
    rng = np.random.default_rng(0)
    mu0 = EmpiricalDistribution(rng.normal(0, 1, 100))
    mu1 = EmpiricalDistribution(rng.normal(1, 1, 100))
    with pytest.raises(NotImplementedError):
        mccann_interpolation(mu0, mu1, t=0.5)
