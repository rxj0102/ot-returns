"""Regression test: the 1-d Wasserstein distance is exact for unequal sizes and weights.

The general path used to integrate the two step-shaped quantile functions with the
trapezoidal rule, which averages across each jump; for a heavy-tailed sample against a
normal one it understated W2 by about 0.35%.
"""
import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from otreturns.distances import wasserstein_1d
from otreturns.distributions import EmpiricalDistribution as E


@pytest.fixture
def samples():
    rng = np.random.default_rng(1)
    return rng.standard_t(4, 800) * 0.01, rng.normal(0.001, 0.012, 1200), rng.random(800), rng.random(1200)


def test_w1_matches_scipy_unequal_sizes(samples):
    x, y, _, _ = samples
    assert wasserstein_1d(E(x), E(y), p=1) == pytest.approx(wasserstein_distance(x, y), rel=1e-12)


def test_w1_matches_scipy_with_weights(samples):
    x, y, wx, wy = samples
    assert wasserstein_1d(E(x, wx), E(y, wy), p=1) == pytest.approx(wasserstein_distance(x, y, wx, wy), rel=1e-12)


def test_w2_matches_direct_sum_over_merged_breakpoints():
    x = np.array([0.0, 1.0, 3.0])            # quantile steps at 1/3, 2/3
    y = np.array([0.5, 2.0])                 # step at 1/2
    # intervals: [0,1/3): 0 vs 0.5; [1/3,1/2): 1 vs 0.5; [1/2,2/3): 1 vs 2; [2/3,1]: 3 vs 2
    w2sq = (1 / 3) * 0.25 + (1 / 6) * 0.25 + (1 / 6) * 1.0 + (1 / 3) * 1.0
    assert wasserstein_1d(E(x), E(y), p=2) == pytest.approx(np.sqrt(w2sq), rel=1e-12)


def test_general_path_agrees_with_equal_size_fast_path():
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=300), rng.normal(size=300) + 0.2
    w = np.full(300, 1 / 300)
    assert wasserstein_1d(E(x, w * (1 + 1e-9)), E(y, w)) == pytest.approx(wasserstein_1d(E(x), E(y)), rel=1e-9)
