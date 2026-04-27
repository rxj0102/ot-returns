"""
Tests for otreturns.interpolation.

Key theoretical results:

  McCann interpolation (1-d):
      F_{mu_t}^{-1}(u) = (1-t)*F_{source}^{-1}(u) + t*F_{target}^{-1}(u)

  Constant-speed geodesic property:
      W_2(source, mu_t) = t * W_2(source, target)

  Midpoint of N(0,1) and N(2,1):
      F_{mu_{0.5}}^{-1}(u) = 0.5*N^{-1}(u,0,1) + 0.5*N^{-1}(u,2,1)
                            = N^{-1}(u, 1, 1)   => mu_{0.5} = N(1,1)
"""

import numpy as np
import pytest
from scipy.stats import norm as scipy_norm

from otreturns.distributions import EmpiricalDistribution
from otreturns.distances import wasserstein_1d
from otreturns.interpolation import (
    mccann_interpolation,
    interpolation_path,
    multi_marginal_interpolation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gaussian_grid(mu, sigma, n=1000):
    """EmpiricalDistribution built on the midpoint quantile grid."""
    u = (np.arange(n) + 0.5) / n
    return EmpiricalDistribution(scipy_norm.ppf(u, loc=mu, scale=sigma))


# ---------------------------------------------------------------------------
# mccann_interpolation — basic properties
# ---------------------------------------------------------------------------

class TestMcCannInterpolationBasic:

    def test_t0_returns_source(self):
        src = gaussian_grid(0, 1)
        tgt = gaussian_grid(2, 1)
        result = mccann_interpolation(src, tgt, 0.0)
        assert result is src

    def test_t1_returns_target(self):
        src = gaussian_grid(0, 1)
        tgt = gaussian_grid(2, 1)
        result = mccann_interpolation(src, tgt, 1.0)
        assert result is tgt

    def test_returns_empirical_distribution(self):
        src = gaussian_grid(0, 1)
        tgt = gaussian_grid(2, 1)
        mid = mccann_interpolation(src, tgt, 0.5)
        assert isinstance(mid, EmpiricalDistribution)

    def test_output_weights_sum_to_one(self):
        src = gaussian_grid(0, 1)
        tgt = gaussian_grid(2, 1)
        mid = mccann_interpolation(src, tgt, 0.5)
        np.testing.assert_allclose(mid.weights.sum(), 1.0, atol=1e-12)

    def test_invalid_t_below_zero(self):
        src = gaussian_grid(0, 1)
        tgt = gaussian_grid(1, 1)
        with pytest.raises(ValueError):
            mccann_interpolation(src, tgt, -0.01)

    def test_invalid_t_above_one(self):
        src = gaussian_grid(0, 1)
        tgt = gaussian_grid(1, 1)
        with pytest.raises(ValueError):
            mccann_interpolation(src, tgt, 1.01)

    def test_n_support_controls_size(self):
        src = gaussian_grid(0, 1)
        tgt = gaussian_grid(2, 1)
        mid = mccann_interpolation(src, tgt, 0.5, n_support=300)
        assert mid.n == 300

    def test_default_n_support(self):
        src = gaussian_grid(0, 1, n=500)
        tgt = gaussian_grid(2, 1, n=500)
        mid = mccann_interpolation(src, tgt, 0.5)
        assert mid.n == 500


# ---------------------------------------------------------------------------
# mccann_interpolation — Gaussian ground truth
# ---------------------------------------------------------------------------

class TestMcCannInterpolationGaussian:
    N = 1000

    def test_midpoint_mean(self):
        """Midpoint of N(0,1) and N(2,1) has mean = 1."""
        src = gaussian_grid(0.0, 1.0, self.N)
        tgt = gaussian_grid(2.0, 1.0, self.N)
        mid = mccann_interpolation(src, tgt, 0.5, n_support=self.N)
        np.testing.assert_allclose(mid.moments(2)["mean"], 1.0, atol=0.01)

    def test_midpoint_std(self):
        """Midpoint of N(0,1) and N(2,1) has std = 1."""
        src = gaussian_grid(0.0, 1.0, self.N)
        tgt = gaussian_grid(2.0, 1.0, self.N)
        mid = mccann_interpolation(src, tgt, 0.5, n_support=self.N)
        np.testing.assert_allclose(np.sqrt(mid.moments(2)["variance"]), 1.0, atol=0.01)

    def test_interpolated_mean_linear_in_t(self):
        """Mean of mu_t = (1-t)*mean(source) + t*mean(target) for Gaussians."""
        src = gaussian_grid(0.0, 1.0, self.N)
        tgt = gaussian_grid(3.0, 1.0, self.N)
        for t in [0.25, 0.5, 0.75]:
            mid = mccann_interpolation(src, tgt, t, n_support=self.N)
            expected_mean = (1 - t) * 0.0 + t * 3.0
            np.testing.assert_allclose(mid.moments(2)["mean"], expected_mean, atol=0.02)

    def test_interpolated_std_linear_in_t(self):
        """std of mu_t = (1-t)*std(source) + t*std(target) for Gaussians."""
        src = gaussian_grid(0.0, 1.0, self.N)
        tgt = gaussian_grid(0.0, 3.0, self.N)
        for t in [0.25, 0.5, 0.75]:
            mid = mccann_interpolation(src, tgt, t, n_support=self.N)
            expected_std = (1 - t) * 1.0 + t * 3.0
            np.testing.assert_allclose(
                np.sqrt(mid.moments(2)["variance"]), expected_std, atol=0.02
            )

    def test_pushforward_approximates_target_at_t1(self):
        """At t=1 (edge case via direct return) we get the exact target."""
        src = gaussian_grid(0.0, 1.0, self.N)
        tgt = gaussian_grid(2.0, 1.5, self.N)
        result = mccann_interpolation(src, tgt, 1.0)
        np.testing.assert_array_equal(result.samples, tgt.samples)


# ---------------------------------------------------------------------------
# mccann_interpolation — metric properties
# ---------------------------------------------------------------------------

class TestMcCannInterpolationMetric:
    N = 1000

    def test_constant_speed_geodesic(self):
        """
        W_2(source, mu_t) = t * W_2(source, target) exactly
        when source, target, and the interpolated distribution all use
        the same midpoint quantile grid.
        """
        src = gaussian_grid(0.0, 1.0, self.N)
        tgt = gaussian_grid(2.0, 1.0, self.N)
        w_total = wasserstein_1d(src, tgt, p=2)

        for t in [0.1, 0.25, 0.5, 0.75, 0.9]:
            mid = mccann_interpolation(src, tgt, t, n_support=self.N)
            w_t = wasserstein_1d(src, mid, p=2)
            np.testing.assert_allclose(w_t, t * w_total, rtol=1e-6,
                err_msg=f"constant-speed violated at t={t}")

    def test_non_decreasing_distance_from_source(self):
        """W_2(source, mu_t) should increase as t increases."""
        src = gaussian_grid(0.0, 1.0, self.N)
        tgt = gaussian_grid(3.0, 1.5, self.N)
        t_values = np.linspace(0.0, 1.0, 11)
        distances = []
        for t in t_values:
            mid = mccann_interpolation(src, tgt, t, n_support=self.N)
            distances.append(wasserstein_1d(src, mid, p=2))
        diffs = np.diff(distances)
        assert np.all(diffs >= -1e-9), f"Non-monotone: {diffs}"

    def test_symmetry_of_endpoint_distances(self):
        """W_2(source, mu_t) + W_2(mu_t, target) = W_2(source, target)."""
        src = gaussian_grid(0.0, 1.0, self.N)
        tgt = gaussian_grid(2.0, 1.0, self.N)
        w_total = wasserstein_1d(src, tgt, p=2)
        t = 0.4
        mid = mccann_interpolation(src, tgt, t, n_support=self.N)
        w_0t = wasserstein_1d(src, mid, p=2)
        w_t1 = wasserstein_1d(mid, tgt, p=2)
        np.testing.assert_allclose(w_0t + w_t1, w_total, rtol=1e-5)

    def test_identity_has_zero_distance(self):
        """W_2(source, mu_0) = 0 and W_2(target, mu_1) = 0."""
        src = gaussian_grid(0.0, 1.0, self.N)
        tgt = gaussian_grid(2.0, 1.0, self.N)
        assert wasserstein_1d(src, mccann_interpolation(src, tgt, 0.0), p=2) == 0.0
        assert wasserstein_1d(tgt, mccann_interpolation(src, tgt, 1.0), p=2) == 0.0


# ---------------------------------------------------------------------------
# interpolation_path
# ---------------------------------------------------------------------------

class TestInterpolationPath:
    N = 500

    def test_length(self):
        src = gaussian_grid(0, 1, self.N)
        tgt = gaussian_grid(2, 1, self.N)
        path = interpolation_path(src, tgt, n_steps=10)
        assert len(path) == 11

    def test_first_element_is_source(self):
        src = gaussian_grid(0, 1, self.N)
        tgt = gaussian_grid(2, 1, self.N)
        path = interpolation_path(src, tgt, n_steps=5)
        assert path[0] is src

    def test_last_element_is_target(self):
        src = gaussian_grid(0, 1, self.N)
        tgt = gaussian_grid(2, 1, self.N)
        path = interpolation_path(src, tgt, n_steps=5)
        assert path[-1] is tgt

    def test_all_elements_are_empirical_distributions(self):
        src = gaussian_grid(0, 1, self.N)
        tgt = gaussian_grid(2, 1, self.N)
        for dist in interpolation_path(src, tgt, n_steps=5):
            assert isinstance(dist, EmpiricalDistribution)

    def test_monotone_mean_along_path(self):
        src = gaussian_grid(0.0, 1.0, self.N)
        tgt = gaussian_grid(4.0, 1.0, self.N)
        path = interpolation_path(src, tgt, n_steps=8)
        means = [d.moments(2)["mean"] for d in path]
        diffs = np.diff(means)
        assert np.all(diffs >= -0.01), f"Non-monotone means: {diffs}"

    def test_n_steps_1_returns_source_and_target(self):
        src = gaussian_grid(0, 1, self.N)
        tgt = gaussian_grid(2, 1, self.N)
        path = interpolation_path(src, tgt, n_steps=1)
        assert len(path) == 2
        assert path[0] is src
        assert path[1] is tgt

    def test_n_steps_zero_raises(self):
        src = gaussian_grid(0, 1, self.N)
        tgt = gaussian_grid(2, 1, self.N)
        with pytest.raises(ValueError):
            interpolation_path(src, tgt, n_steps=0)

    def test_constant_speed_along_path(self):
        src = gaussian_grid(0.0, 1.0, self.N)
        tgt = gaussian_grid(3.0, 1.0, self.N)
        n_steps = 5
        path = interpolation_path(src, tgt, n_steps=n_steps, n_support=self.N)
        w_total = wasserstein_1d(src, tgt, p=2)
        for k in range(1, n_steps):
            t = k / n_steps
            w_k = wasserstein_1d(src, path[k], p=2)
            np.testing.assert_allclose(w_k, t * w_total, rtol=1e-5,
                err_msg=f"constant speed violated at step {k}")

    def test_n_support_propagated(self):
        src = gaussian_grid(0, 1, self.N)
        tgt = gaussian_grid(2, 1, self.N)
        path = interpolation_path(src, tgt, n_steps=4, n_support=200)
        # Intermediate elements (not endpoints) should have n_support=200
        for d in path[1:-1]:
            assert d.n == 200


# ---------------------------------------------------------------------------
# multi_marginal_interpolation
# ---------------------------------------------------------------------------

class TestMultiMarginalInterpolation:
    N = 500

    def test_at_first_observed_time(self):
        dists = [gaussian_grid(float(i), 1.0, self.N) for i in range(4)]
        times = np.array([0.0, 1.0, 2.0, 3.0])
        result = multi_marginal_interpolation(dists, times, 0.0)
        assert result is dists[0]

    def test_at_last_observed_time(self):
        dists = [gaussian_grid(float(i), 1.0, self.N) for i in range(4)]
        times = np.array([0.0, 1.0, 2.0, 3.0])
        result = multi_marginal_interpolation(dists, times, 3.0)
        assert result is dists[-1]

    def test_before_first_time_returns_first(self):
        dists = [gaussian_grid(float(i), 1.0, self.N) for i in range(3)]
        times = np.array([1.0, 2.0, 3.0])
        result = multi_marginal_interpolation(dists, times, 0.0)
        assert result is dists[0]

    def test_after_last_time_returns_last(self):
        dists = [gaussian_grid(float(i), 1.0, self.N) for i in range(3)]
        times = np.array([0.0, 1.0, 2.0])
        result = multi_marginal_interpolation(dists, times, 5.0)
        assert result is dists[-1]

    def test_midpoint_of_two(self):
        src = gaussian_grid(0.0, 1.0, self.N)
        tgt = gaussian_grid(2.0, 1.0, self.N)
        result = multi_marginal_interpolation([src, tgt], np.array([0.0, 1.0]), 0.5,
                                              n_support=self.N)
        np.testing.assert_allclose(result.moments(2)["mean"], 1.0, atol=0.02)

    def test_uses_correct_interval(self):
        """At t=1.5 the function should interpolate between dists[1] and dists[2]."""
        dists = [gaussian_grid(float(i) * 2.0, 1.0, self.N) for i in range(4)]
        times = np.array([0.0, 1.0, 2.0, 3.0])
        # t=1.5 is halfway between times[1]=1 and times[2]=2
        # => should interpolate between dists[1] (mean=2) and dists[2] (mean=4)
        # => expected mean ≈ 3.0
        result = multi_marginal_interpolation(dists, times, 1.5, n_support=self.N)
        np.testing.assert_allclose(result.moments(2)["mean"], 3.0, atol=0.05)

    def test_monotone_means_along_query(self):
        dists = [gaussian_grid(float(i), 1.0, self.N) for i in range(5)]
        times = np.linspace(0.0, 1.0, 5)
        query_times = np.linspace(0.0, 1.0, 20)
        means = [
            multi_marginal_interpolation(dists, times, qt, n_support=self.N).moments(2)["mean"]
            for qt in query_times
        ]
        diffs = np.diff(means)
        assert np.all(diffs >= -0.05), f"Non-monotone means: {diffs}"

    def test_unsorted_times_handled(self):
        """Times don't need to be sorted; function sorts internally."""
        d0 = gaussian_grid(0.0, 1.0, self.N)
        d1 = gaussian_grid(2.0, 1.0, self.N)
        # Pass in reverse order
        result = multi_marginal_interpolation([d1, d0], np.array([1.0, 0.0]), 0.5,
                                              n_support=self.N)
        np.testing.assert_allclose(result.moments(2)["mean"], 1.0, atol=0.05)

    def test_too_few_distributions_raises(self):
        d = gaussian_grid(0, 1, self.N)
        with pytest.raises(ValueError, match="at least 2"):
            multi_marginal_interpolation([d], np.array([0.0]), 0.5)

    def test_mismatched_lengths_raises(self):
        dists = [gaussian_grid(0, 1, self.N), gaussian_grid(1, 1, self.N)]
        with pytest.raises(ValueError):
            multi_marginal_interpolation(dists, np.array([0.0]), 0.5)

    def test_returns_empirical_distribution(self):
        dists = [gaussian_grid(float(i), 1.0, self.N) for i in range(3)]
        times = np.array([0.0, 0.5, 1.0])
        result = multi_marginal_interpolation(dists, times, 0.3)
        assert isinstance(result, EmpiricalDistribution)
