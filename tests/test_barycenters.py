"""
Tests for otreturns.barycenters.

Key theoretical results:

  1-d barycenter closed form (Agueh & Carlier 2011):
      F_{mu*}^{-1}(u) = sum_k w_k * F_{mu_k}^{-1}(u)

  Barycenter of N(mu_k, sigma_k^2) with equal weights:
      mean = mean(mu_k)
      std  = mean(sigma_k)          [NOT sqrt(mean(sigma_k^2))]

  Example: W_2-barycenter of N(0, 1^2) and N(0, 3^2) has std = 2,
           not sqrt(5) ≈ 2.24 as the L^2 mean would give.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm as scipy_norm

from otreturns.distributions import EmpiricalDistribution, DistributionPanel
from otreturns.barycenters import (
    wasserstein_barycenter_1d,
    wasserstein_barycenter_sinkhorn,
    regime_barycenters,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gaussian_grid(mu, sigma, n=1000):
    """EmpiricalDistribution at the exact midpoint quantile grid for N(mu, sigma^2)."""
    u = (np.arange(n) + 0.5) / n
    return EmpiricalDistribution(scipy_norm.ppf(u, loc=mu, scale=sigma))


def make_panel(n_dates=20, n_stocks=300, mus=None, seed=0):
    rng = np.random.default_rng(seed)
    if mus is None:
        mus = [0.0] * n_dates
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    rows = []
    for d, mu in zip(dates, mus):
        ret = rng.normal(mu, 0.01, n_stocks)
        for i, r in enumerate(ret):
            rows.append({"date": d, "ticker": f"S{i}", "return": r})
    df = pd.DataFrame(rows)
    return DistributionPanel.from_panel(df, min_stocks=50)


# ---------------------------------------------------------------------------
# wasserstein_barycenter_1d — exact closed-form tests
# ---------------------------------------------------------------------------

class TestWassersteinBarycenter1d:

    def test_returns_empirical_distribution(self):
        d = gaussian_grid(0, 1)
        b = wasserstein_barycenter_1d([d])
        assert isinstance(b, EmpiricalDistribution)

    def test_single_distribution_identity(self):
        d = gaussian_grid(0.5, 1.2, n=500)
        b = wasserstein_barycenter_1d([d], n_support=500)
        m_b = b.moments(2)
        m_d = d.moments(2)
        np.testing.assert_allclose(m_b["mean"],     m_d["mean"],     atol=0.01)
        np.testing.assert_allclose(m_b["variance"], m_d["variance"], rtol=0.01)

    def test_identical_distributions_identity(self):
        d = gaussian_grid(0.3, 0.8, n=800)
        b = wasserstein_barycenter_1d([d, d, d], n_support=800)
        m_b = b.moments(2)
        m_d = d.moments(2)
        np.testing.assert_allclose(m_b["mean"],     m_d["mean"],     atol=0.01)
        np.testing.assert_allclose(m_b["variance"], m_d["variance"], rtol=0.02)

    def test_two_gaussians_equal_weights_mean(self):
        # Barycenter of N(0,1) and N(2,1) = N(1,1)
        d0 = gaussian_grid(0.0, 1.0)
        d1 = gaussian_grid(2.0, 1.0)
        b = wasserstein_barycenter_1d([d0, d1], n_support=1000)
        np.testing.assert_allclose(b.moments(2)["mean"], 1.0, atol=0.01)

    def test_two_gaussians_equal_weights_std(self):
        # Both have sigma=1, so barycenter also has sigma=1
        d0 = gaussian_grid(0.0, 1.0)
        d1 = gaussian_grid(2.0, 1.0)
        b = wasserstein_barycenter_1d([d0, d1], n_support=1000)
        std_b = np.sqrt(b.moments(2)["variance"])
        np.testing.assert_allclose(std_b, 1.0, atol=0.01)

    def test_scale_barycenter_averages_stds_not_variances(self):
        """
        W_2-barycenter of N(0, 1^2) and N(0, 3^2) has std = (1+3)/2 = 2,
        NOT sqrt((1^2 + 3^2)/2) = sqrt(5) ≈ 2.236.
        """
        d0 = gaussian_grid(0.0, 1.0, n=2000)
        d1 = gaussian_grid(0.0, 3.0, n=2000)
        b = wasserstein_barycenter_1d([d0, d1], n_support=2000)
        std_b = np.sqrt(b.moments(2)["variance"])

        np.testing.assert_allclose(std_b, 2.0, atol=0.02)   # W_2 average of stds
        assert abs(std_b - np.sqrt(5.0)) > 0.1              # NOT L^2 average of vars

    def test_weighted_location(self):
        # Weighted barycenter of N(0,1) (weight 0.75) and N(4,1) (weight 0.25)
        # => mean = 0.75*0 + 0.25*4 = 1.0
        d0 = gaussian_grid(0.0, 1.0)
        d1 = gaussian_grid(4.0, 1.0)
        b = wasserstein_barycenter_1d([d0, d1], weights=np.array([0.75, 0.25]),
                                      n_support=1000)
        np.testing.assert_allclose(b.moments(2)["mean"], 1.0, atol=0.02)

    def test_output_weights_sum_to_one(self):
        d0 = gaussian_grid(0.0, 1.0)
        d1 = gaussian_grid(2.0, 1.5)
        b = wasserstein_barycenter_1d([d0, d1])
        np.testing.assert_allclose(b.weights.sum(), 1.0, atol=1e-12)

    def test_output_weights_are_uniform(self):
        d0 = gaussian_grid(0.0, 1.0, n=200)
        d1 = gaussian_grid(2.0, 1.5, n=200)
        b = wasserstein_barycenter_1d([d0, d1], n_support=200)
        np.testing.assert_allclose(b.weights, np.full(200, 1.0 / 200), atol=1e-12)

    def test_n_support_controls_output_size(self):
        d0 = gaussian_grid(0.0, 1.0)
        b = wasserstein_barycenter_1d([d0], n_support=300)
        assert b.n == 300

    def test_barycenter_mean_between_inputs(self):
        d0 = gaussian_grid(-2.0, 1.0)
        d1 = gaussian_grid( 3.0, 1.0)
        b  = wasserstein_barycenter_1d([d0, d1], n_support=1000)
        mean_b = b.moments(2)["mean"]
        assert -2.0 <= mean_b <= 3.0

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            wasserstein_barycenter_1d([])

    def test_wrong_weights_length_raises(self):
        d0 = gaussian_grid(0, 1)
        d1 = gaussian_grid(1, 1)
        with pytest.raises(ValueError):
            wasserstein_barycenter_1d([d0, d1], weights=np.array([0.5]))

    def test_three_gaussians(self):
        # Barycenter of N(0,1), N(3,1), N(6,1) with equal weights => N(3,1)
        d0 = gaussian_grid(0.0, 1.0)
        d1 = gaussian_grid(3.0, 1.0)
        d2 = gaussian_grid(6.0, 1.0)
        b = wasserstein_barycenter_1d([d0, d1, d2], n_support=1000)
        np.testing.assert_allclose(b.moments(2)["mean"], 3.0, atol=0.02)

    def test_scale_barycenter_three_gaussians(self):
        # W_2-barycenter of N(0,1), N(0,2), N(0,3) with equal weights
        # => std = (1+2+3)/3 = 2
        d0 = gaussian_grid(0.0, 1.0, n=2000)
        d1 = gaussian_grid(0.0, 2.0, n=2000)
        d2 = gaussian_grid(0.0, 3.0, n=2000)
        b = wasserstein_barycenter_1d([d0, d1, d2], n_support=2000)
        std_b = np.sqrt(b.moments(2)["variance"])
        np.testing.assert_allclose(std_b, 2.0, atol=0.02)


# ---------------------------------------------------------------------------
# wasserstein_barycenter_sinkhorn — approximate tests
# ---------------------------------------------------------------------------

class TestWassersteinBarycentreSinkhorn:
    """
    Sinkhorn barycenter is approximate; tests use loose tolerances.
    With moderate reg the result should still recover the right location/scale.
    """

    def test_returns_empirical_distribution(self):
        d0 = gaussian_grid(0.0, 1.0, n=100)
        d1 = gaussian_grid(2.0, 1.0, n=100)
        b = wasserstein_barycenter_sinkhorn([d0, d1], reg=0.5, n_support=50,
                                            max_iter_outer=20, max_iter_sinkhorn=20)
        assert isinstance(b, EmpiricalDistribution)

    def test_output_weights_sum_to_one(self):
        d0 = gaussian_grid(0.0, 1.0, n=100)
        d1 = gaussian_grid(2.0, 1.0, n=100)
        b = wasserstein_barycenter_sinkhorn([d0, d1], reg=0.5, n_support=50,
                                            max_iter_outer=20, max_iter_sinkhorn=20)
        np.testing.assert_allclose(b.weights.sum(), 1.0, atol=1e-10)

    def test_identical_distributions_mean(self):
        # Barycenter of identical distributions should reproduce the mean
        d = gaussian_grid(1.5, 0.8, n=100)
        b = wasserstein_barycenter_sinkhorn([d, d], reg=0.5, n_support=100,
                                            max_iter_outer=30, max_iter_sinkhorn=30)
        np.testing.assert_allclose(b.moments(2)["mean"], 1.5, atol=0.15)

    def test_two_gaussians_mean_between(self):
        # Barycenter mean should be between the two input means
        d0 = gaussian_grid(0.0, 1.0, n=100)
        d1 = gaussian_grid(2.0, 1.0, n=100)
        b = wasserstein_barycenter_sinkhorn([d0, d1], reg=0.3, n_support=100,
                                            max_iter_outer=30, max_iter_sinkhorn=30)
        mean_b = b.moments(2)["mean"]
        assert 0.0 <= mean_b <= 2.0

    def test_n_support_controls_output_size(self):
        d0 = gaussian_grid(0.0, 1.0, n=50)
        b = wasserstein_barycenter_sinkhorn([d0], reg=0.5, n_support=80,
                                            max_iter_outer=5, max_iter_sinkhorn=5)
        assert b.n == 80

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            wasserstein_barycenter_sinkhorn([])

    def test_positive_reg_required(self):
        d = gaussian_grid(0, 1, n=50)
        with pytest.raises((ValueError, ZeroDivisionError)):
            wasserstein_barycenter_sinkhorn([d, d], reg=0.0, n_support=20,
                                            max_iter_outer=2, max_iter_sinkhorn=2)

    def test_support_range_respected(self):
        d0 = gaussian_grid(0.0, 1.0, n=50)
        d1 = gaussian_grid(2.0, 1.0, n=50)
        b = wasserstein_barycenter_sinkhorn([d0, d1], reg=0.5, n_support=50,
                                            support_range=(-5.0, 7.0),
                                            max_iter_outer=5, max_iter_sinkhorn=5)
        assert b.samples.min() >= -5.0 - 1e-10
        assert b.samples.max() <=  7.0 + 1e-10

    def test_matches_1d_barycenter_approximately(self):
        """Sinkhorn barycenter should approximate the exact 1-d barycenter."""
        d0 = gaussian_grid(0.0, 1.0, n=300)
        d1 = gaussian_grid(2.0, 1.0, n=300)
        exact = wasserstein_barycenter_1d([d0, d1], n_support=200)
        approx = wasserstein_barycenter_sinkhorn([d0, d1], reg=0.1, n_support=200,
                                                  max_iter_outer=50,
                                                  max_iter_sinkhorn=50)
        # Means should be close
        np.testing.assert_allclose(approx.moments(2)["mean"],
                                   exact.moments(2)["mean"], atol=0.15)


# ---------------------------------------------------------------------------
# regime_barycenters
# ---------------------------------------------------------------------------

class TestRegimeBarycenters:

    def _panel_with_regimes(self, n_per_regime=10, n_stocks=300, seed=0):
        """Two-regime panel: regime 0 has mean -0.005, regime 1 has mean +0.005."""
        rng = np.random.default_rng(seed)
        n_total = 2 * n_per_regime
        dates = pd.bdate_range("2020-01-02", periods=n_total)
        labels = np.array([0] * n_per_regime + [1] * n_per_regime)
        rows = []
        for d, lbl in zip(dates, labels):
            mu = -0.005 if lbl == 0 else 0.005
            for i, r in enumerate(rng.normal(mu, 0.01, n_stocks)):
                rows.append({"date": d, "ticker": f"S{i}", "return": r})
        df = pd.DataFrame(rows)
        panel = DistributionPanel.from_panel(df, min_stocks=50)
        return panel, labels

    def test_returns_dict(self):
        panel, labels = self._panel_with_regimes()
        result = regime_barycenters(panel, labels)
        assert isinstance(result, dict)

    def test_correct_regime_keys(self):
        panel, labels = self._panel_with_regimes()
        result = regime_barycenters(panel, labels)
        assert set(result.keys()) == {0, 1}

    def test_values_are_empirical_distributions(self):
        panel, labels = self._panel_with_regimes()
        result = regime_barycenters(panel, labels)
        for v in result.values():
            assert isinstance(v, EmpiricalDistribution)

    def test_regime_means_differ(self):
        """The two regime barycenters should have distinct means."""
        panel, labels = self._panel_with_regimes(n_per_regime=15, seed=1)
        result = regime_barycenters(panel, labels)
        mean0 = result[0].moments(2)["mean"]
        mean1 = result[1].moments(2)["mean"]
        # Regime 0 has negative mean, regime 1 has positive mean
        assert mean0 < mean1

    def test_regime_mean_correct_sign(self):
        panel, labels = self._panel_with_regimes(n_per_regime=20, seed=2)
        result = regime_barycenters(panel, labels)
        assert result[0].moments(2)["mean"] < 0.0
        assert result[1].moments(2)["mean"] > 0.0

    def test_three_regimes(self):
        rng = np.random.default_rng(3)
        n = 30
        dates = pd.bdate_range("2020-01-02", periods=n)
        labels = np.array([0]*10 + [1]*10 + [2]*10)
        rows = []
        mus = {0: -0.01, 1: 0.0, 2: 0.01}
        for d, lbl in zip(dates, labels):
            for i, r in enumerate(rng.normal(mus[lbl], 0.01, 300)):
                rows.append({"date": d, "ticker": f"S{i}", "return": r})
        df = pd.DataFrame(rows)
        panel = DistributionPanel.from_panel(df, min_stocks=50)
        result = regime_barycenters(panel, labels)
        assert set(result.keys()) == {0, 1, 2}
        means = [result[k].moments(2)["mean"] for k in [0, 1, 2]]
        assert means[0] < means[1] < means[2]

    def test_wrong_labels_length_raises(self):
        panel, labels = self._panel_with_regimes()
        with pytest.raises(ValueError, match="length"):
            regime_barycenters(panel, labels[:-1])

    def test_sinkhorn_method(self):
        panel, labels = self._panel_with_regimes(n_per_regime=5)
        result = regime_barycenters(panel, labels, method="sinkhorn",
                                    reg=0.001, n_support=50,
                                    max_iter_outer=10, max_iter_sinkhorn=10)
        assert isinstance(result, dict)
        assert set(result.keys()) == {0, 1}

    def test_unknown_method_raises(self):
        panel, labels = self._panel_with_regimes()
        with pytest.raises(ValueError, match="Unknown method"):
            regime_barycenters(panel, labels, method="bogus")
