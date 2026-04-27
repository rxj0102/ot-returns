"""Tests for otreturns.distributions — EmpiricalDistribution and DistributionPanel."""

import numpy as np
import pandas as pd
import pytest

from otreturns.distributions import EmpiricalDistribution, DistributionPanel


# ---------------------------------------------------------------------------
# EmpiricalDistribution — construction
# ---------------------------------------------------------------------------

class TestEmpiricalDistributionConstruction:
    def test_basic_uniform_weights(self):
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dist = EmpiricalDistribution(samples)
        assert dist.n == 5
        np.testing.assert_allclose(dist.weights, np.full(5, 0.2))

    def test_custom_weights_stored(self):
        samples = np.array([1.0, 2.0, 3.0])
        weights = np.array([0.5, 0.3, 0.2])
        dist = EmpiricalDistribution(samples, weights)
        np.testing.assert_allclose(dist.weights, weights)

    def test_custom_weights_unnormalised_are_normalised(self):
        samples = np.array([1.0, 2.0, 3.0])
        weights = np.array([2.0, 3.0, 5.0])
        dist = EmpiricalDistribution(samples, weights)
        np.testing.assert_allclose(dist.weights.sum(), 1.0)
        np.testing.assert_allclose(dist.weights, weights / 10.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            EmpiricalDistribution(np.array([]))

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="1-dimensional"):
            EmpiricalDistribution(np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_negative_weights_raise(self):
        with pytest.raises(ValueError, match="non-negative"):
            EmpiricalDistribution(np.array([1.0, 2.0]), np.array([-0.5, 1.5]))

    def test_zero_total_weights_raise(self):
        with pytest.raises(ValueError):
            EmpiricalDistribution(np.array([1.0, 2.0]), np.array([0.0, 0.0]))

    def test_weights_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            EmpiricalDistribution(np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.5]))

    def test_len(self):
        dist = EmpiricalDistribution(np.arange(7, dtype=float))
        assert len(dist) == 7

    def test_sorted_samples_are_sorted(self):
        rng = np.random.default_rng(0)
        samples = rng.normal(0, 1, 50)
        dist = EmpiricalDistribution(samples)
        assert np.all(np.diff(dist._sorted_samples) >= 0)

    def test_cumulative_weights_end_at_one(self):
        dist = EmpiricalDistribution(np.arange(1, 11, dtype=float))
        np.testing.assert_allclose(dist._cumulative_weights[-1], 1.0)


# ---------------------------------------------------------------------------
# EmpiricalDistribution — to_histogram
# ---------------------------------------------------------------------------

class TestToHistogram:
    def test_output_shape(self):
        dist = EmpiricalDistribution(np.random.default_rng(0).normal(0, 1, 300))
        centers, w = dist.to_histogram(n_bins=50)
        assert centers.shape == (50,)
        assert w.shape == (50,)

    def test_weights_sum_to_one(self):
        dist = EmpiricalDistribution(np.random.default_rng(1).normal(0, 1, 500))
        _, w = dist.to_histogram(n_bins=100)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)

    def test_bin_centers_within_range(self):
        samples = np.random.default_rng(2).uniform(-1, 1, 200)
        dist = EmpiricalDistribution(samples)
        centers, _ = dist.to_histogram(n_bins=20, range=(-1, 1))
        assert centers.min() >= -1
        assert centers.max() <= 1

    def test_custom_range(self):
        dist = EmpiricalDistribution(np.array([0.5, 1.5, 2.5]))
        centers, w = dist.to_histogram(n_bins=3, range=(0.0, 3.0))
        assert len(centers) == 3

    def test_non_negative_weights(self):
        dist = EmpiricalDistribution(np.random.default_rng(3).normal(0, 1, 100))
        _, w = dist.to_histogram(n_bins=20)
        assert np.all(w >= 0)


# ---------------------------------------------------------------------------
# EmpiricalDistribution — quantile_function
# ---------------------------------------------------------------------------

class TestQuantileFunction:
    def test_monotone(self):
        dist = EmpiricalDistribution(np.random.default_rng(5).normal(0, 1, 1000))
        p = np.linspace(0.01, 0.99, 200)
        q = dist.quantile_function(p)
        assert np.all(np.diff(q) >= 0)

    def test_p_zero_is_min(self):
        samples = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
        dist = EmpiricalDistribution(samples)
        assert dist.quantile_function(np.array([0.0])) == pytest.approx(1.0)

    def test_p_one_is_max(self):
        samples = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
        dist = EmpiricalDistribution(samples)
        assert dist.quantile_function(np.array([1.0])) == pytest.approx(5.0)

    def test_median_of_symmetric_distribution(self):
        rng = np.random.default_rng(7)
        dist = EmpiricalDistribution(rng.normal(0, 1, 10000))
        median = dist.quantile_function(np.array([0.5]))
        np.testing.assert_allclose(median, 0.0, atol=0.05)

    def test_scalar_input_returns_scalar(self):
        dist = EmpiricalDistribution(np.array([1.0, 2.0, 3.0]))
        result = dist.quantile_function(0.5)
        assert np.ndim(result) == 0

    def test_array_input_returns_array(self):
        dist = EmpiricalDistribution(np.array([1.0, 2.0, 3.0]))
        result = dist.quantile_function(np.array([0.25, 0.5, 0.75]))
        assert result.shape == (3,)

    def test_out_of_range_raises(self):
        dist = EmpiricalDistribution(np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError):
            dist.quantile_function(np.array([1.1]))

    def test_known_uniform_quartiles(self):
        # Uniform on {1,2,...,100} — quartile function should be consistent
        dist = EmpiricalDistribution(np.arange(1, 101, dtype=float))
        q25 = dist.quantile_function(np.array([0.25]))
        q75 = dist.quantile_function(np.array([0.75]))
        assert q25 < q75


# ---------------------------------------------------------------------------
# EmpiricalDistribution — cdf
# ---------------------------------------------------------------------------

class TestCDF:
    def test_monotone(self):
        dist = EmpiricalDistribution(np.random.default_rng(9).normal(0, 1, 500))
        x = np.linspace(-4, 4, 300)
        c = dist.cdf(x)
        assert np.all(np.diff(c) >= 0)

    def test_below_min_is_zero(self):
        dist = EmpiricalDistribution(np.array([1.0, 2.0, 3.0]))
        assert dist.cdf(np.array([0.0])) == pytest.approx(0.0)

    def test_at_max_is_one(self):
        dist = EmpiricalDistribution(np.array([1.0, 2.0, 3.0]))
        assert dist.cdf(np.array([3.0])) == pytest.approx(1.0)

    def test_values_in_unit_interval(self):
        dist = EmpiricalDistribution(np.random.default_rng(11).normal(0, 1, 200))
        x = np.linspace(-5, 5, 100)
        c = dist.cdf(x)
        assert np.all(c >= 0)
        assert np.all(c <= 1)

    def test_scalar_input(self):
        dist = EmpiricalDistribution(np.array([1.0, 2.0, 3.0]))
        result = dist.cdf(np.array([2.0]))
        assert np.isscalar(result) or result.ndim <= 1

    def test_half_mass_at_median(self):
        # Uniform: CDF at median ≈ 0.5
        dist = EmpiricalDistribution(np.arange(1, 1001, dtype=float))
        np.testing.assert_allclose(dist.cdf(np.array([500.0])), 0.5, atol=0.01)

    def test_consistency_with_quantile(self):
        # F(F^{-1}(p)) should be close to p for interior points
        rng = np.random.default_rng(13)
        dist = EmpiricalDistribution(rng.normal(0, 1, 2000))
        p = np.linspace(0.05, 0.95, 30)
        q = dist.quantile_function(p)
        p_recovered = dist.cdf(q)
        np.testing.assert_allclose(p_recovered, p, atol=0.02)


# ---------------------------------------------------------------------------
# EmpiricalDistribution — moments
# ---------------------------------------------------------------------------

class TestMoments:
    def test_mean_known(self):
        rng = np.random.default_rng(15)
        mu, sigma = 0.002, 0.015
        dist = EmpiricalDistribution(rng.normal(mu, sigma, 50_000))
        m = dist.moments()
        np.testing.assert_allclose(m["mean"], mu, atol=5 * sigma / np.sqrt(50_000))

    def test_variance_known(self):
        rng = np.random.default_rng(17)
        sigma = 0.02
        dist = EmpiricalDistribution(rng.normal(0, sigma, 50_000))
        m = dist.moments()
        np.testing.assert_allclose(m["variance"], sigma ** 2, rtol=0.05)

    def test_skewness_normal_near_zero(self):
        rng = np.random.default_rng(19)
        dist = EmpiricalDistribution(rng.normal(0, 1, 50_000))
        m = dist.moments()
        np.testing.assert_allclose(m["skewness"], 0.0, atol=0.1)

    def test_excess_kurtosis_normal_near_zero(self):
        rng = np.random.default_rng(21)
        dist = EmpiricalDistribution(rng.normal(0, 1, 50_000))
        m = dist.moments()
        np.testing.assert_allclose(m["kurtosis"], 0.0, atol=0.2)

    def test_order_2_keys(self):
        dist = EmpiricalDistribution(np.arange(1, 11, dtype=float))
        m = dist.moments(order=2)
        assert "mean" in m and "variance" in m
        assert "skewness" not in m

    def test_order_3_has_skewness(self):
        dist = EmpiricalDistribution(np.arange(1, 11, dtype=float))
        m = dist.moments(order=3)
        assert "skewness" in m
        assert "kurtosis" not in m

    def test_order_4_has_kurtosis(self):
        dist = EmpiricalDistribution(np.arange(1, 11, dtype=float))
        m = dist.moments(order=4)
        assert "kurtosis" in m

    def test_single_value_variance_zero(self):
        dist = EmpiricalDistribution(np.array([5.0]))
        m = dist.moments()
        assert m["variance"] == pytest.approx(0.0)
        assert m["skewness"] == pytest.approx(0.0)
        assert m["kurtosis"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# EmpiricalDistribution — descriptive_stats
# ---------------------------------------------------------------------------

class TestDescriptiveStats:
    def test_required_keys(self):
        dist = EmpiricalDistribution(np.random.default_rng(23).normal(0, 1, 100))
        stats = dist.descriptive_stats()
        for key in ("mean", "median", "std", "skewness", "kurtosis",
                    "iqr", "p5", "p95", "n"):
            assert key in stats, f"Missing key: {key}"

    def test_n_matches(self):
        n = 317
        dist = EmpiricalDistribution(np.arange(n, dtype=float))
        assert dist.descriptive_stats()["n"] == n

    def test_std_positive_for_spread_dist(self):
        dist = EmpiricalDistribution(np.random.default_rng(25).normal(0, 1, 200))
        assert dist.descriptive_stats()["std"] > 0

    def test_std_zero_for_constant(self):
        dist = EmpiricalDistribution(np.full(10, 3.14))
        assert dist.descriptive_stats()["std"] == pytest.approx(0.0, abs=1e-10)

    def test_iqr_uniform(self):
        # Uniform on [0, 1]: IQR = 0.5
        rng = np.random.default_rng(27)
        dist = EmpiricalDistribution(rng.uniform(0, 1, 20_000))
        np.testing.assert_allclose(dist.descriptive_stats()["iqr"], 0.5, atol=0.01)

    def test_p5_less_than_p95(self):
        dist = EmpiricalDistribution(np.random.default_rng(29).normal(0, 1, 500))
        stats = dist.descriptive_stats()
        assert stats["p5"] < stats["p95"]

    def test_median_between_p5_and_p95(self):
        dist = EmpiricalDistribution(np.random.default_rng(31).normal(0, 1, 500))
        stats = dist.descriptive_stats()
        assert stats["p5"] <= stats["median"] <= stats["p95"]

    def test_repr_contains_class_name(self):
        dist = EmpiricalDistribution(np.array([1.0, 2.0, 3.0]))
        assert "EmpiricalDistribution" in repr(dist)
        assert "n=3" in repr(dist)


# ---------------------------------------------------------------------------
# DistributionPanel — construction and basic access
# ---------------------------------------------------------------------------

@pytest.fixture
def small_panel():
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-02", periods=10)
    dists = {d: EmpiricalDistribution(rng.normal(0, 1, 100)) for d in dates}
    return DistributionPanel(distributions=dists, dates=list(dates))


class TestDistributionPanelConstruction:
    def test_len(self, small_panel):
        assert len(small_panel) == 10

    def test_dates_are_sorted(self):
        rng = np.random.default_rng(0)
        dates = pd.bdate_range("2020-01-02", periods=5)
        dists = {d: EmpiricalDistribution(rng.normal(0, 1, 50)) for d in dates}
        # Pass reversed
        panel = DistributionPanel(distributions=dists, dates=list(reversed(dates)))
        assert panel.dates == sorted(dates)

    def test_getitem_returns_empirical_dist(self, small_panel):
        date = small_panel.dates[0]
        assert isinstance(small_panel[date], EmpiricalDistribution)

    def test_iteration_yields_dates(self, small_panel):
        assert list(small_panel) == small_panel.dates

    def test_repr_contains_class_name(self, small_panel):
        assert "DistributionPanel" in repr(small_panel)

    def test_repr_empty(self):
        p = DistributionPanel(distributions={}, dates=[])
        assert "empty" in repr(p).lower()


# ---------------------------------------------------------------------------
# DistributionPanel — from_panel
# ---------------------------------------------------------------------------

class TestFromPanel:
    def _make_df(self, n_dates=20, n_stocks=200, seed=0):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2020-01-02", periods=n_dates)
        date_col = np.repeat(dates, n_stocks)
        ticker_col = np.tile([f"S{i}" for i in range(n_stocks)], n_dates)
        return_col = rng.normal(0, 0.01, n_dates * n_stocks)
        return pd.DataFrame({"date": date_col, "ticker": ticker_col, "return": return_col})

    def test_all_dates_included(self):
        df = self._make_df(n_dates=15, n_stocks=150)
        panel = DistributionPanel.from_panel(df, min_stocks=100)
        assert len(panel) == 15

    def test_min_stocks_filters_sparse_dates(self):
        rng = np.random.default_rng(1)
        # Date A: 200 stocks (passes)
        records = [{"date": "2020-01-02", "ticker": f"S{i}",
                    "return": rng.normal()} for i in range(200)]
        # Date B: 50 stocks (filtered out)
        records += [{"date": "2020-01-03", "ticker": f"S{i}",
                     "return": rng.normal()} for i in range(50)]
        df = pd.DataFrame(records)
        panel = DistributionPanel.from_panel(df, min_stocks=100)
        assert len(panel) == 1

    def test_winsorize_clips_extremes(self):
        rng = np.random.default_rng(3)
        returns = rng.normal(0, 0.01, 500)
        returns[0] = 10.0    # extreme outlier
        returns[1] = -10.0   # extreme outlier
        records = [{"date": "2020-01-02", "ticker": f"S{i}", "return": r}
                   for i, r in enumerate(returns)]
        df = pd.DataFrame(records)
        panel = DistributionPanel.from_panel(df, min_stocks=100, winsorize=0.01)
        dist = panel[panel.dates[0]]
        assert dist.samples.max() < 5.0
        assert dist.samples.min() > -5.0

    def test_winsorize_zero_no_clipping(self):
        rng = np.random.default_rng(5)
        returns = rng.normal(0, 0.01, 300)
        returns[0] = 99.0
        records = [{"date": "2020-01-02", "ticker": f"S{i}", "return": r}
                   for i, r in enumerate(returns)]
        df = pd.DataFrame(records)
        panel = DistributionPanel.from_panel(df, min_stocks=100, winsorize=0.0)
        dist = panel[panel.dates[0]]
        assert dist.samples.max() == pytest.approx(99.0)

    def test_nan_returns_dropped(self):
        rng = np.random.default_rng(7)
        returns = rng.normal(0, 0.01, 300).tolist()
        returns[0] = float("nan")
        records = [{"date": "2020-01-02", "ticker": f"S{i}", "return": r}
                   for i, r in enumerate(returns)]
        df = pd.DataFrame(records)
        # Should not raise; nan is simply dropped
        panel = DistributionPanel.from_panel(df, min_stocks=100)
        assert len(panel) == 1

    def test_distributions_are_empirical_distribution(self):
        df = self._make_df(n_dates=5, n_stocks=150)
        panel = DistributionPanel.from_panel(df, min_stocks=100)
        for date in panel.dates:
            assert isinstance(panel[date], EmpiricalDistribution)


# ---------------------------------------------------------------------------
# DistributionPanel — rolling_window
# ---------------------------------------------------------------------------

class TestRollingWindow:
    def test_returns_empirical_distribution(self, small_panel):
        date = small_panel.dates[5]
        rolled = small_panel.rolling_window(date, lookback=3)
        assert isinstance(rolled, EmpiricalDistribution)

    def test_lookback_1_equals_single_date(self, small_panel):
        date = small_panel.dates[4]
        rolled = small_panel.rolling_window(date, lookback=1)
        single = small_panel[date]
        # Same samples (order may differ)
        np.testing.assert_array_equal(
            np.sort(rolled.samples), np.sort(single.samples)
        )

    def test_lookback_pools_samples(self, small_panel):
        # With lookback=3, should have up to 3*100=300 samples
        date = small_panel.dates[5]
        rolled = small_panel.rolling_window(date, lookback=3)
        assert rolled.n == 300  # dates[3], [4], [5]

    def test_lookback_at_start_uses_available(self, small_panel):
        # dates[0] with lookback=10 should only use dates[0]
        date = small_panel.dates[0]
        rolled = small_panel.rolling_window(date, lookback=10)
        assert rolled.n == 100  # only 1 date available

    def test_lookback_2_at_second_date(self, small_panel):
        date = small_panel.dates[1]
        rolled = small_panel.rolling_window(date, lookback=2)
        assert rolled.n == 200  # dates[0] and dates[1]

    def test_uniform_weights(self, small_panel):
        date = small_panel.dates[5]
        rolled = small_panel.rolling_window(date, lookback=3)
        expected_w = 1.0 / rolled.n
        np.testing.assert_allclose(rolled.weights, expected_w)


# ---------------------------------------------------------------------------
# DistributionPanel — subsample
# ---------------------------------------------------------------------------

class TestSubsample:
    def test_returns_distribution_panel(self, small_panel):
        start = small_panel.dates[2]
        end = small_panel.dates[7]
        sub = small_panel.subsample(start, end)
        assert isinstance(sub, DistributionPanel)

    def test_correct_date_range(self, small_panel):
        start = small_panel.dates[2]
        end = small_panel.dates[7]
        sub = small_panel.subsample(start, end)
        assert all(start <= d <= end for d in sub.dates)

    def test_correct_count(self, small_panel):
        start = small_panel.dates[2]
        end = small_panel.dates[7]
        sub = small_panel.subsample(start, end)
        assert len(sub) == 6  # dates[2] through dates[7] inclusive

    def test_single_date(self, small_panel):
        d = small_panel.dates[4]
        sub = small_panel.subsample(d, d)
        assert len(sub) == 1
        assert sub.dates[0] == d

    def test_empty_range(self, small_panel):
        end = small_panel.dates[0]
        start = small_panel.dates[5]
        sub = small_panel.subsample(start, end)  # start > end
        assert len(sub) == 0

    def test_distributions_preserved(self, small_panel):
        start = small_panel.dates[3]
        end = small_panel.dates[6]
        sub = small_panel.subsample(start, end)
        for d in sub.dates:
            np.testing.assert_array_equal(sub[d].samples, small_panel[d].samples)
