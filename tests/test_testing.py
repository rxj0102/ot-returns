"""
Tests for otreturns/testing.py — Prompt 5.

Coverage:
    TestWassersteinTwoSampleTest   — basic contract, H0/H1 power, p-value validity
    TestChangePointTest            — change-point detection around a known shift date
    TestWassersteinCusum           — flat under H0, rising after a shift
    TestEnergyDistanceTest         — basic contract, H0/H1 consistency
    TestDistributionStabilityTest  — output shape and column names
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from otreturns.distributions import EmpiricalDistribution, DistributionPanel
from otreturns.testing import (
    wasserstein_two_sample_test,
    wasserstein_change_point_test,
    wasserstein_cusum,
    energy_distance_test,
    distribution_stability_test,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_panel(means: list, n_stocks: int = 200, seed: int = 0) -> DistributionPanel:
    """Build a DistributionPanel with one date per mean value."""
    rng = np.random.default_rng(seed)
    dates = list(range(len(means)))
    dists = {d: EmpiricalDistribution(rng.normal(m, 1.0, n_stocks))
             for d, m in zip(dates, means)}
    return DistributionPanel(dists, dates)


def _stable_panel(n_dates: int = 60, n_stocks: int = 200, seed: int = 1) -> DistributionPanel:
    """Panel with constant N(0,1) distributions — no change."""
    return _make_panel([0.0] * n_dates, n_stocks=n_stocks, seed=seed)


def _shift_panel(n_before: int = 40, n_after: int = 40,
                 shift: float = 3.0, n_stocks: int = 200, seed: int = 2) -> DistributionPanel:
    """Panel where the mean jumps by `shift` halfway through."""
    means = [0.0] * n_before + [shift] * n_after
    return _make_panel(means, n_stocks=n_stocks, seed=seed)


# ---------------------------------------------------------------------------
# TestWassersteinTwoSampleTest
# ---------------------------------------------------------------------------

class TestWassersteinTwoSampleTest:

    def test_returns_dict_with_required_keys(self):
        rng = np.random.default_rng(0)
        s1 = rng.normal(0, 1, 100)
        s2 = rng.normal(0, 1, 100)
        result = wasserstein_two_sample_test(s1, s2, n_permutations=99, seed=0)
        assert set(result.keys()) >= {"statistic", "p_value",
                                      "permutation_distribution", "reject_h0"}

    def test_statistic_is_nonneg(self):
        rng = np.random.default_rng(1)
        s1, s2 = rng.normal(0, 1, 80), rng.normal(1, 1, 80)
        result = wasserstein_two_sample_test(s1, s2, n_permutations=99, seed=1)
        assert result["statistic"] >= 0.0

    def test_p_value_in_unit_interval(self):
        rng = np.random.default_rng(2)
        s1, s2 = rng.normal(0, 1, 100), rng.normal(5, 1, 100)
        result = wasserstein_two_sample_test(s1, s2, n_permutations=99, seed=2)
        assert 0.0 < result["p_value"] <= 1.0

    def test_permutation_distribution_length(self):
        rng = np.random.default_rng(3)
        s1, s2 = rng.normal(0, 1, 50), rng.normal(0, 1, 50)
        B = 150
        result = wasserstein_two_sample_test(s1, s2, n_permutations=B, seed=3)
        assert len(result["permutation_distribution"]) == B

    def test_permutation_distribution_nonneg(self):
        rng = np.random.default_rng(4)
        s1, s2 = rng.normal(0, 1, 60), rng.normal(0, 1, 60)
        result = wasserstein_two_sample_test(s1, s2, n_permutations=99, seed=4)
        assert np.all(result["permutation_distribution"] >= 0)

    def test_reject_h0_consistent_with_p_value(self):
        rng = np.random.default_rng(5)
        s1, s2 = rng.normal(0, 1, 100), rng.normal(10, 1, 100)
        result = wasserstein_two_sample_test(s1, s2, n_permutations=99, seed=5)
        assert result["reject_h0"] == (result["p_value"] < 0.05)

    def test_identical_samples_high_p_value(self):
        rng = np.random.default_rng(6)
        s = rng.normal(0, 1, 200)
        result = wasserstein_two_sample_test(s, s, n_permutations=199, seed=6)
        assert result["p_value"] > 0.05

    def test_h0_not_rejected_most_of_time(self):
        """Under H0 with alpha=0.05, expect <=15% false rejections."""
        rng = np.random.default_rng(42)
        n_trials = 30
        rejections = 0
        for seed in range(n_trials):
            s1 = rng.normal(0, 1, 100)
            s2 = rng.normal(0, 1, 100)
            r = wasserstein_two_sample_test(s1, s2, n_permutations=199, seed=seed)
            rejections += int(r["reject_h0"])
        assert rejections / n_trials <= 0.15

    def test_h1_rejected_most_of_time(self):
        """Under H1 (large separation), expect >=90% rejections."""
        rng = np.random.default_rng(99)
        n_trials = 20
        rejections = 0
        for seed in range(n_trials):
            s1 = rng.normal(0, 1, 150)
            s2 = rng.normal(5, 1, 150)
            r = wasserstein_two_sample_test(s1, s2, n_permutations=199, seed=seed)
            rejections += int(r["reject_h0"])
        assert rejections / n_trials >= 0.90

    def test_statistic_scales_with_separation(self):
        rng = np.random.default_rng(7)
        s = rng.normal(0, 1, 200)
        t_small = wasserstein_two_sample_test(
            s, rng.normal(1, 1, 200), n_permutations=49, seed=7)["statistic"]
        t_large = wasserstein_two_sample_test(
            s, rng.normal(5, 1, 200), n_permutations=49, seed=8)["statistic"]
        assert t_large > t_small

    def test_seeded_reproducible(self):
        rng = np.random.default_rng(10)
        s1, s2 = rng.normal(0, 1, 80), rng.normal(1, 1, 80)
        r1 = wasserstein_two_sample_test(s1, s2, n_permutations=99, seed=42)
        r2 = wasserstein_two_sample_test(s1, s2, n_permutations=99, seed=42)
        assert r1["p_value"] == r2["p_value"]
        np.testing.assert_array_equal(
            r1["permutation_distribution"], r2["permutation_distribution"])

    def test_different_seeds_different_null(self):
        rng = np.random.default_rng(11)
        s1, s2 = rng.normal(0, 1, 100), rng.normal(0.3, 1, 100)
        r1 = wasserstein_two_sample_test(s1, s2, n_permutations=99, seed=1)
        r2 = wasserstein_two_sample_test(s1, s2, n_permutations=99, seed=2)
        assert not np.array_equal(
            r1["permutation_distribution"], r2["permutation_distribution"])

    def test_unequal_sample_sizes(self):
        rng = np.random.default_rng(12)
        s1, s2 = rng.normal(0, 1, 50), rng.normal(0, 1, 150)
        result = wasserstein_two_sample_test(s1, s2, n_permutations=99, seed=12)
        assert 0.0 < result["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# TestChangePointTest
# ---------------------------------------------------------------------------

class TestChangePointTest:

    def _panel_and_date(self):
        panel = _shift_panel(n_before=30, n_after=30, shift=4.0, seed=20)
        candidate = panel.dates[30]   # first date of the shifted regime
        return panel, candidate

    def test_returns_dict_with_required_keys(self):
        panel, date = self._panel_and_date()
        result = wasserstein_change_point_test(panel, date, window=10,
                                               n_permutations=99)
        assert set(result.keys()) >= {
            "statistic", "p_value", "reject_h0", "effect_size"}

    def test_p_value_in_unit_interval(self):
        panel, date = self._panel_and_date()
        result = wasserstein_change_point_test(panel, date, window=10,
                                               n_permutations=99)
        assert 0.0 < result["p_value"] <= 1.0

    def test_rejects_at_known_change_point(self):
        panel, date = self._panel_and_date()
        result = wasserstein_change_point_test(panel, date, window=15,
                                               n_permutations=299)
        assert result["reject_h0"], (
            f"Expected rejection but p_value={result['p_value']:.4f}")

    def test_does_not_reject_at_stable_mid_point(self):
        panel = _stable_panel(n_dates=60, seed=21)
        mid = panel.dates[30]
        result = wasserstein_change_point_test(panel, mid, window=10,
                                               n_permutations=299)
        # Effect size should be small compared to an actual shift
        assert result["effect_size"] < 1.0

    def test_effect_size_equals_statistic(self):
        panel, date = self._panel_and_date()
        result = wasserstein_change_point_test(panel, date, window=10,
                                               n_permutations=49)
        assert result["effect_size"] == result["statistic"]

    def test_invalid_date_at_boundary_raises(self):
        panel = _stable_panel(n_dates=20, seed=22)
        first_date = panel.dates[0]
        with pytest.raises(ValueError):
            wasserstein_change_point_test(panel, first_date, window=10,
                                          n_permutations=49)


# ---------------------------------------------------------------------------
# TestWassersteinCusum
# ---------------------------------------------------------------------------

class TestWassersteinCusum:

    def test_returns_array(self):
        panel = _stable_panel(n_dates=40, seed=30)
        cusum = wasserstein_cusum(panel)
        assert isinstance(cusum, np.ndarray)

    def test_length_equals_post_reference_dates(self):
        panel = _stable_panel(n_dates=40, seed=31)
        n_ref = max(2, 40 // 4)   # 10 dates in reference
        cusum = wasserstein_cusum(panel)
        assert len(cusum) == 40 - n_ref

    def test_starts_near_zero_under_h0(self):
        """With no regime change, CUSUM stays bounded near zero."""
        panel = _stable_panel(n_dates=80, seed=32)
        cusum = wasserstein_cusum(panel)
        assert np.abs(cusum).max() < 5.0

    def test_rises_after_shift(self):
        n_ref = 20
        n_after = 40
        panel = _shift_panel(n_before=n_ref, n_after=n_after, shift=3.0, seed=33)
        dates = panel.dates
        ref_end = dates[n_ref - 1]
        cusum = wasserstein_cusum(panel, reference_period=(dates[0], ref_end))
        assert len(cusum) == n_after
        assert cusum[-1] > 0.5, f"Expected positive CUSUM, got {cusum[-1]:.4f}"
        assert cusum[-1] > cusum[0]

    def test_custom_reference_period(self):
        panel = _stable_panel(n_dates=60, seed=34)
        dates = panel.dates
        ref = (dates[0], dates[19])
        cusum = wasserstein_cusum(panel, reference_period=ref)
        assert len(cusum) == 40

    def test_too_small_reference_raises(self):
        panel = _stable_panel(n_dates=10, seed=35)
        dates = panel.dates
        with pytest.raises(ValueError):
            wasserstein_cusum(panel, reference_period=(dates[0], dates[0]))

    def test_cusum_flat_then_rising(self):
        """Stable post-ref period followed by a shift: later CUSUM > earlier."""
        n_ref = 20
        n_stable = 20
        n_shift = 30
        means = [0.0] * (n_ref + n_stable) + [4.0] * n_shift
        panel = _make_panel(means, n_stocks=300, seed=36)
        dates = panel.dates
        ref_end = dates[n_ref - 1]
        cusum = wasserstein_cusum(panel, reference_period=(dates[0], ref_end))

        stable_part = cusum[:n_stable]
        shift_part = cusum[n_stable:]
        assert shift_part.mean() > stable_part.mean()


# ---------------------------------------------------------------------------
# TestEnergyDistanceTest
# ---------------------------------------------------------------------------

class TestEnergyDistanceTest:

    def test_returns_dict_with_required_keys(self):
        rng = np.random.default_rng(40)
        s1, s2 = rng.normal(0, 1, 100), rng.normal(0, 1, 100)
        result = energy_distance_test(s1, s2, n_permutations=99)
        assert set(result.keys()) >= {"statistic", "p_value",
                                      "permutation_distribution", "reject_h0"}

    def test_statistic_nonneg(self):
        rng = np.random.default_rng(41)
        s1, s2 = rng.normal(0, 1, 80), rng.normal(2, 1, 80)
        result = energy_distance_test(s1, s2, n_permutations=99)
        assert result["statistic"] >= 0.0

    def test_p_value_in_unit_interval(self):
        rng = np.random.default_rng(42)
        s1, s2 = rng.normal(0, 1, 100), rng.normal(5, 1, 100)
        result = energy_distance_test(s1, s2, n_permutations=99)
        assert 0.0 < result["p_value"] <= 1.0

    def test_permutation_distribution_length(self):
        rng = np.random.default_rng(43)
        s1, s2 = rng.normal(0, 1, 60), rng.normal(0, 1, 60)
        B = 120
        result = energy_distance_test(s1, s2, n_permutations=B)
        assert len(result["permutation_distribution"]) == B

    def test_identical_distributions_high_p_value(self):
        rng = np.random.default_rng(44)
        s = rng.normal(0, 1, 200)
        result = energy_distance_test(s, s, n_permutations=199)
        assert result["p_value"] > 0.05

    def test_h0_not_rejected_most_of_time(self):
        rng = np.random.default_rng(45)
        n_trials = 30
        rejections = 0
        for seed in range(n_trials):
            s1 = rng.normal(0, 1, 100)
            s2 = rng.normal(0, 1, 100)
            r = energy_distance_test(s1, s2, n_permutations=199)
            rejections += int(r["reject_h0"])
        assert rejections / n_trials <= 0.15

    def test_h1_rejected_most_of_time(self):
        rng = np.random.default_rng(46)
        n_trials = 20
        rejections = 0
        for seed in range(n_trials):
            s1 = rng.normal(0, 1, 150)
            s2 = rng.normal(5, 1, 150)
            r = energy_distance_test(s1, s2, n_permutations=199)
            rejections += int(r["reject_h0"])
        assert rejections / n_trials >= 0.90

    def test_consistent_with_wasserstein_under_h0(self):
        """Both tests should fail to reject under H0."""
        rng = np.random.default_rng(47)
        s1, s2 = rng.normal(0, 1, 200), rng.normal(0, 1, 200)
        w_result = wasserstein_two_sample_test(s1, s2, n_permutations=299, seed=47)
        e_result = energy_distance_test(s1, s2, n_permutations=299)
        assert w_result["p_value"] > 0.01
        assert e_result["p_value"] > 0.01

    def test_consistent_with_wasserstein_under_h1(self):
        """Both tests should reject under a large shift."""
        rng = np.random.default_rng(48)
        s1, s2 = rng.normal(0, 1, 200), rng.normal(6, 1, 200)
        w_result = wasserstein_two_sample_test(s1, s2, n_permutations=299, seed=48)
        e_result = energy_distance_test(s1, s2, n_permutations=299)
        assert w_result["reject_h0"]
        assert e_result["reject_h0"]

    def test_energy_zero_for_same_sample(self):
        """E(X, X) = 2E|X-X'| - E|X-X'| - E|X-X'| = 0."""
        rng = np.random.default_rng(49)
        s = rng.normal(0, 1, 100)
        result = energy_distance_test(s, s, n_permutations=49)
        assert abs(result["statistic"]) < 1e-10


# ---------------------------------------------------------------------------
# TestDistributionStabilityTest
# ---------------------------------------------------------------------------

class TestDistributionStabilityTest:

    def _panel(self, seed=50):
        return _stable_panel(n_dates=100, n_stocks=150, seed=seed)

    def test_returns_dataframe(self):
        panel = self._panel()
        df = distribution_stability_test(panel, window=10, step=5)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        panel = self._panel()
        df = distribution_stability_test(panel, window=10, step=5)
        required = {"date", "wasserstein_distance", "p_value",
                    "location_shift", "scale_change", "shape_change"}
        assert required.issubset(set(df.columns))

    def test_p_values_in_unit_interval(self):
        panel = self._panel()
        df = distribution_stability_test(panel, window=10, step=5)
        assert df["p_value"].between(0, 1).all()

    def test_wasserstein_distances_nonneg(self):
        panel = self._panel()
        df = distribution_stability_test(panel, window=10, step=5)
        assert (df["wasserstein_distance"] >= 0).all()

    def test_nonempty_output(self):
        panel = self._panel()
        df = distribution_stability_test(panel, window=10, step=5)
        assert len(df) > 0

    def test_correct_number_of_rows(self):
        panel = self._panel()
        n_dates = len(panel.dates)
        window, step = 10, 5
        expected = len(range(2 * window, n_dates, step))
        df = distribution_stability_test(panel, window=window, step=step)
        assert len(df) == expected

    def test_large_shift_high_wasserstein(self):
        panel = _shift_panel(n_before=50, n_after=50, shift=5.0, seed=51)
        df = distribution_stability_test(panel, window=10, step=5)
        max_dist = df["wasserstein_distance"].max()
        assert max_dist > 1.0, f"Expected large W distance near shift, got {max_dist:.4f}"

    def test_stable_panel_low_wasserstein(self):
        panel = _stable_panel(n_dates=100, n_stocks=300, seed=52)
        df = distribution_stability_test(panel, window=10, step=5)
        assert df["wasserstein_distance"].mean() < 1.0

    def test_location_shift_large_after_mean_change(self):
        panel = _shift_panel(n_before=50, n_after=50, shift=5.0, seed=53)
        df = distribution_stability_test(panel, window=10, step=5)
        max_loc = df["location_shift"].max()
        assert max_loc > 1.0
