"""
Tests for otreturns/regimes.py — Prompt 6.

Coverage:
    TestDistanceMatrix            — shape, symmetry, diagonal, positivity
    TestWassersteinRegimeDetector — fit/predict API, clustering correctness
    TestTransitionMatrix          — row-stochastic, correct shape
    TestSilhouetteAnalysis        — scores in [-1,1], iid null vs structured data
    TestOptimalNRegimes           — recommended_k on synthetic data
    TestRollingRegimeDetection    — output shape, confidence in [0,1]
    TestClusteringMethods         — hierarchical and kmeans_wasserstein backends
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from otreturns.distributions import EmpiricalDistribution, DistributionPanel
from otreturns.regimes import (
    WassersteinRegimeDetector,
    rolling_regime_detection,
    compute_distance_matrix,
    detect_regimes,
    regime_transition_matrix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_panel(means: list, stds: list = None, n_stocks: int = 200,
                seed: int = 0) -> DistributionPanel:
    rng = np.random.default_rng(seed)
    if stds is None:
        stds = [1.0] * len(means)
    dates = list(range(len(means)))
    dists = {
        d: EmpiricalDistribution(rng.normal(m, s, n_stocks))
        for d, m, s in zip(dates, means, stds)
    }
    return DistributionPanel(dists, dates)


def _three_regime_panel(n_per: int = 20, n_stocks: int = 300,
                         seed: int = 0) -> tuple:
    """
    Three clearly-separated regimes: N(-3,1), N(0,1), N(3,1).
    Returns (panel, true_labels).
    """
    means = [-3.0] * n_per + [0.0] * n_per + [3.0] * n_per
    true_labels = np.array([0] * n_per + [1] * n_per + [2] * n_per)
    panel = _make_panel(means, n_stocks=n_stocks, seed=seed)
    return panel, true_labels


def _iid_panel(n_dates: int = 40, n_stocks: int = 200, seed: int = 1) -> DistributionPanel:
    """Panel with no regime structure (iid N(0,1))."""
    return _make_panel([0.0] * n_dates, n_stocks=n_stocks, seed=seed)


# ---------------------------------------------------------------------------
# TestDistanceMatrix
# ---------------------------------------------------------------------------

class TestDistanceMatrix:

    def test_shape(self):
        panel = _iid_panel(n_dates=10)
        D = compute_distance_matrix(panel)
        assert D.shape == (10, 10)

    def test_symmetric(self):
        panel = _iid_panel(n_dates=10)
        D = compute_distance_matrix(panel)
        np.testing.assert_allclose(D, D.T, atol=1e-12)

    def test_diagonal_zero(self):
        panel = _iid_panel(n_dates=10)
        D = compute_distance_matrix(panel)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-12)

    def test_off_diagonal_positive(self):
        panel = _three_regime_panel(n_per=5)[0]
        D = compute_distance_matrix(panel)
        off = D[~np.eye(len(D), dtype=bool)]
        assert np.all(off >= 0)

    def test_between_regime_larger_than_within(self):
        """Cross-regime distances should exceed within-regime on average."""
        panel, _ = _three_regime_panel(n_per=10)
        D = compute_distance_matrix(panel)
        n = 10
        # Within-regime 0
        within = D[:n, :n]
        # Cross regime 0 vs 2
        cross = D[:n, 2 * n:]
        assert cross.mean() > within.mean()


# ---------------------------------------------------------------------------
# TestWassersteinRegimeDetector
# ---------------------------------------------------------------------------

class TestWassersteinRegimeDetector:

    def test_fit_returns_self(self):
        panel = _iid_panel(n_dates=15)
        det = WassersteinRegimeDetector(n_regimes=2)
        assert det.fit(panel) is det

    def test_fit_predict_length(self):
        panel, _ = _three_regime_panel(n_per=10)
        det = WassersteinRegimeDetector(n_regimes=3)
        labels = det.fit_predict(panel)
        assert len(labels) == len(panel.dates)

    def test_labels_in_range(self):
        panel, _ = _three_regime_panel(n_per=10)
        det = WassersteinRegimeDetector(n_regimes=3)
        labels = det.fit_predict(panel)
        assert set(labels).issubset({0, 1, 2})

    def test_ari_high_on_structured_data(self):
        """Adjusted Rand Index > 0.7 on clearly separated 3-regime data."""
        panel, true_labels = _three_regime_panel(n_per=20, n_stocks=300, seed=7)
        det = WassersteinRegimeDetector(n_regimes=3)
        labels = det.fit_predict(panel)
        ari = adjusted_rand_score(true_labels, labels)
        assert ari > 0.7, f"ARI={ari:.3f} too low for well-separated regimes"

    def test_distance_matrix_property(self):
        panel = _iid_panel(n_dates=12)
        det = WassersteinRegimeDetector(n_regimes=2)
        det.fit(panel)
        D = det.distance_matrix
        assert D.shape == (12, 12)
        np.testing.assert_allclose(D, D.T, atol=1e-12)

    def test_regime_barycenters_keys(self):
        panel, _ = _three_regime_panel(n_per=10)
        det = WassersteinRegimeDetector(n_regimes=3)
        det.fit(panel)
        bary = det.regime_barycenters
        assert len(bary) == 3
        assert all(isinstance(v, EmpiricalDistribution) for v in bary.values())

    def test_predict_returns_int(self):
        panel, _ = _three_regime_panel(n_per=10)
        det = WassersteinRegimeDetector(n_regimes=3)
        det.fit(panel)
        rng = np.random.default_rng(42)
        new_dist = EmpiricalDistribution(rng.normal(0, 1, 100))
        label = det.predict(new_dist)
        assert isinstance(label, int)
        assert label in {0, 1, 2}

    def test_predict_correct_regime(self):
        """Distribution clearly from regime 0 (N(-3,1)) → assigned to regime 0's label."""
        panel, true_labels = _three_regime_panel(n_per=20, n_stocks=300, seed=5)
        det = WassersteinRegimeDetector(n_regimes=3)
        det.fit(panel)

        # Find which fitted label corresponds to the mean=-3 cluster
        bary = det.regime_barycenters
        # The barycenter with the lowest mean should correspond to N(-3,1) regime
        means = {lbl: b.moments(1)["mean"] for lbl, b in bary.items()}
        neg_label = min(means, key=means.get)

        rng = np.random.default_rng(99)
        dist_neg = EmpiricalDistribution(rng.normal(-3.0, 1.0, 500))
        assert det.predict(dist_neg) == neg_label

    def test_predict_before_fit_raises(self):
        det = WassersteinRegimeDetector(n_regimes=2)
        rng = np.random.default_rng(0)
        new_dist = EmpiricalDistribution(rng.normal(0, 1, 50))
        with pytest.raises(RuntimeError):
            det.predict(new_dist)

    def test_n_regimes_less_than_2_raises(self):
        with pytest.raises(ValueError):
            WassersteinRegimeDetector(n_regimes=1)

    def test_two_regime_detection(self):
        panel = _make_panel([-5.0] * 15 + [5.0] * 15, n_stocks=200, seed=10)
        true_labels = np.array([0] * 15 + [1] * 15)
        det = WassersteinRegimeDetector(n_regimes=2)
        labels = det.fit_predict(panel)
        ari = adjusted_rand_score(true_labels, labels)
        assert ari > 0.7

    def test_window_mode(self):
        """Rolling-window mode should run without error."""
        panel = _iid_panel(n_dates=20, n_stocks=100)
        det = WassersteinRegimeDetector(n_regimes=2, window=5)
        labels = det.fit_predict(panel)
        assert len(labels) == 20

    def test_unknown_clustering_method_raises(self):
        panel = _iid_panel(n_dates=10)
        det = WassersteinRegimeDetector(n_regimes=2, clustering_method="bad_method")
        with pytest.raises(ValueError):
            det.fit(panel)


# ---------------------------------------------------------------------------
# TestTransitionMatrix
# ---------------------------------------------------------------------------

class TestTransitionMatrix:

    def test_shape(self):
        panel, _ = _three_regime_panel(n_per=15)
        det = WassersteinRegimeDetector(n_regimes=3)
        det.fit(panel)
        P = det.transition_matrix()
        assert P.shape == (3, 3)

    def test_rows_sum_to_one(self):
        panel, _ = _three_regime_panel(n_per=15)
        det = WassersteinRegimeDetector(n_regimes=3)
        det.fit(panel)
        P = det.transition_matrix()
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)

    def test_entries_nonneg(self):
        panel, _ = _three_regime_panel(n_per=15)
        det = WassersteinRegimeDetector(n_regimes=3)
        det.fit(panel)
        P = det.transition_matrix()
        assert np.all(P >= 0)

    def test_diagonal_high_for_persistent_regimes(self):
        """
        With 20 consecutive dates in each regime, diagonal entries
        (self-transitions) should dominate.
        """
        panel, _ = _three_regime_panel(n_per=20)
        det = WassersteinRegimeDetector(n_regimes=3)
        det.fit(panel)
        P = det.transition_matrix()
        # Diagonal entries collectively hold more probability than off-diagonal
        assert P.trace() / 3 > 0.5

    def test_functional_api(self):
        labels = np.array([0, 0, 1, 1, 2, 2, 0, 1])
        P = regime_transition_matrix(labels)
        assert P.shape == (3, 3)
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# TestSilhouetteAnalysis
# ---------------------------------------------------------------------------

class TestSilhouetteAnalysis:

    def test_returns_dict_with_required_keys(self):
        panel, _ = _three_regime_panel(n_per=10)
        det = WassersteinRegimeDetector(n_regimes=3)
        det.fit(panel)
        result = det.silhouette_analysis()
        assert set(result.keys()) >= {"scores", "mean_score", "per_regime_scores"}

    def test_scores_length(self):
        panel, _ = _three_regime_panel(n_per=10)
        det = WassersteinRegimeDetector(n_regimes=3)
        det.fit(panel)
        result = det.silhouette_analysis()
        assert len(result["scores"]) == len(panel.dates)

    def test_scores_in_valid_range(self):
        panel, _ = _three_regime_panel(n_per=10)
        det = WassersteinRegimeDetector(n_regimes=3)
        det.fit(panel)
        result = det.silhouette_analysis()
        assert np.all(result["scores"] >= -1 - 1e-9)
        assert np.all(result["scores"] <= 1 + 1e-9)

    def test_mean_score_high_on_structured_data(self):
        """Well-separated clusters → high silhouette."""
        panel, _ = _three_regime_panel(n_per=20, n_stocks=300, seed=3)
        det = WassersteinRegimeDetector(n_regimes=3)
        det.fit(panel)
        result = det.silhouette_analysis()
        assert result["mean_score"] > 0.3, \
            f"Expected high silhouette on structured data, got {result['mean_score']:.3f}"

    def test_mean_score_lower_on_iid(self):
        """iid data → silhouette should be lower than structured data."""
        panel_iid = _iid_panel(n_dates=30, n_stocks=300, seed=4)
        panel_str, _ = _three_regime_panel(n_per=10, n_stocks=300, seed=4)

        det_iid = WassersteinRegimeDetector(n_regimes=3)
        det_iid.fit(panel_iid)
        sil_iid = det_iid.silhouette_analysis()["mean_score"]

        det_str = WassersteinRegimeDetector(n_regimes=3)
        det_str.fit(panel_str)
        sil_str = det_str.silhouette_analysis()["mean_score"]

        assert sil_str > sil_iid

    def test_per_regime_scores_keys(self):
        panel, _ = _three_regime_panel(n_per=10)
        det = WassersteinRegimeDetector(n_regimes=3)
        det.fit(panel)
        result = det.silhouette_analysis()
        assert len(result["per_regime_scores"]) == 3


# ---------------------------------------------------------------------------
# TestOptimalNRegimes
# ---------------------------------------------------------------------------

class TestOptimalNRegimes:

    def test_returns_dict_with_required_keys(self):
        panel, _ = _three_regime_panel(n_per=10)
        det = WassersteinRegimeDetector(n_regimes=2)
        result = det.optimal_n_regimes(panel, max_regimes=4)
        assert set(result.keys()) >= {
            "k_values", "silhouette_scores", "gap_statistics", "recommended_k"}

    def test_k_values_range(self):
        panel, _ = _three_regime_panel(n_per=10)
        det = WassersteinRegimeDetector(n_regimes=2)
        result = det.optimal_n_regimes(panel, max_regimes=5)
        assert result["k_values"] == [2, 3, 4, 5]

    def test_lengths_consistent(self):
        panel, _ = _three_regime_panel(n_per=10)
        det = WassersteinRegimeDetector(n_regimes=2)
        result = det.optimal_n_regimes(panel, max_regimes=4)
        assert (len(result["k_values"])
                == len(result["silhouette_scores"])
                == len(result["gap_statistics"]))

    def test_recommended_k_in_range(self):
        panel, _ = _three_regime_panel(n_per=10)
        det = WassersteinRegimeDetector(n_regimes=2)
        result = det.optimal_n_regimes(panel, max_regimes=5)
        assert result["recommended_k"] in result["k_values"]

    def test_recovers_true_k_on_synthetic_data(self):
        """
        On well-separated 3-regime data the recommended k should be 3.
        We allow ±1 tolerance since silhouette is heuristic.
        """
        panel, _ = _three_regime_panel(n_per=20, n_stocks=400, seed=42)
        det = WassersteinRegimeDetector(n_regimes=2)
        result = det.optimal_n_regimes(panel, max_regimes=6)
        k_hat = result["recommended_k"]
        assert abs(k_hat - 3) <= 1, \
            f"Expected k≈3 for 3-regime data, recommended_k={k_hat}"


# ---------------------------------------------------------------------------
# TestRollingRegimeDetection
# ---------------------------------------------------------------------------

class TestRollingRegimeDetection:

    def _panel(self):
        return _three_regime_panel(n_per=25, n_stocks=200, seed=20)[0]

    def test_returns_dataframe(self):
        panel = self._panel()
        df = rolling_regime_detection(panel, lookback=20, n_regimes=3, step=5)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        panel = self._panel()
        df = rolling_regime_detection(panel, lookback=20, n_regimes=3, step=5)
        assert set(df.columns) >= {"date", "regime", "confidence"}

    def test_regime_values_valid(self):
        panel = self._panel()
        df = rolling_regime_detection(panel, lookback=20, n_regimes=3, step=5)
        assert df["regime"].between(0, 2).all()

    def test_confidence_in_unit_interval(self):
        panel = self._panel()
        df = rolling_regime_detection(panel, lookback=20, n_regimes=3, step=5)
        assert df["confidence"].between(0.0, 1.0).all()

    def test_nonempty_output(self):
        panel = self._panel()
        df = rolling_regime_detection(panel, lookback=20, n_regimes=3, step=5)
        assert len(df) > 0

    def test_dates_are_panel_dates(self):
        panel = self._panel()
        df = rolling_regime_detection(panel, lookback=20, n_regimes=3, step=5)
        assert all(d in panel.dates for d in df["date"])

    def test_dates_monotone(self):
        panel = self._panel()
        df = rolling_regime_detection(panel, lookback=20, n_regimes=3, step=5)
        dates = list(df["date"])
        assert dates == sorted(dates)


# ---------------------------------------------------------------------------
# TestClusteringMethods
# ---------------------------------------------------------------------------

class TestClusteringMethods:

    def test_hierarchical_runs(self):
        panel, _ = _three_regime_panel(n_per=10)
        det = WassersteinRegimeDetector(n_regimes=3, clustering_method="hierarchical")
        labels = det.fit_predict(panel)
        assert len(labels) == len(panel.dates)
        assert set(labels).issubset({0, 1, 2})

    def test_hierarchical_ari(self):
        panel, true_labels = _three_regime_panel(n_per=20, n_stocks=300, seed=8)
        det = WassersteinRegimeDetector(n_regimes=3, clustering_method="hierarchical")
        labels = det.fit_predict(panel)
        ari = adjusted_rand_score(true_labels, labels)
        assert ari > 0.7, f"Hierarchical ARI={ari:.3f}"

    def test_kmeans_wasserstein_runs(self):
        panel, _ = _three_regime_panel(n_per=10)
        det = WassersteinRegimeDetector(n_regimes=3,
                                        clustering_method="kmeans_wasserstein")
        labels = det.fit_predict(panel)
        assert len(labels) == len(panel.dates)

    def test_kmeans_wasserstein_ari(self):
        panel, true_labels = _three_regime_panel(n_per=20, n_stocks=300, seed=9)
        det = WassersteinRegimeDetector(n_regimes=3,
                                        clustering_method="kmeans_wasserstein")
        labels = det.fit_predict(panel)
        ari = adjusted_rand_score(true_labels, labels)
        assert ari > 0.7, f"K-medoids ARI={ari:.3f}"

    def test_all_methods_agree_on_clear_data(self):
        """All three methods should achieve ARI > 0.7 on obvious clusters."""
        panel, true_labels = _three_regime_panel(n_per=20, n_stocks=400, seed=11)
        for method in ["spectral", "hierarchical", "kmeans_wasserstein"]:
            det = WassersteinRegimeDetector(n_regimes=3, clustering_method=method)
            labels = det.fit_predict(panel)
            ari = adjusted_rand_score(true_labels, labels)
            assert ari > 0.7, f"method={method} ARI={ari:.3f}"
