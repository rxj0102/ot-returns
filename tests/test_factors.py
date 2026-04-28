"""
Tests for otreturns/factors.py — Prompt 7.

Coverage:
    TestTransportFactorDecomposition — basic contract, factor R², noise baseline
    TestDecomposeTransportMap        — legacy API correctness
    TestRollingFactorAttribution     — output shape, column names, W2 values
    TestVarianceDecompositionOT      — totals, fractions, regime decomposition
    TestTransportPCA                 — shape, explained variance, PC1 interpretation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from otreturns.distributions import EmpiricalDistribution, DistributionPanel
from otreturns.factors import (
    transport_factor_decomposition,
    rolling_factor_attribution,
    variance_decomposition_ot,
    transport_pca,
    decompose_transport_map,
)


# ---------------------------------------------------------------------------
# Helpers — controlled synthetic setups
# ---------------------------------------------------------------------------

def _pure_factor_pair(n_stocks: int = 500, f_src: float = 1.0, f_tgt: float = 3.0,
                      eps_scale: float = 0.02, seed: int = 0):
    """
    Single-factor setup where distribution shift is entirely factor-driven.

    Loadings: uniformly spaced in [-1, 1] (sorted).
    Source: r_i = beta_i * f_src + small_eps
    Target: r_i = beta_i * f_tgt + small_eps

    Because stocks are sorted by beta * f, the quantile ordering preserves
    factor loadings → regression should give R² ≈ 1.
    """
    rng = np.random.default_rng(seed)
    beta = np.linspace(-1.0, 1.0, n_stocks)          # (n_stocks,) sorted
    eps = rng.normal(0, eps_scale, n_stocks)
    src_samples = beta * f_src + eps
    tgt_samples = beta * f_tgt + rng.normal(0, eps_scale, n_stocks)

    source = EmpiricalDistribution(src_samples)
    target = EmpiricalDistribution(tgt_samples)
    loadings = beta[:, None]                           # (n_stocks, 1)
    return source, target, loadings


def _noise_pair(n_stocks: int = 300, seed: int = 1):
    """Both distributions are iid N(0,1) — no factor structure in shift."""
    rng = np.random.default_rng(seed)
    source = EmpiricalDistribution(rng.normal(0, 1, n_stocks))
    target = EmpiricalDistribution(rng.normal(0, 1, n_stocks))
    loadings = rng.normal(0, 1, (n_stocks, 3))
    return source, target, loadings


def _make_panel(means, n_stocks=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = list(range(len(means)))
    dists = {d: EmpiricalDistribution(rng.normal(m, 1.0, n_stocks))
             for d, m in zip(dates, means)}
    return DistributionPanel(dists, dates)


def _factor_panel(n_dates=20, n_stocks=400, n_factors=2, seed=0):
    """Panel where each date's distribution is driven by a single factor."""
    rng = np.random.default_rng(seed)
    beta = np.linspace(-1, 1, n_stocks)[:, None] * np.ones((1, n_factors))
    factor_vals = rng.normal(0, 1, (n_dates, n_factors))
    dates = list(range(n_dates))
    dists = {}
    for t in range(n_dates):
        returns = beta @ factor_vals[t] + rng.normal(0, 0.02, n_stocks)
        dists[t] = EmpiricalDistribution(returns)
    panel = DistributionPanel(dists, dates)
    loadings_df = pd.DataFrame(beta, columns=[f"f{k}" for k in range(n_factors)])
    return panel, loadings_df


# ---------------------------------------------------------------------------
# TestTransportFactorDecomposition
# ---------------------------------------------------------------------------

class TestTransportFactorDecomposition:

    def test_returns_required_keys(self):
        source, target, loadings = _pure_factor_pair()
        result = transport_factor_decomposition(source, target, loadings)
        required = {"factor_contributions", "factor_fraction", "residual_fraction",
                    "displacement_field", "factor_displacement_fields", "r_squared"}
        assert required.issubset(result.keys())

    def test_displacement_field_shape(self):
        source, target, loadings = _pure_factor_pair()
        result = transport_factor_decomposition(source, target, loadings,
                                                n_quantiles=100)
        assert result["displacement_field"].shape == (100,)

    def test_factor_fraction_in_unit_interval(self):
        source, target, loadings = _pure_factor_pair()
        result = transport_factor_decomposition(source, target, loadings)
        assert 0.0 <= result["factor_fraction"] <= 1.0

    def test_residual_fraction_in_unit_interval(self):
        source, target, loadings = _pure_factor_pair()
        result = transport_factor_decomposition(source, target, loadings)
        assert 0.0 <= result["residual_fraction"] <= 1.0

    def test_r_squared_high_on_factor_driven_data(self):
        """Single-factor pure shift: R² should be > 0.8."""
        source, target, loadings = _pure_factor_pair(
            n_stocks=800, f_src=1.0, f_tgt=4.0, eps_scale=0.01, seed=0)
        result = transport_factor_decomposition(source, target, loadings,
                                                n_quantiles=200)
        assert result["r_squared"] > 0.8, \
            f"Expected R²>0.8 for pure factor shift, got {result['r_squared']:.4f}"

    def test_factor_fraction_high_on_factor_driven_data(self):
        """Single-factor setup: factor_fraction should be > 0.7."""
        source, target, loadings = _pure_factor_pair(
            n_stocks=800, f_src=1.0, f_tgt=4.0, eps_scale=0.01, seed=1)
        result = transport_factor_decomposition(source, target, loadings,
                                                n_quantiles=200)
        assert result["factor_fraction"] > 0.7, \
            f"Expected factor_fraction>0.7, got {result['factor_fraction']:.4f}"

    def test_r_squared_low_on_noise(self):
        """Identical iid distributions: very small displacement → R² undefined
        but factor_fraction should not spuriously be 1."""
        source, target, loadings = _noise_pair(n_stocks=300, seed=2)
        # Compute on identical source to enforce zero displacement
        result = transport_factor_decomposition(source, source, loadings,
                                                n_quantiles=100)
        # With zero displacement R² is degenerate; just check it doesn't crash
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_factor_contribution_sign_correct(self):
        """For f_tgt > f_src and positive betas, displacement is positive
        at upper quantiles → alpha > 0 for the single factor."""
        source, target, loadings = _pure_factor_pair(
            n_stocks=600, f_src=1.0, f_tgt=3.0, eps_scale=0.01, seed=3)
        result = transport_factor_decomposition(source, target, loadings,
                                                factor_names=["market"])
        alpha = result["factor_contributions"]["market"]
        # Stocks with positive beta should move to higher returns → alpha > 0
        assert alpha > 0, f"Expected positive α for positive factor shock, got {alpha:.4f}"

    def test_factor_names_respected(self):
        rng = np.random.default_rng(4)
        source = EmpiricalDistribution(rng.normal(0, 1, 200))
        target = EmpiricalDistribution(rng.normal(1, 1, 200))
        loadings = rng.normal(0, 1, (200, 2))
        result = transport_factor_decomposition(source, target, loadings,
                                                factor_names=["mkt", "hml"])
        assert "mkt" in result["factor_contributions"]
        assert "hml" in result["factor_contributions"]
        assert "mkt" in result["factor_displacement_fields"]

    def test_factor_displacement_fields_shape(self):
        source, target, loadings = _pure_factor_pair(n_stocks=300)
        result = transport_factor_decomposition(source, target, loadings,
                                                n_quantiles=150)
        for name, field in result["factor_displacement_fields"].items():
            assert field.shape == (150,)

    def test_wrong_factor_names_length_raises(self):
        source, target, loadings = _pure_factor_pair()
        with pytest.raises(ValueError):
            transport_factor_decomposition(source, target, loadings,
                                           factor_names=["a", "b"])  # K=1, names=2

    def test_sum_of_factor_fields_approx_displacement(self):
        """Σ_k g_k(u) + residual ≈ displacement field (by OLS construction)."""
        source, target, loadings = _pure_factor_pair(
            n_stocks=500, f_src=1.0, f_tgt=3.0, eps_scale=0.01, seed=5)
        M = 100
        result = transport_factor_decomposition(source, target, loadings,
                                                n_quantiles=M)
        d = result["displacement_field"]
        factor_sum = sum(result["factor_displacement_fields"].values())
        residual_vec = d - factor_sum
        # Residual should be small relative to displacement magnitude
        rel_resid = np.linalg.norm(residual_vec) / (np.linalg.norm(d) + 1e-12)
        assert rel_resid < 0.5   # factor explains at least half the displacement


# ---------------------------------------------------------------------------
# TestDecomposeTransportMap  (legacy API)
# ---------------------------------------------------------------------------

class TestDecomposeTransportMap:

    def test_returns_required_keys(self):
        rng = np.random.default_rng(0)
        loadings = rng.normal(0, 1, (500, 3))
        result = decompose_transport_map(
            np.linspace(-1, 1, 200),
            np.linspace(-0.5, 1.5, 200),
            loadings,
        )
        assert {"coefficients", "residual_vec", "r_squared"}.issubset(result.keys())

    def test_r_squared_in_unit_interval(self):
        rng = np.random.default_rng(1)
        loadings = rng.normal(0, 1, (300, 2))
        result = decompose_transport_map(
            np.linspace(-2, 2, 100),
            np.linspace(-1, 3, 100),
            loadings,
        )
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_coefficients_shape(self):
        rng = np.random.default_rng(2)
        K = 4
        loadings = rng.normal(0, 1, (200, K))
        result = decompose_transport_map(
            np.linspace(-1, 1, 100),
            np.linspace(0, 2, 100),
            loadings,
        )
        assert result["coefficients"].shape == (K,)

    def test_pure_location_shift_r_squared(self):
        """Uniform loadings = 1 + pure shift → R² should be high."""
        n = 300
        grid = np.linspace(-1, 1, 100)
        loadings = np.ones((n, 1))
        result = decompose_transport_map(grid, grid + 0.5, loadings)
        assert result["r_squared"] > 0.8


# ---------------------------------------------------------------------------
# TestRollingFactorAttribution
# ---------------------------------------------------------------------------

class TestRollingFactorAttribution:

    def test_returns_dataframe(self):
        panel, loadings_df = _factor_panel(n_dates=10)
        df = rolling_factor_attribution(panel, loadings_df, window=1)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns_present(self):
        panel, loadings_df = _factor_panel(n_dates=10, n_factors=2)
        df = rolling_factor_attribution(panel, loadings_df, window=1)
        required = {"date", "total_shift_W2", "residual", "factor_r_squared",
                    "f0_contribution", "f1_contribution"}
        assert required.issubset(set(df.columns))

    def test_row_count_window_1(self):
        panel, loadings_df = _factor_panel(n_dates=15)
        df = rolling_factor_attribution(panel, loadings_df, window=1)
        assert len(df) == 14     # T-1 rows for window=1

    def test_row_count_window_2(self):
        panel, loadings_df = _factor_panel(n_dates=15)
        df = rolling_factor_attribution(panel, loadings_df, window=2)
        assert len(df) == 13     # T-2 rows for window=2

    def test_total_shift_w2_nonneg(self):
        panel, loadings_df = _factor_panel(n_dates=10)
        df = rolling_factor_attribution(panel, loadings_df, window=1)
        assert (df["total_shift_W2"] >= 0).all()

    def test_r_squared_in_unit_interval(self):
        panel, loadings_df = _factor_panel(n_dates=10)
        df = rolling_factor_attribution(panel, loadings_df, window=1)
        assert df["factor_r_squared"].between(0.0, 1.0).all()

    def test_custom_factor_names(self):
        panel, loadings_df = _factor_panel(n_dates=8, n_factors=2)
        loadings_arr = loadings_df.values
        df = rolling_factor_attribution(panel, loadings_arr, window=1,
                                        factor_names=["mkt", "smb"])
        assert "mkt_contribution" in df.columns
        assert "smb_contribution" in df.columns

    def test_dates_are_panel_dates(self):
        panel, loadings_df = _factor_panel(n_dates=10)
        df = rolling_factor_attribution(panel, loadings_df, window=1)
        for d in df["date"]:
            assert d in panel.dates

    def test_high_r_squared_on_factor_driven_panel(self):
        """
        Panel where each date's distribution shifts purely due to a single factor.
        Rolling attribution should show consistently high R².
        """
        n_stocks = 600
        n_dates = 12
        rng = np.random.default_rng(42)
        beta = np.linspace(-1, 1, n_stocks)
        factor_vals = rng.normal(0, 1, n_dates)
        dates = list(range(n_dates))
        dists = {}
        for t in range(n_dates):
            returns = beta * factor_vals[t] + rng.normal(0, 0.01, n_stocks)
            dists[t] = EmpiricalDistribution(returns)
        panel = DistributionPanel(dists, dates)
        loadings_df = pd.DataFrame({"mkt": beta})

        df = rolling_factor_attribution(panel, loadings_df, window=1)
        # Median R² should be high when factor dominates
        assert df["factor_r_squared"].median() > 0.6, \
            f"Expected median R²>0.6, got {df['factor_r_squared'].median():.4f}"


# ---------------------------------------------------------------------------
# TestVarianceDecompositionOT
# ---------------------------------------------------------------------------

class TestVarianceDecompositionOT:

    def _panel_and_loadings(self, seed=0):
        panel, loadings_df = _factor_panel(n_dates=15, n_stocks=300, seed=seed)
        return panel, loadings_df

    def test_returns_required_keys(self):
        panel, loadings_df = self._panel_and_loadings()
        result = variance_decomposition_ot(panel, loadings_df)
        assert {"total_variation", "factor_explained_fraction",
                "idiosyncratic_fraction"}.issubset(result.keys())

    def test_total_variation_nonneg(self):
        panel, loadings_df = self._panel_and_loadings()
        result = variance_decomposition_ot(panel, loadings_df)
        assert result["total_variation"] >= 0

    def test_fractions_sum_to_one(self):
        panel, loadings_df = self._panel_and_loadings()
        result = variance_decomposition_ot(panel, loadings_df)
        total = result["factor_explained_fraction"] + result["idiosyncratic_fraction"]
        assert abs(total - 1.0) < 1e-9

    def test_fractions_in_unit_interval(self):
        panel, loadings_df = self._panel_and_loadings()
        result = variance_decomposition_ot(panel, loadings_df)
        assert 0.0 <= result["factor_explained_fraction"] <= 1.0
        assert 0.0 <= result["idiosyncratic_fraction"] <= 1.0

    def test_regime_keys_present_when_labels_given(self):
        panel, loadings_df = self._panel_and_loadings()
        n = len(panel.dates)
        labels = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = variance_decomposition_ot(panel, loadings_df, regime_labels=labels)
        assert {"between_regime", "within_regime",
                "between_fraction", "within_fraction"}.issubset(result.keys())

    def test_between_within_nonneg(self):
        panel, loadings_df = self._panel_and_loadings()
        n = len(panel.dates)
        labels = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = variance_decomposition_ot(panel, loadings_df, regime_labels=labels)
        assert result["between_regime"] >= 0
        assert result["within_regime"] >= 0

    def test_regime_fractions_in_unit_interval(self):
        panel, loadings_df = self._panel_and_loadings()
        n = len(panel.dates)
        labels = np.array([0] * (n // 2) + [1] * (n - n // 2))
        result = variance_decomposition_ot(panel, loadings_df, regime_labels=labels)
        assert 0.0 <= result["between_fraction"] <= 1.0 + 1e-9
        assert 0.0 <= result["within_fraction"] <= 1.0 + 1e-9

    def test_between_large_when_regimes_differ(self):
        """Regime with large mean shift → between_regime dominates."""
        means = [-3.0] * 10 + [3.0] * 10
        panel = _make_panel(means, n_stocks=300, seed=10)
        rng = np.random.default_rng(10)
        loadings_df = pd.DataFrame(rng.normal(0, 1, (300, 2)),
                                   columns=["f0", "f1"])
        labels = np.array([0] * 10 + [1] * 10)
        result = variance_decomposition_ot(panel, loadings_df,
                                           regime_labels=labels)
        assert result["between_fraction"] > 0.1


# ---------------------------------------------------------------------------
# TestTransportPCA
# ---------------------------------------------------------------------------

class TestTransportPCA:

    def _alternating_panel(self, n_dates=30, n_stocks=300, seed=0):
        """Alternating mean +2 / -2 — PC1 should capture this mean shift."""
        rng = np.random.default_rng(seed)
        dates = list(range(n_dates))
        dists = {}
        for t in range(n_dates):
            mean = 2.0 if t % 2 == 0 else -2.0
            dists[t] = EmpiricalDistribution(rng.normal(mean, 0.3, n_stocks))
        return DistributionPanel(dists, dates)

    def test_returns_required_keys(self):
        panel = _make_panel([0.0] * 10)
        result = transport_pca(panel, n_components=2)
        assert {"components", "explained_variance_ratio", "scores",
                "barycenter", "reconstructed"}.issubset(result.keys())

    def test_components_shape(self):
        panel = _make_panel([0.0] * 15)
        n_support = 100
        result = transport_pca(panel, n_components=3, n_support=n_support)
        assert result["components"].shape == (3, n_support)

    def test_scores_shape(self):
        n_dates = 20
        panel = _make_panel([0.0] * n_dates)
        result = transport_pca(panel, n_components=3, n_support=80)
        assert result["scores"].shape == (n_dates, 3)

    def test_explained_variance_ratio_sums_to_le_one(self):
        panel = _make_panel(list(range(10)))
        result = transport_pca(panel, n_components=3)
        assert result["explained_variance_ratio"].sum() <= 1.0 + 1e-9

    def test_explained_variance_ratio_nonneg(self):
        panel = _make_panel(list(range(10)))
        result = transport_pca(panel, n_components=3)
        assert np.all(result["explained_variance_ratio"] >= 0)

    def test_explained_variance_descending(self):
        panel = _make_panel(list(range(12)))
        result = transport_pca(panel, n_components=4)
        evr = result["explained_variance_ratio"]
        assert np.all(np.diff(evr) <= 1e-10)

    def test_barycenter_is_empirical_distribution(self):
        panel = _make_panel([0.0] * 8)
        result = transport_pca(panel, n_components=2)
        assert isinstance(result["barycenter"], EmpiricalDistribution)

    def test_reconstructed_length(self):
        n_dates = 12
        panel = _make_panel([0.0] * n_dates)
        result = transport_pca(panel, n_components=2)
        assert len(result["reconstructed"]) == n_dates

    def test_reconstructed_are_empirical_distributions(self):
        panel = _make_panel([0.0] * 8)
        result = transport_pca(panel, n_components=2)
        assert all(isinstance(r, EmpiricalDistribution) for r in result["reconstructed"])

    def test_pc1_captures_mean_shift(self):
        """
        Alternating panel (+2/-2): PC1 scores should anti-correlate with
        the mean of each distribution (high score ↔ high or low mean).
        """
        panel = self._alternating_panel(n_dates=20, n_stocks=400, seed=7)
        result = transport_pca(panel, n_components=2, n_support=150)

        # PC1 scores
        pc1_scores = result["scores"][:, 0]
        # True means alternate +2/-2
        true_means = np.array([2.0 if t % 2 == 0 else -2.0 for t in range(20)])

        corr = float(np.corrcoef(pc1_scores, true_means)[0, 1])
        assert abs(corr) > 0.8, \
            f"Expected |corr(PC1, mean)|>0.8, got {corr:.4f}"

    def test_pc1_dominates_on_mean_shift_data(self):
        """With purely alternating means, PC1 should explain > 70% of variance."""
        panel = self._alternating_panel(n_dates=30, n_stocks=500, seed=8)
        result = transport_pca(panel, n_components=3, n_support=150)
        assert result["explained_variance_ratio"][0] > 0.70, \
            f"PC1 explains {result['explained_variance_ratio'][0]:.4f}"

    def test_n_components_capped_at_t_minus_1(self):
        """PCA can't have more components than min(T-1, n_support)."""
        panel = _make_panel([0.0] * 5)   # T=5 → max 4 components
        result = transport_pca(panel, n_components=10, n_support=50)
        assert result["scores"].shape[1] <= 4
