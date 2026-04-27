"""
Tests for otreturns.transport.

Key theoretical results used:

  OT map between N(mu_s, sigma_s^2) and N(mu_t, sigma_t^2) is affine:
      T(x) = mu_t + (sigma_t / sigma_s) * (x - mu_s)
           = (sigma_t/sigma_s) * x + (mu_t - sigma_t*mu_s/sigma_s)

  Special cases:
      Pure shift (sigma_t = sigma_s): T(x) = x + (mu_t - mu_s)  =>  d(x) = constant
      Pure scale (mu_t = mu_s = 0):   T(x) = (sigma_t/sigma_s) * x  =>  d(x) = (sigma_t/sigma_s - 1)*x

  For Gaussian -> Gaussian: shape component of cost decomposition == 0.
"""

import numpy as np
import pytest
import pandas as pd

from otreturns.distributions import EmpiricalDistribution, DistributionPanel
from otreturns.transport import (
    optimal_transport_map_1d,
    optimal_transport_plan,
    transport_map_displacement,
    barycentric_projection,
    transport_cost_decomposition,
    TransportPath,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gaussian(mu, sigma, n, seed):
    rng = np.random.default_rng(seed)
    return EmpiricalDistribution(rng.normal(mu, sigma, n))


def make_panel(n_dates=10, n_stocks=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    rows = []
    for d in dates:
        ret = rng.normal(0, 0.01, n_stocks)
        for i, r in enumerate(ret):
            rows.append({"date": d, "ticker": f"S{i}", "return": r})
    df = pd.DataFrame(rows)
    return DistributionPanel.from_panel(df, min_stocks=50)


# ---------------------------------------------------------------------------
# optimal_transport_map_1d — basic properties
# ---------------------------------------------------------------------------

class TestOTMap1dProperties:
    N = 5000

    def test_returns_callable(self):
        mu = gaussian(0, 1, self.N, 0)
        nu = gaussian(1, 1, self.N, 1)
        T = optimal_transport_map_1d(mu, nu)
        assert callable(T)

    def test_identity_map(self):
        mu = gaussian(0, 1, self.N, 2)
        T = optimal_transport_map_1d(mu, mu)
        x = np.linspace(-2, 2, 100)
        np.testing.assert_allclose(T(x), x, atol=0.05)

    def test_map_is_monotone(self):
        mu = gaussian(0.0, 1.0, self.N, 3)
        nu = gaussian(1.0, 1.5, self.N, 4)
        T = optimal_transport_map_1d(mu, nu)
        x = np.linspace(-3, 3, 200)
        Tx = T(x)
        assert np.all(np.diff(Tx) >= -1e-10), "T must be non-decreasing"

    def test_vectorised(self):
        mu = gaussian(0, 1, self.N, 5)
        nu = gaussian(1, 1, self.N, 6)
        T = optimal_transport_map_1d(mu, nu)
        x_arr = np.array([-1.0, 0.0, 1.0])
        result = T(x_arr)
        assert result.shape == (3,)

    def test_scalar_input(self):
        mu = gaussian(0, 1, self.N, 7)
        nu = gaussian(1, 1, self.N, 8)
        T = optimal_transport_map_1d(mu, nu)
        result = T(np.array([0.0]))
        assert result.shape == (1,)


# ---------------------------------------------------------------------------
# optimal_transport_map_1d — affine Gaussian maps
# ---------------------------------------------------------------------------

class TestOTMap1dGaussian:
    """
    For N(mu_s, sigma_s^2) -> N(mu_t, sigma_t^2) the OT map is:
        T(x) = mu_t + (sigma_t / sigma_s) * (x - mu_s)
    """
    N = 30_000

    def _affine_reference(self, mu_s, sigma_s, mu_t, sigma_t, x):
        return mu_t + (sigma_t / sigma_s) * (x - mu_s)

    def test_pure_shift(self):
        shift = 1.5
        mu = gaussian(0.0, 1.0, self.N, seed=10)
        nu = gaussian(shift, 1.0, self.N, seed=11)
        T = optimal_transport_map_1d(mu, nu)
        x = np.linspace(-2, 2, 50)
        expected = self._affine_reference(0.0, 1.0, shift, 1.0, x)
        np.testing.assert_allclose(T(x), expected, atol=0.05)

    def test_pure_scale_expand(self):
        sigma_t = 2.0
        mu = gaussian(0.0, 1.0, self.N, seed=12)
        nu = gaussian(0.0, sigma_t, self.N, seed=13)
        T = optimal_transport_map_1d(mu, nu)
        x = np.linspace(-2, 2, 50)
        expected = self._affine_reference(0.0, 1.0, 0.0, sigma_t, x)
        np.testing.assert_allclose(T(x), expected, atol=0.1)

    def test_pure_scale_contract(self):
        sigma_t = 0.5
        mu = gaussian(0.0, 1.0, self.N, seed=14)
        nu = gaussian(0.0, sigma_t, self.N, seed=15)
        T = optimal_transport_map_1d(mu, nu)
        x = np.linspace(-2, 2, 50)
        expected = self._affine_reference(0.0, 1.0, 0.0, sigma_t, x)
        np.testing.assert_allclose(T(x), expected, atol=0.1)

    def test_shift_and_scale(self):
        mu_s, sigma_s, mu_t, sigma_t = 0.5, 1.2, -0.3, 0.8
        mu = gaussian(mu_s, sigma_s, self.N, seed=16)
        nu = gaussian(mu_t, sigma_t, self.N, seed=17)
        T = optimal_transport_map_1d(mu, nu)
        x = np.linspace(-2, 3, 50)
        expected = self._affine_reference(mu_s, sigma_s, mu_t, sigma_t, x)
        np.testing.assert_allclose(T(x), expected, atol=0.1)

    def test_pushforward_approximates_target(self):
        """T#mu should approximate nu: mean and std of T(samples) ≈ target moments."""
        mu_s, sigma_s, mu_t, sigma_t = 0.0, 1.0, 1.0, 1.5
        source = gaussian(mu_s, sigma_s, self.N, seed=18)
        target = gaussian(mu_t, sigma_t, self.N, seed=19)
        T = optimal_transport_map_1d(source, target)

        pushed = T(source.samples)
        pushed_dist = EmpiricalDistribution(pushed)
        m = pushed_dist.moments(order=2)

        np.testing.assert_allclose(m["mean"], mu_t, atol=0.05)
        np.testing.assert_allclose(np.sqrt(m["variance"]), sigma_t, rtol=0.05)


# ---------------------------------------------------------------------------
# transport_map_displacement — shape of displacement fields
# ---------------------------------------------------------------------------

class TestDisplacementField:
    N = 30_000

    def test_returns_two_arrays(self):
        mu = gaussian(0, 1, self.N, 20)
        nu = gaussian(1, 1, self.N, 21)
        result = transport_map_displacement(mu, nu)
        assert len(result) == 2
        x_grid, disp = result
        assert x_grid.shape == disp.shape

    def test_pure_shift_constant_displacement(self):
        """d(x) = shift everywhere for a pure location shift."""
        shift = 1.0
        mu = gaussian(0.0, 1.0, self.N, seed=22)
        nu = gaussian(shift, 1.0, self.N, seed=23)
        _, disp = transport_map_displacement(mu, nu, n_quantiles=500)
        # Displacement should be approximately constant = shift
        np.testing.assert_allclose(disp, shift, atol=0.08)

    def test_pure_shift_std_of_displacement_near_zero(self):
        """For pure shift, all quantiles move by the same amount => std(d) ≈ 0."""
        shift = 0.5
        mu = gaussian(0.0, 1.0, self.N, seed=24)
        nu = gaussian(shift, 1.0, self.N, seed=25)
        _, disp = transport_map_displacement(mu, nu, n_quantiles=500)
        assert np.std(disp) < 0.06

    def test_pure_scale_displacement_linear(self):
        """d(x) = (sigma_t - 1)*x for pure scale change centred at 0."""
        sigma_t = 2.0
        mu = gaussian(0.0, 1.0, self.N, seed=26)
        nu = gaussian(0.0, sigma_t, self.N, seed=27)
        x_grid, disp = transport_map_displacement(mu, nu, n_quantiles=500)

        # Expected displacement: T(x) - x = sigma_t*x - x = (sigma_t-1)*x
        # Exclude the outermost 2 % of quantiles where empirical tail
        # estimation variance is high (n=30k => ~1 % quantile has std ~ 0.1).
        interior = slice(10, 490)
        expected = (sigma_t - 1.0) * x_grid[interior]
        np.testing.assert_allclose(disp[interior], expected, atol=0.15)

    def test_scale_contraction_negative_tails(self):
        """Contracting scale: displacement is negative for large |x|."""
        mu = gaussian(0.0, 1.0, self.N, seed=28)
        nu = gaussian(0.0, 0.5, self.N, seed=29)
        x_grid, disp = transport_map_displacement(mu, nu, n_quantiles=500)
        tail_mask = np.abs(x_grid) > 1.0
        # Extreme quantiles should move inward (toward 0)
        assert np.mean(disp[x_grid > 1.0]) < 0.0
        assert np.mean(disp[x_grid < -1.0]) > 0.0

    def test_identity_zero_displacement(self):
        mu = gaussian(0.0, 1.0, self.N, seed=30)
        _, disp = transport_map_displacement(mu, mu, n_quantiles=500)
        np.testing.assert_allclose(disp, 0.0, atol=0.05)

    def test_displacement_length(self):
        mu = gaussian(0, 1, 200, 31)
        nu = gaussian(1, 2, 200, 32)
        x_grid, disp = transport_map_displacement(mu, nu, n_quantiles=300)
        assert len(x_grid) == 300
        assert len(disp) == 300


# ---------------------------------------------------------------------------
# optimal_transport_plan — marginal constraints
# ---------------------------------------------------------------------------

class TestOTPlan:
    N = 200

    def test_shape(self):
        rng = np.random.default_rng(40)
        a = rng.normal(0, 1, self.N)
        b = rng.normal(1, 1, self.N)
        gamma = optimal_transport_plan(a, b)
        assert gamma.shape == (self.N, self.N)

    def test_non_negative(self):
        rng = np.random.default_rng(41)
        a = rng.normal(0, 1, self.N)
        b = rng.normal(1, 1, self.N)
        gamma = optimal_transport_plan(a, b)
        assert np.all(gamma >= -1e-10)

    def test_source_marginal(self):
        """Row sums of gamma must equal source weights."""
        n, m = 100, 150
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, n)
        b = rng.normal(1, 1, m)
        gamma = optimal_transport_plan(a, b)
        np.testing.assert_allclose(
            gamma.sum(axis=1), np.full(n, 1.0 / n), atol=1e-6
        )

    def test_target_marginal(self):
        """Column sums of gamma must equal target weights."""
        n, m = 150, 100
        rng = np.random.default_rng(43)
        a = rng.normal(0, 1, n)
        b = rng.normal(1, 1, m)
        gamma = optimal_transport_plan(a, b)
        np.testing.assert_allclose(
            gamma.sum(axis=0), np.full(m, 1.0 / m), atol=1e-6
        )

    def test_custom_weights_marginals(self):
        n, m = 80, 80
        rng = np.random.default_rng(44)
        a = rng.normal(0, 1, n)
        b = rng.normal(1, 1, m)
        w_a = rng.dirichlet(np.ones(n))
        w_b = rng.dirichlet(np.ones(m))
        gamma = optimal_transport_plan(a, b, source_weights=w_a, target_weights=w_b)
        np.testing.assert_allclose(gamma.sum(axis=1), w_a, atol=1e-6)
        np.testing.assert_allclose(gamma.sum(axis=0), w_b, atol=1e-6)

    def test_total_mass_one(self):
        rng = np.random.default_rng(45)
        a = rng.normal(0, 1, self.N)
        b = rng.normal(1, 1, self.N)
        gamma = optimal_transport_plan(a, b)
        np.testing.assert_allclose(gamma.sum(), 1.0, atol=1e-6)

    def test_sinkhorn_marginals_approx(self):
        n = 100
        rng = np.random.default_rng(46)
        a = rng.normal(0, 1, n)
        b = rng.normal(1, 1, n)
        gamma = optimal_transport_plan(a, b, reg=0.05)
        np.testing.assert_allclose(gamma.sum(axis=1), np.full(n, 1.0/n), atol=1e-3)
        np.testing.assert_allclose(gamma.sum(axis=0), np.full(n, 1.0/n), atol=1e-3)

    def test_unequal_sizes(self):
        rng = np.random.default_rng(47)
        a = rng.normal(0, 1, 60)
        b = rng.normal(1, 1, 90)
        gamma = optimal_transport_plan(a, b)
        assert gamma.shape == (60, 90)
        np.testing.assert_allclose(gamma.sum(axis=1), np.full(60, 1/60), atol=1e-6)
        np.testing.assert_allclose(gamma.sum(axis=0), np.full(90, 1/90), atol=1e-6)


# ---------------------------------------------------------------------------
# barycentric_projection
# ---------------------------------------------------------------------------

class TestBarycentricProjection:
    def test_identity_plan(self):
        """With diagonal plan, barycentric projection == target samples."""
        n = 50
        rng = np.random.default_rng(50)
        src = rng.normal(0, 1, n)
        tgt = rng.normal(1, 1, n)
        mu = EmpiricalDistribution(src)
        # Diagonal coupling: each source atom maps to corresponding target atom
        gamma = np.diag(np.full(n, 1.0 / n))
        proj = barycentric_projection(mu, gamma, tgt)
        np.testing.assert_allclose(proj, tgt, atol=1e-10)

    def test_uniform_plan_gives_mean(self):
        """Uniform coupling maps every source atom to the target mean."""
        n = 50
        rng = np.random.default_rng(51)
        src = rng.normal(0, 1, n)
        tgt = rng.normal(2, 0.5, n)
        mu = EmpiricalDistribution(src)
        gamma = np.full((n, n), 1.0 / (n * n))
        proj = barycentric_projection(mu, gamma, tgt)
        np.testing.assert_allclose(proj, tgt.mean(), atol=1e-10)

    def test_output_shape(self):
        n, m = 40, 60
        rng = np.random.default_rng(52)
        src = rng.normal(0, 1, n)
        tgt = rng.normal(1, 1, m)
        mu = EmpiricalDistribution(src)
        gamma = optimal_transport_plan(src, tgt)
        proj = barycentric_projection(mu, gamma, tgt)
        assert proj.shape == (n,)

    def test_projection_within_target_range(self):
        n, m = 100, 100
        rng = np.random.default_rng(53)
        src = rng.normal(0, 1, n)
        tgt = rng.normal(1, 1, m)
        mu = EmpiricalDistribution(src)
        gamma = optimal_transport_plan(src, tgt)
        proj = barycentric_projection(mu, gamma, tgt)
        assert proj.min() >= tgt.min() - 1e-10
        assert proj.max() <= tgt.max() + 1e-10


# ---------------------------------------------------------------------------
# transport_cost_decomposition
# ---------------------------------------------------------------------------

class TestCostDecomposition:
    N = 30_000

    def test_returns_required_keys(self):
        mu = gaussian(0, 1, 500, 60)
        nu = gaussian(1, 1, 500, 61)
        d = transport_cost_decomposition(mu, nu)
        for k in ("total", "location", "scale", "shape",
                  "location_fraction", "scale_fraction", "shape_fraction"):
            assert k in d

    def test_fractions_sum_to_one(self):
        mu = gaussian(0, 1, 1000, 62)
        nu = gaussian(1, 1.5, 1000, 63)
        d = transport_cost_decomposition(mu, nu)
        np.testing.assert_allclose(
            d["location_fraction"] + d["scale_fraction"] + d["shape_fraction"],
            1.0, atol=1e-6,
        )

    def test_gaussian_shape_near_zero(self):
        """For Gaussians W_2^2 = location + scale => shape ~ 0."""
        mu = gaussian(0.0, 1.0, self.N, seed=64)
        nu = gaussian(1.0, 1.5, self.N, seed=65)
        d = transport_cost_decomposition(mu, nu)
        # Shape fraction should be small (< 5 %) for Gaussians
        assert d["shape_fraction"] < 0.05

    def test_pure_shift_location_dominates(self):
        shift = 2.0
        mu = gaussian(0.0, 1.0, self.N, seed=66)
        nu = gaussian(shift, 1.0, self.N, seed=67)
        d = transport_cost_decomposition(mu, nu)
        assert d["location_fraction"] > 0.85

    def test_pure_scale_scale_dominates(self):
        mu = gaussian(0.0, 1.0, self.N, seed=68)
        nu = gaussian(0.0, 3.0, self.N, seed=69)
        d = transport_cost_decomposition(mu, nu)
        assert d["scale_fraction"] > 0.85

    def test_non_gaussian_has_shape_component(self):
        """Student-t vs Normal should produce non-negligible shape fraction."""
        rng = np.random.default_rng(70)
        normal_samples = rng.normal(0.0, 1.0, self.N)
        t_samples = rng.standard_t(df=3, size=self.N)
        # Normalise t to unit std
        t_samples = t_samples / np.sqrt(3.0 / (3.0 - 2.0))
        mu = EmpiricalDistribution(normal_samples)
        nu = EmpiricalDistribution(t_samples)
        d = transport_cost_decomposition(mu, nu)
        # Shape fraction should be non-negligible
        assert d["shape_fraction"] > 0.05

    def test_total_equals_w2_squared(self):
        from otreturns.distances import wasserstein_1d
        mu = gaussian(0.5, 1.2, self.N, seed=71)
        nu = gaussian(-0.3, 0.8, self.N, seed=72)
        d = transport_cost_decomposition(mu, nu)
        w2 = wasserstein_1d(mu, nu, p=2)
        np.testing.assert_allclose(d["total"], w2 ** 2, rtol=1e-6)

    def test_components_non_negative(self):
        mu = gaussian(0, 1, 1000, 73)
        nu = gaussian(1, 2, 1000, 74)
        d = transport_cost_decomposition(mu, nu)
        assert d["location"] >= 0
        assert d["scale"] >= 0
        assert d["shape"] >= 0

    def test_identity_zero_cost(self):
        mu = gaussian(0, 1, 500, 75)
        d = transport_cost_decomposition(mu, mu)
        np.testing.assert_allclose(d["total"], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# TransportPath
# ---------------------------------------------------------------------------

class TestTransportPath:
    def test_requires_compute_maps(self):
        panel = make_panel(n_dates=4)
        tp = TransportPath(panel)
        with pytest.raises(RuntimeError, match="compute_maps"):
            tp.displacement_time_series()

    def test_compute_maps_runs(self):
        panel = make_panel(n_dates=5, seed=80)
        tp = TransportPath(panel)
        tp.compute_maps()
        assert tp._computed

    def test_maps_count(self):
        n = 6
        panel = make_panel(n_dates=n, seed=81)
        tp = TransportPath(panel).compute_maps()
        assert len(tp._maps) == n - 1

    def test_maps_are_callable(self):
        panel = make_panel(n_dates=4, seed=82)
        tp = TransportPath(panel).compute_maps()
        for T in tp._maps:
            assert callable(T)

    def test_displacement_time_series_length(self):
        n = 7
        panel = make_panel(n_dates=n, seed=83)
        tp = TransportPath(panel).compute_maps()
        ts = tp.displacement_time_series(quantile=0.5)
        assert len(ts) == n - 1

    def test_displacement_time_series_quantile_bounds(self):
        panel = make_panel(n_dates=5, seed=84)
        tp = TransportPath(panel).compute_maps()
        with pytest.raises(ValueError):
            tp.displacement_time_series(quantile=0.0)
        with pytest.raises(ValueError):
            tp.displacement_time_series(quantile=1.0)

    def test_cumulative_displacement_shape(self):
        panel = make_panel(n_dates=6, seed=85)
        tp = TransportPath(panel).compute_maps()
        start = tp.dates[0]
        end = tp.dates[4]
        x0, cum_disp = tp.cumulative_displacement(start, end)
        assert x0.shape == cum_disp.shape

    def test_cumulative_displacement_same_date_raises(self):
        panel = make_panel(n_dates=5, seed=86)
        tp = TransportPath(panel).compute_maps()
        d = tp.dates[2]
        with pytest.raises(ValueError):
            tp.cumulative_displacement(d, d)

    def test_cumulative_zero_for_trivial_panel(self):
        """Panel of identical distributions => cumulative displacement ~ 0."""
        rng = np.random.default_rng(87)
        samples = rng.normal(0, 1, 300)
        dates = pd.bdate_range("2020-01-02", periods=5)
        dists = {d: EmpiricalDistribution(samples.copy()) for d in dates}
        panel = DistributionPanel(dists, list(dates))
        tp = TransportPath(panel).compute_maps()
        _, cum_disp = tp.cumulative_displacement(dates[0], dates[4])
        np.testing.assert_allclose(cum_disp, 0.0, atol=0.05)
