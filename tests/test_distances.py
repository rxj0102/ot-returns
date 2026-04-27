"""
Tests for otreturns.distances.

Key theoretical results used:
  W_2(N(mu1,s1^2), N(mu2,s2^2))^2 = (mu1-mu2)^2 + (s1-s2)^2
  W_p(mu, mu) = 0
  W_p(mu, nu) = W_p(nu, mu)
  Triangle inequality: W(mu,nu) <= W(mu,pi) + W(pi,nu)
"""

import numpy as np
import pytest
from otreturns.distributions import EmpiricalDistribution, DistributionPanel
from otreturns.distances import (
    wasserstein_1d,
    wasserstein_nd,
    sinkhorn_distance,
    sliced_wasserstein,
    wasserstein_distance_matrix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gaussian_dist(mu, sigma, n, seed):
    rng = np.random.default_rng(seed)
    return EmpiricalDistribution(rng.normal(mu, sigma, n))


def gaussian_w2_exact(mu1, s1, mu2, s2):
    """Closed-form W_2 for Gaussians: sqrt((mu1-mu2)^2 + (s1-s2)^2)."""
    return np.sqrt((mu1 - mu2) ** 2 + (s1 - s2) ** 2)


# ---------------------------------------------------------------------------
# wasserstein_1d — exact Gaussian checks
# ---------------------------------------------------------------------------

class TestWasserstein1dGaussian:
    """
    For large n, empirical W_2 between Gaussian samples should approximate
    the theoretical W_2 for Gaussian distributions.
    """

    N = 50_000  # large enough that empirical ≈ theoretical

    def test_shift_only(self):
        # W_2(N(0,1), N(m,1)) = |m|
        for m in [0.5, 1.0, 2.0]:
            mu = gaussian_dist(0.0, 1.0, self.N, seed=0)
            nu = gaussian_dist(m,   1.0, self.N, seed=1)
            got = wasserstein_1d(mu, nu)
            expected = gaussian_w2_exact(0.0, 1.0, m, 1.0)
            np.testing.assert_allclose(got, expected, rtol=0.03,
                err_msg=f"shift m={m}")

    def test_scale_only(self):
        # W_2(N(0,1), N(0,s)) = |1-s|
        for s in [0.5, 1.5, 2.0]:
            mu = gaussian_dist(0.0, 1.0, self.N, seed=2)
            nu = gaussian_dist(0.0, s,   self.N, seed=3)
            got = wasserstein_1d(mu, nu)
            expected = gaussian_w2_exact(0.0, 1.0, 0.0, s)
            np.testing.assert_allclose(got, expected, rtol=0.03,
                err_msg=f"scale s={s}")

    def test_shift_and_scale(self):
        mu1, s1, mu2, s2 = 0.5, 1.2, -0.3, 0.8
        mu = gaussian_dist(mu1, s1, self.N, seed=4)
        nu = gaussian_dist(mu2, s2, self.N, seed=5)
        got = wasserstein_1d(mu, nu)
        expected = gaussian_w2_exact(mu1, s1, mu2, s2)
        np.testing.assert_allclose(got, expected, rtol=0.03)

    def test_p1_shift(self):
        # W_1(N(0,1), N(m,1)) ≈ m  (W_1 between shifted Gaussians equals the shift)
        m = 1.0
        mu = gaussian_dist(0.0, 1.0, self.N, seed=6)
        nu = gaussian_dist(m,   1.0, self.N, seed=7)
        got = wasserstein_1d(mu, nu, p=1)
        np.testing.assert_allclose(got, m, rtol=0.03)


class TestWasserstein1dProperties:
    N = 2000

    def test_identity(self):
        d = gaussian_dist(0.0, 1.0, self.N, seed=10)
        np.testing.assert_allclose(wasserstein_1d(d, d), 0.0, atol=1e-10)

    def test_symmetry(self):
        mu = gaussian_dist(0.0, 1.0, self.N, seed=11)
        nu = gaussian_dist(1.0, 1.5, self.N, seed=12)
        np.testing.assert_allclose(
            wasserstein_1d(mu, nu), wasserstein_1d(nu, mu), rtol=1e-10
        )

    def test_non_negative(self):
        mu = gaussian_dist(0.0, 1.0, self.N, seed=13)
        nu = gaussian_dist(2.0, 0.5, self.N, seed=14)
        assert wasserstein_1d(mu, nu) >= 0.0

    def test_triangle_inequality(self):
        rng = np.random.default_rng(15)
        mu  = EmpiricalDistribution(rng.normal(0.0, 1.0, self.N))
        nu  = EmpiricalDistribution(rng.normal(2.0, 1.0, self.N))
        pi_ = EmpiricalDistribution(rng.normal(1.0, 1.2, self.N))
        w_mn = wasserstein_1d(mu, nu)
        w_mp = wasserstein_1d(mu, pi_)
        w_pn = wasserstein_1d(pi_, nu)
        assert w_mn <= w_mp + w_pn + 1e-9

    def test_larger_distance_for_larger_shift(self):
        mu = gaussian_dist(0.0, 1.0, self.N, seed=20)
        nu_near = gaussian_dist(0.5, 1.0, self.N, seed=21)
        nu_far  = gaussian_dist(2.0, 1.0, self.N, seed=22)
        assert wasserstein_1d(mu, nu_near) < wasserstein_1d(mu, nu_far)

    def test_unequal_sample_sizes(self):
        rng = np.random.default_rng(23)
        mu = EmpiricalDistribution(rng.normal(0.0, 1.0, 300))
        nu = EmpiricalDistribution(rng.normal(1.0, 1.0, 700))
        w = wasserstein_1d(mu, nu)
        assert w > 0.0
        # Should be close to 1.0 (the shift) with moderate n
        np.testing.assert_allclose(w, 1.0, atol=0.1)

    def test_deterministic_result(self):
        mu = gaussian_dist(0.0, 1.0, 500, seed=30)
        nu = gaussian_dist(1.0, 1.0, 500, seed=31)
        assert wasserstein_1d(mu, nu) == wasserstein_1d(mu, nu)

    def test_p_order_1_vs_2(self):
        # W_1 <= W_2 for distributions supported on bounded interval
        # (by Hölder's inequality)
        rng = np.random.default_rng(32)
        mu = EmpiricalDistribution(rng.uniform(0, 1, 1000))
        nu = EmpiricalDistribution(rng.uniform(0, 1, 1000))
        assert wasserstein_1d(mu, nu, p=1) <= wasserstein_1d(mu, nu, p=2) + 1e-9

    def test_returns_float(self):
        mu = gaussian_dist(0.0, 1.0, 100, seed=40)
        nu = gaussian_dist(1.0, 1.0, 100, seed=41)
        result = wasserstein_1d(mu, nu)
        assert isinstance(result, float)

    def test_identical_samples(self):
        samples = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mu = EmpiricalDistribution(samples)
        nu = EmpiricalDistribution(samples.copy())
        np.testing.assert_allclose(wasserstein_1d(mu, nu), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# sinkhorn_distance
# ---------------------------------------------------------------------------

class TestSinkhornDistance:
    N = 500

    def test_returns_dict_with_required_keys(self):
        mu = gaussian_dist(0.0, 1.0, self.N, seed=50)
        nu = gaussian_dist(1.0, 1.0, self.N, seed=51)
        result = sinkhorn_distance(mu.samples, nu.samples)
        for k in ("distance", "transport_plan", "n_iterations",
                  "dual_potentials", "marginal_error"):
            assert k in result

    def test_identity_near_zero(self):
        mu = gaussian_dist(0.0, 1.0, self.N, seed=52)
        result = sinkhorn_distance(mu.samples, mu.samples, reg=0.001)
        assert result["distance"] < 0.05

    def test_non_negative(self):
        mu = gaussian_dist(0.0, 1.0, self.N, seed=53)
        nu = gaussian_dist(1.0, 1.0, self.N, seed=54)
        assert sinkhorn_distance(mu.samples, nu.samples)["distance"] >= 0.0

    def test_transport_plan_shape(self):
        n, m = 100, 120
        rng = np.random.default_rng(55)
        a = rng.normal(0, 1, n)
        b = rng.normal(1, 1, m)
        result = sinkhorn_distance(a, b)
        assert result["transport_plan"].shape == (n, m)

    def test_transport_plan_marginals(self):
        n = 100
        rng = np.random.default_rng(56)
        a_s = rng.normal(0, 1, n)
        b_s = rng.normal(1, 1, n)
        result = sinkhorn_distance(a_s, b_s, reg=0.05, tol=1e-10, max_iter=2000)
        gamma = result["transport_plan"]
        np.testing.assert_allclose(gamma.sum(axis=1), np.full(n, 1.0 / n), atol=1e-4)
        np.testing.assert_allclose(gamma.sum(axis=0), np.full(n, 1.0 / n), atol=1e-4)

    def test_converges_to_emd_as_reg_decreases(self):
        """Sinkhorn distance decreases toward W_2 as reg -> 0."""
        rng = np.random.default_rng(57)
        a_s = rng.normal(0.0, 1.0, 200)
        b_s = rng.normal(1.0, 1.0, 200)
        mu = EmpiricalDistribution(a_s)
        nu = EmpiricalDistribution(b_s)
        w2_exact = wasserstein_1d(mu, nu, p=2)

        distances = []
        for reg in [0.5, 0.1, 0.01]:
            d = sinkhorn_distance(a_s, b_s, reg=reg)["distance"]
            distances.append(d)

        # Should decrease as reg decreases
        assert distances[0] >= distances[1] - 1e-6
        assert distances[1] >= distances[2] - 1e-6
        # Small reg should be close to exact
        np.testing.assert_allclose(distances[2], w2_exact, rtol=0.15)

    def test_symmetry(self):
        # Sinkhorn is not analytically symmetric (iterative convergence),
        # but should be approximately so within a few percent.
        rng = np.random.default_rng(58)
        a_s = rng.normal(0.0, 1.0, 150)
        b_s = rng.normal(1.0, 1.5, 150)
        d_ab = sinkhorn_distance(a_s, b_s)["distance"]
        d_ba = sinkhorn_distance(b_s, a_s)["distance"]
        np.testing.assert_allclose(d_ab, d_ba, rtol=0.05)

    def test_custom_weights(self):
        rng = np.random.default_rng(59)
        n = 50
        a_s = rng.normal(0, 1, n)
        b_s = rng.normal(1, 1, n)
        w = np.ones(n) / n
        result = sinkhorn_distance(a_s, b_s, mu_weights=w, nu_weights=w)
        assert result["distance"] >= 0.0


# ---------------------------------------------------------------------------
# sliced_wasserstein
# ---------------------------------------------------------------------------

class TestSlicedWasserstein:
    N = 2000

    def test_1d_matches_exact(self):
        """In 1-d, sliced W_2 = exact W_2."""
        rng = np.random.default_rng(60)
        a = rng.normal(0.0, 1.0, self.N)
        b = rng.normal(1.0, 1.0, self.N)
        sw = sliced_wasserstein(a, b, p=2, seed=0)
        mu = EmpiricalDistribution(a)
        nu = EmpiricalDistribution(b)
        exact = wasserstein_1d(mu, nu, p=2)
        np.testing.assert_allclose(sw, exact, rtol=1e-6)

    def test_2d_gaussian_approximation(self):
        """2-d sliced W_2 should be in a reasonable range vs exact for Gaussians."""
        rng = np.random.default_rng(61)
        n = 3000
        a = rng.multivariate_normal([0, 0], np.eye(2), n)
        b = rng.multivariate_normal([1, 0], np.eye(2), n)
        sw = sliced_wasserstein(a, b, n_projections=200, seed=42)
        # Theoretical W_2 = 1.0; sliced W_2 is a lower bound, typically close
        assert 0.5 < sw < 1.5

    def test_identity_near_zero(self):
        rng = np.random.default_rng(62)
        a = rng.normal(0.0, 1.0, self.N)
        sw = sliced_wasserstein(a, a, seed=0)
        np.testing.assert_allclose(sw, 0.0, atol=1e-10)

    def test_non_negative(self):
        rng = np.random.default_rng(63)
        a = rng.normal(0, 1, 500)
        b = rng.normal(1, 1, 500)
        assert sliced_wasserstein(a, b, seed=0) >= 0.0

    def test_symmetry(self):
        rng = np.random.default_rng(64)
        a = rng.normal(0, 1, 500)
        b = rng.normal(1, 2, 500)
        np.testing.assert_allclose(
            sliced_wasserstein(a, b, seed=7),
            sliced_wasserstein(b, a, seed=7),
            rtol=1e-10,
        )

    def test_seed_reproducible(self):
        rng = np.random.default_rng(65)
        a = rng.multivariate_normal([0, 0], np.eye(2), 300)
        b = rng.multivariate_normal([1, 1], np.eye(2), 300)
        d1 = sliced_wasserstein(a, b, n_projections=50, seed=99)
        d2 = sliced_wasserstein(a, b, n_projections=50, seed=99)
        assert d1 == d2


# ---------------------------------------------------------------------------
# wasserstein_distance_matrix
# ---------------------------------------------------------------------------

class TestWassersteinDistanceMatrix:
    def _make_panel(self, n_dates=5, n_stocks=200, seed=0):
        rng = np.random.default_rng(seed)
        import pandas as pd
        dates = pd.bdate_range("2020-01-02", periods=n_dates)
        rows = []
        for d in dates:
            rets = rng.normal(0, 0.01, n_stocks)
            for i, r in enumerate(rets):
                rows.append({"date": d, "ticker": f"S{i}", "return": r})
        df = pd.DataFrame(rows)
        return DistributionPanel.from_panel(df, min_stocks=50)

    def test_shape(self):
        panel = self._make_panel(n_dates=5)
        D = wasserstein_distance_matrix(panel)
        assert D.shape == (5, 5)

    def test_zero_diagonal(self):
        panel = self._make_panel(n_dates=4)
        D = wasserstein_distance_matrix(panel)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)

    def test_symmetric(self):
        panel = self._make_panel(n_dates=4)
        D = wasserstein_distance_matrix(panel)
        np.testing.assert_allclose(D, D.T, atol=1e-12)

    def test_non_negative(self):
        panel = self._make_panel(n_dates=4)
        D = wasserstein_distance_matrix(panel)
        assert np.all(D >= 0)

    def test_date_subset(self):
        panel = self._make_panel(n_dates=6)
        subset = panel.dates[:3]
        D = wasserstein_distance_matrix(panel, dates=subset)
        assert D.shape == (3, 3)

    def test_list_input(self):
        rng = np.random.default_rng(70)
        dists = [EmpiricalDistribution(rng.normal(i, 1, 200)) for i in range(4)]
        D = wasserstein_distance_matrix(dists)
        assert D.shape == (4, 4)
        np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)
        np.testing.assert_allclose(D, D.T, atol=1e-12)

    def test_larger_shift_gives_larger_distance(self):
        rng = np.random.default_rng(71)
        d_near = EmpiricalDistribution(rng.normal(0.1, 1.0, 500))
        d_mid  = EmpiricalDistribution(rng.normal(0.0, 1.0, 500))
        d_far  = EmpiricalDistribution(rng.normal(2.0, 1.0, 500))
        D = wasserstein_distance_matrix([d_mid, d_near, d_far])
        assert D[0, 1] < D[0, 2]

    def test_triangle_inequality_matrix(self):
        rng = np.random.default_rng(72)
        dists = [EmpiricalDistribution(rng.normal(i * 0.5, 1, 1000)) for i in range(4)]
        D = wasserstein_distance_matrix(dists)
        T = len(dists)
        for i in range(T):
            for j in range(T):
                for k in range(T):
                    assert D[i, j] <= D[i, k] + D[k, j] + 1e-9, \
                        f"Triangle inequality violated for ({i},{j},{k})"


# ---------------------------------------------------------------------------
# wasserstein_nd
# ---------------------------------------------------------------------------

class TestWassersteinNd:
    N = 1000

    def test_emd_1d_matches_exact(self):
        rng = np.random.default_rng(80)
        a = rng.normal(0.0, 1.0, self.N)
        b = rng.normal(1.0, 1.0, self.N)
        mu = EmpiricalDistribution(a)
        nu = EmpiricalDistribution(b)
        exact = wasserstein_1d(mu, nu)
        nd = wasserstein_nd(a, b, method="emd")
        np.testing.assert_allclose(nd, exact, rtol=1e-4)

    def test_identity_emd(self):
        rng = np.random.default_rng(81)
        a = rng.normal(0, 1, 100)
        np.testing.assert_allclose(wasserstein_nd(a, a, method="emd"), 0.0, atol=1e-6)

    def test_sinkhorn_method(self):
        rng = np.random.default_rng(82)
        a = rng.normal(0.0, 1.0, 200)
        b = rng.normal(1.0, 1.0, 200)
        d = wasserstein_nd(a, b, method="sinkhorn", reg=0.01)
        assert d > 0.0

    def test_sliced_method(self):
        rng = np.random.default_rng(83)
        a = rng.normal(0.0, 1.0, 300)
        b = rng.normal(1.0, 1.0, 300)
        d = wasserstein_nd(a, b, method="sliced", seed=0)
        assert d > 0.0

    def test_invalid_method_raises(self):
        rng = np.random.default_rng(84)
        a = rng.normal(0, 1, 50)
        b = rng.normal(1, 1, 50)
        with pytest.raises(ValueError, match="Unknown method"):
            wasserstein_nd(a, b, method="bogus")
