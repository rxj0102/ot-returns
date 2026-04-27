"""
Optimal transport maps and plans.

In 1-d the OT plan is fully determined by the monotone rearrangement (Brenier):
    T = F_nu^{-1} o F_mu

i.e. map x to F_nu^{-1}(F_mu(x)).  This is the unique W_2-optimal map.

For discrete/multi-dimensional distributions we use POT's exact EMD solver
(network simplex) or the regularised Sinkhorn algorithm.
"""

from __future__ import annotations

import numpy as np
import ot
from scipy.interpolate import interp1d

from otreturns.distributions import EmpiricalDistribution, DistributionPanel
from otreturns.distances import wasserstein_1d


# ---------------------------------------------------------------------------
# 1-d optimal transport map
# ---------------------------------------------------------------------------

def optimal_transport_map_1d(
    source: EmpiricalDistribution,
    target: EmpiricalDistribution,
    n_quantiles: int = 1000,
) -> callable:
    """
    Compute the optimal transport map T: R -> R such that T#mu = nu.

    For 1-d distributions the OT map is the monotone rearrangement:
        T(x) = F_nu^{-1}(F_mu(x))

    This is the unique W_2-optimal map (Brenier's theorem in 1-d).

    The map is built by evaluating the composition on a fine quantile grid
    and returning a callable backed by linear interpolation.

    Args:
        source:      source distribution mu
        target:      target distribution nu
        n_quantiles: grid resolution for the interpolant

    Returns:
        T: callable that maps R -> R (vectorised over numpy arrays)
    """
    u = np.linspace(0.0, 1.0, n_quantiles + 2)[1:-1]   # interior of (0,1)

    x_grid = source.quantile_function(u)   # source support points
    Tx_grid = target.quantile_function(u)  # target support points T(x)

    # Build interpolant; clamp outside training range to boundary values
    T_interp = interp1d(
        x_grid, Tx_grid,
        kind="linear",
        bounds_error=False,
        fill_value=(Tx_grid[0], Tx_grid[-1]),
    )

    def T(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return T_interp(x).astype(float)

    return T


# ---------------------------------------------------------------------------
# Discrete transport plan (EMD or Sinkhorn)
# ---------------------------------------------------------------------------

def optimal_transport_plan(
    source: np.ndarray,
    target: np.ndarray,
    source_weights: np.ndarray = None,
    target_weights: np.ndarray = None,
    reg: float = None,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> np.ndarray:
    """
    Compute the optimal transport plan gamma in Pi(mu, nu).

    If reg is None: exact EMD via network simplex (POT).
    If reg > 0:     Sinkhorn-regularised plan.

    gamma satisfies:
        gamma_{ij} >= 0
        sum_j gamma_{ij} = source_weights[i]   (source marginal)
        sum_i gamma_{ij} = target_weights[j]   (target marginal)

    Args:
        source:         (n,) or (n, d) source samples
        target:         (m,) or (m, d) target samples
        source_weights: length-n weights (default uniform 1/n)
        target_weights: length-m weights (default uniform 1/m)
        reg:            Sinkhorn regularisation (None => exact EMD)
        max_iter:       max Sinkhorn iterations (unused for EMD)
        tol:            Sinkhorn convergence tolerance (unused for EMD)

    Returns:
        gamma: (n, m) coupling matrix
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.ndim == 1:
        source = source[:, None]
    if target.ndim == 1:
        target = target[:, None]

    n = source.shape[0]
    m = target.shape[0]

    a = np.full(n, 1.0 / n) if source_weights is None else _normalize(source_weights)
    b = np.full(m, 1.0 / m) if target_weights is None else _normalize(target_weights)

    M = ot.dist(source, target, metric="sqeuclidean")

    if reg is None:
        gamma = ot.emd(a, b, M)
    else:
        if reg <= 0:
            raise ValueError("reg must be positive")
        gamma = ot.sinkhorn(a, b, M, reg=reg, numItermax=max_iter, stopThr=tol)

    return np.asarray(gamma, dtype=float)


# ---------------------------------------------------------------------------
# Displacement field
# ---------------------------------------------------------------------------

def transport_map_displacement(
    source: EmpiricalDistribution,
    target: EmpiricalDistribution,
    n_quantiles: int = 1000,
) -> tuple:
    """
    Compute the displacement field d(x) = T(x) - x.

    This shows how each quantile of the distribution moves:
        d > 0 everywhere  => location shift (mean return increase)
        d > 0 at low x, d < 0 at high x => compression (vol decrease)
        d > 0 at extremes => tail fattening

    Args:
        source:      source distribution
        target:      target distribution
        n_quantiles: quantile grid resolution

    Returns:
        (x_grid, displacement): source support points and d(x) = T(x) - x
    """
    u = np.linspace(0.0, 1.0, n_quantiles + 2)[1:-1]
    x_grid = source.quantile_function(u)
    Tx_grid = target.quantile_function(u)
    displacement = Tx_grid - x_grid
    return x_grid, displacement


# ---------------------------------------------------------------------------
# Barycentric projection
# ---------------------------------------------------------------------------

def barycentric_projection(
    source: EmpiricalDistribution,
    gamma: np.ndarray,
    target_samples: np.ndarray,
) -> np.ndarray:
    """
    Barycentric projection: T_bar(x_i) = sum_j gamma_{ij} * y_j / a_i.

    Given a coupling matrix gamma and target support points y_j, the
    barycentric projection gives the "expected" target location for each
    source atom, weighted by the plan.

    Args:
        source:         source distribution (n atoms)
        gamma:          (n, m) coupling matrix
        target_samples: (m,) target support points y_j

    Returns:
        T_bar: (n,) projected target positions for each source atom
    """
    target_samples = np.asarray(target_samples, dtype=float)
    row_sums = gamma.sum(axis=1)                    # = source.weights
    # Avoid division by zero for zero-weight atoms
    safe_sums = np.where(row_sums > 0, row_sums, 1.0)
    return (gamma @ target_samples) / safe_sums


# ---------------------------------------------------------------------------
# Transport cost decomposition
# ---------------------------------------------------------------------------

def transport_cost_decomposition(
    source: EmpiricalDistribution,
    target: EmpiricalDistribution,
) -> dict:
    """
    Decompose W_2^2(mu, nu) into location, scale, and shape components.

    For Gaussian distributions:
        W_2^2 = (mean_mu - mean_nu)^2 + (std_mu - std_nu)^2

    For arbitrary distributions we decompose as:
        total     = W_2^2(mu, nu)
        location  = (mean_mu - mean_nu)^2
        scale     = (std_mu - std_nu)^2
        shape     = total - location - scale   [residual: skewness, kurtosis, tails]

    The shape component measures the part of the transport cost that cannot
    be explained by a shift and rescaling alone.  For Gaussians it is zero.

    Returns:
        dict with keys:
            total, location, scale, shape,
            location_fraction, scale_fraction, shape_fraction
    """
    s_stats = source.moments(order=2)
    t_stats = target.moments(order=2)

    mean_s, var_s = s_stats["mean"], s_stats["variance"]
    mean_t, var_t = t_stats["mean"], t_stats["variance"]
    std_s = float(np.sqrt(var_s))
    std_t = float(np.sqrt(var_t))

    total = wasserstein_1d(source, target, p=2) ** 2

    location = (mean_s - mean_t) ** 2
    scale = (std_s - std_t) ** 2
    shape = max(0.0, total - location - scale)   # numerical noise guard

    safe_total = total if total > 1e-30 else 1.0
    return {
        "total":              total,
        "location":           location,
        "scale":              scale,
        "shape":              shape,
        "location_fraction":  location / safe_total,
        "scale_fraction":     scale    / safe_total,
        "shape_fraction":     shape    / safe_total,
    }


# ---------------------------------------------------------------------------
# TransportPath
# ---------------------------------------------------------------------------

class TransportPath:
    """
    Sequence of consecutive optimal transport maps over a DistributionPanel.

    Given distributions mu_1, mu_2, ..., mu_T the path stores:
        T_{t -> t+1}: optimal map from mu_t to mu_{t+1}
        d_t = T_{t->t+1} - Id: displacement field at each step

    This represents how the cross-sectional return distribution evolves.
    """

    def __init__(
        self,
        panel: DistributionPanel,
        dates: list = None,
        n_quantiles: int = 500,
    ):
        self.panel = panel
        self.dates = dates if dates is not None else panel.dates
        self.n_quantiles = n_quantiles

        # Populated by compute_maps()
        self._maps: list[callable] = []        # T_{t->t+1}
        self._x_grids: list[np.ndarray] = []   # source quantile grids
        self._displacements: list[np.ndarray] = []
        self._computed = False

    def compute_maps(self) -> "TransportPath":
        """Compute all consecutive transport maps T_{t->t+1}."""
        self._maps = []
        self._x_grids = []
        self._displacements = []

        for t in range(len(self.dates) - 1):
            mu_t = self.panel[self.dates[t]]
            mu_t1 = self.panel[self.dates[t + 1]]

            T = optimal_transport_map_1d(mu_t, mu_t1, self.n_quantiles)
            x_grid, disp = transport_map_displacement(mu_t, mu_t1, self.n_quantiles)

            self._maps.append(T)
            self._x_grids.append(x_grid)
            self._displacements.append(disp)

        self._computed = True
        return self

    def _require_computed(self):
        if not self._computed:
            raise RuntimeError("Call compute_maps() first.")

    def cumulative_displacement(
        self, start_date, end_date
    ) -> tuple:
        """
        Total cumulative displacement from start_date to end_date,
        evaluated on the start_date quantile grid.

        Computes T_{end->start}^{-1} ... T_{start->start+1} by
        composing maps through sequential interpolation.

        Returns:
            (x_grid, cumulative_disp): both of shape (n_quantiles,)
        """
        self._require_computed()
        i_start = self.dates.index(start_date)
        i_end = self.dates.index(end_date)
        if i_end <= i_start:
            raise ValueError("end_date must be strictly after start_date")

        u = np.linspace(0.0, 1.0, self.n_quantiles + 2)[1:-1]
        x0 = self.panel[start_date].quantile_function(u)
        x_current = x0.copy()

        for t in range(i_start, i_end):
            x_current = self._maps[t](x_current)

        return x0, x_current - x0

    def displacement_time_series(self, quantile: float = 0.5) -> np.ndarray:
        """
        Track the displacement of a specific quantile over time.

        Args:
            quantile: probability level in (0, 1)

        Returns:
            displacements: (T-1,) array of d_t(F_mu_t^{-1}(quantile))
        """
        self._require_computed()
        if not (0.0 < quantile < 1.0):
            raise ValueError("quantile must be in (0, 1)")

        result = np.empty(len(self.dates) - 1)
        for t, (x_grid, disp) in enumerate(zip(self._x_grids, self._displacements)):
            # Interpolate displacement at the requested quantile level
            mu_t = self.panel[self.dates[t]]
            x_q = float(mu_t.quantile_function(quantile))
            result[t] = float(np.interp(x_q, x_grid, disp))

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    s = w.sum()
    if s <= 0:
        raise ValueError("weights must sum to a positive value")
    return w / s
