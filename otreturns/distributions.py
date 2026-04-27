"""
Empirical distribution representations for cross-sectional equity returns.

At each date t, the cross-section of stock returns {r_{1,t}, ..., r_{N_t,t}}
forms a discrete probability measure:
    mu_t = (1/N_t) sum_{i=1}^{N_t} delta_{r_{i,t}}

This module provides efficient representations for working with these
distributions via optimal transport.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


class EmpiricalDistribution:
    """
    Represents an empirical probability distribution from a sample of observations.

    In our context, each distribution is the cross-section of equity returns
    at a single date: {r_{1,t}, r_{2,t}, ..., r_{N_t,t}} where r_{i,t} is
    the return of stock i at date t.

    Stores:
    - samples: raw observations
    - weights: probability weights (default: uniform 1/n)
    - Support for both discrete (point mass) and histogram representations

    The discrete representation is:
        mu = sum_{i=1}^{n} w_i * delta_{x_i}
    where delta_x is the Dirac mass at x.
    """

    def __init__(self, samples: np.ndarray, weights: np.ndarray = None):
        samples = np.asarray(samples, dtype=float)
        if samples.ndim != 1:
            raise ValueError("samples must be 1-dimensional")
        if len(samples) == 0:
            raise ValueError("samples must be non-empty")

        if weights is None:
            weights = np.ones(len(samples)) / len(samples)
        else:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != samples.shape:
                raise ValueError("weights must have the same shape as samples")
            if np.any(weights < 0):
                raise ValueError("weights must be non-negative")
            total = weights.sum()
            if total <= 0:
                raise ValueError("weights must sum to a positive value")
            weights = weights / total

        self.samples = samples
        self.weights = weights
        self.n = len(samples)

        sort_idx = np.argsort(self.samples)
        self._sorted_samples = self.samples[sort_idx]
        self._sorted_weights = self.weights[sort_idx]
        self._cumulative_weights = np.cumsum(self._sorted_weights)

    def to_histogram(self, n_bins: int = 100, range: tuple = None) -> tuple:
        """
        Convert to histogram representation (bin_centers, bin_weights).
        Useful for Sinkhorn computation on a fixed grid.
        """
        if range is None:
            range = (float(self._sorted_samples[0]), float(self._sorted_samples[-1]))

        counts, edges = np.histogram(
            self.samples, bins=n_bins, range=range,
            weights=self.weights, density=False,
        )
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        total = counts.sum()
        bin_weights = counts / total if total > 0 else counts
        return bin_centers, bin_weights

    def quantile_function(self, p: np.ndarray) -> np.ndarray:
        """
        Compute the quantile function F^{-1}(p).

        For 1-d distributions, the optimal transport map IS the
        quantile rearrangement: T = F_target^{-1} o F_source.
        """
        p = np.asarray(p, dtype=float)
        scalar = p.ndim == 0
        p = np.atleast_1d(p)

        if np.any(p < 0) or np.any(p > 1):
            raise ValueError("p must be in [0, 1]")

        # Place each atom at the midpoint of its CDF interval for smooth interpolation.
        # Interval for atom i is [cum_w[i-1], cum_w[i]]; midpoint = cum_w[i] - w[i]/2.
        cum_w_mid = self._cumulative_weights - self._sorted_weights / 2.0

        result = np.interp(
            p, cum_w_mid, self._sorted_samples,
            left=self._sorted_samples[0],
            right=self._sorted_samples[-1],
        )
        return float(result[0]) if scalar else result

    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Empirical CDF: F(x) = sum_{x_i <= x} w_i."""
        x = np.asarray(x, dtype=float)
        scalar = x.ndim == 0
        x = np.atleast_1d(x)

        idx = np.searchsorted(self._sorted_samples, x, side="right")
        result = np.where(
            idx == 0,
            0.0,
            self._cumulative_weights[np.minimum(idx - 1, self.n - 1)],
        )
        result = np.clip(result, 0.0, 1.0)
        return float(result[0]) if scalar else result

    def moments(self, order: int = 4) -> dict:
        """Compute moments: mean, variance, skewness, excess kurtosis."""
        mean = float(np.dot(self._sorted_weights, self._sorted_samples))
        centered = self._sorted_samples - mean
        variance = float(np.dot(self._sorted_weights, centered ** 2))
        std = float(np.sqrt(variance))

        result = {"mean": mean, "variance": variance}
        if order >= 3:
            skewness = (
                float(np.dot(self._sorted_weights, (centered / std) ** 3))
                if std > 0 else 0.0
            )
            result["skewness"] = skewness
        if order >= 4:
            kurtosis = (
                float(np.dot(self._sorted_weights, (centered / std) ** 4)) - 3.0
                if std > 0 else 0.0
            )
            result["kurtosis"] = kurtosis

        return result

    def descriptive_stats(self) -> dict:
        """
        Return: mean, median, std, skewness, kurtosis,
        interquartile range, 5th/95th percentiles,
        number of observations.
        """
        m = self.moments(order=4)
        return {
            "mean": m["mean"],
            "median": self.quantile_function(0.5),
            "std": float(np.sqrt(m["variance"])),
            "skewness": m["skewness"],
            "kurtosis": m["kurtosis"],
            "iqr": self.quantile_function(0.75) - self.quantile_function(0.25),
            "p5": self.quantile_function(0.05),
            "p95": self.quantile_function(0.95),
            "n": self.n,
        }

    def __repr__(self) -> str:
        s = self.descriptive_stats()
        return (
            f"EmpiricalDistribution(n={self.n}, "
            f"mean={s['mean']:.4f}, std={s['std']:.4f})"
        )

    def __len__(self) -> int:
        return self.n


class DistributionPanel:
    """
    A time series of cross-sectional distributions.

    For each date t, stores the empirical distribution of returns.
    This is the fundamental data structure for the library.

    Construction from panel data:
    - Input: DataFrame with columns ['date', 'ticker'/'permno', 'return']
    - For each date, extract the cross-section of returns
    - Optionally filter: min number of stocks, winsorize extremes
    """

    def __init__(self, distributions: dict, dates: list):
        """
        Args:
            distributions: {date: EmpiricalDistribution} mapping
            dates: sorted list of dates
        """
        self._distributions = dict(distributions)
        self.dates = sorted(dates)

    @classmethod
    def from_panel(
        cls,
        df: pd.DataFrame,
        date_col: str = "date",
        return_col: str = "return",
        min_stocks: int = 100,
        winsorize: float = 0.005,
    ) -> "DistributionPanel":
        """
        Build from long-format panel data.

        Winsorize: clip returns at the (winsorize, 1-winsorize) quantiles
        each date to remove data errors without losing distributional shape.
        """
        distributions: dict = {}
        dates: list = []

        for date, group in df.groupby(date_col, sort=True):
            returns = group[return_col].dropna().to_numpy(dtype=float)
            if len(returns) < min_stocks:
                continue

            if winsorize > 0.0:
                lo = float(np.quantile(returns, winsorize))
                hi = float(np.quantile(returns, 1.0 - winsorize))
                returns = np.clip(returns, lo, hi)

            distributions[date] = EmpiricalDistribution(returns)
            dates.append(date)

        return cls(distributions=distributions, dates=dates)

    def __getitem__(self, date) -> EmpiricalDistribution:
        return self._distributions[date]

    def __len__(self) -> int:
        return len(self.dates)

    def __iter__(self):
        return iter(self.dates)

    def rolling_window(self, date, lookback: int = 21) -> EmpiricalDistribution:
        """
        Pool returns over [date - lookback + 1, date] into a single distribution.
        Useful for smoothing when daily cross-sections are noisy.
        """
        date_idx = self.dates.index(date)
        start_idx = max(0, date_idx - lookback + 1)
        window_dates = self.dates[start_idx : date_idx + 1]

        all_samples = np.concatenate(
            [self._distributions[d].samples for d in window_dates]
        )
        return EmpiricalDistribution(all_samples)

    def subsample(self, start_date, end_date) -> "DistributionPanel":
        """Return a new panel restricted to [start_date, end_date]."""
        subset_dates = [d for d in self.dates if start_date <= d <= end_date]
        subset_dists = {d: self._distributions[d] for d in subset_dates}
        return DistributionPanel(distributions=subset_dists, dates=subset_dates)

    def __repr__(self) -> str:
        if self.dates:
            return (
                f"DistributionPanel(n_dates={len(self.dates)}, "
                f"start={self.dates[0]}, end={self.dates[-1]})"
            )
        return "DistributionPanel(empty)"
