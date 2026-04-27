"""
Synthetic return distribution generators for testing and benchmarking.

All generators return long-format DataFrames with columns:
    date    (pd.Timestamp)
    ticker  (str)
    return  (float)

plus any auxiliary data documented per function.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def generate_normal_panel(
    n_dates: int = 252,
    n_stocks: int = 500,
    mu: float = 0.0,
    sigma: float = 0.02,
    seed: int = None,
) -> pd.DataFrame:
    """
    Panel where cross-sectional returns are iid N(mu, sigma^2) each day.
    No regime changes. Baseline for testing.

    Args:
        n_dates:  number of trading days
        n_stocks: number of stocks per day
        mu:       cross-sectional mean return
        sigma:    cross-sectional standard deviation
        seed:     random seed for reproducibility

    Returns:
        Long-format DataFrame with columns [date, ticker, return].
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    tickers = [f"STOCK_{i:04d}" for i in range(n_stocks)]

    returns_matrix = rng.normal(mu, sigma, size=(n_dates, n_stocks))

    date_col = np.repeat(dates, n_stocks)
    ticker_col = np.tile(tickers, n_dates)
    return_col = returns_matrix.ravel()

    return pd.DataFrame({"date": date_col, "ticker": ticker_col, "return": return_col})


def generate_regime_switching_panel(
    n_dates: int = 504,
    n_stocks: int = 500,
    regimes: list = None,
    seed: int = None,
) -> tuple:
    """
    Panel with distinct distributional regimes.

    Default regimes (equal-length blocks):
        0 "calm":     N(+0.03%, 1.0%^2), slight positive skew via lognormal
        1 "crisis":   Student-t(df=5) scaled to N(-0.10%, 2.5%^2)
        2 "recovery": N(+0.10%, 1.5%^2)

    Args:
        n_dates:  total number of trading days
        n_stocks: stocks per day
        regimes:  list of dicts with keys: name, mu, sigma, dist, [df, skew]
        seed:     random seed

    Returns:
        (panel_df, regime_labels) where regime_labels[t] in {0, 1, 2, ...}
    """
    rng = np.random.default_rng(seed)

    if regimes is None:
        regimes = [
            {"name": "calm",     "mu": 0.0003,  "sigma": 0.010, "dist": "normal"},
            {"name": "crisis",   "mu": -0.001,  "sigma": 0.025, "dist": "t", "df": 5},
            {"name": "recovery", "mu": 0.001,   "sigma": 0.015, "dist": "normal"},
        ]

    n_regimes = len(regimes)
    regime_labels = np.zeros(n_dates, dtype=int)
    block = n_dates // n_regimes
    for k in range(n_regimes):
        lo = k * block
        hi = (k + 1) * block if k < n_regimes - 1 else n_dates
        regime_labels[lo:hi] = k

    dates = pd.bdate_range("2015-01-02", periods=n_dates)
    tickers = [f"STOCK_{i:04d}" for i in range(n_stocks)]

    all_returns = np.empty((n_dates, n_stocks), dtype=float)

    for k, regime in enumerate(regimes):
        mask = regime_labels == k
        n_k = int(mask.sum())
        mu_k = regime["mu"]
        sigma_k = regime["sigma"]

        if regime.get("dist") == "t":
            df = regime.get("df", 5)
            raw = rng.standard_t(df, size=(n_k, n_stocks))
            # Scale so cross-sectional std ≈ sigma_k
            raw = raw / np.sqrt(df / (df - 2))
            all_returns[mask] = mu_k + sigma_k * raw
        else:
            all_returns[mask] = rng.normal(mu_k, sigma_k, size=(n_k, n_stocks))

    date_col = np.repeat(dates, n_stocks)
    ticker_col = np.tile(tickers, n_dates)
    return_col = all_returns.ravel()

    panel_df = pd.DataFrame(
        {"date": date_col, "ticker": ticker_col, "return": return_col}
    )
    return panel_df, regime_labels


def generate_factor_driven_panel(
    n_dates: int = 252,
    n_stocks: int = 500,
    n_factors: int = 3,
    seed: int = None,
) -> tuple:
    """
    Panel where cross-sectional returns are driven by common factors:

        r_{i,t} = sum_k beta_{i,k} * f_{k,t} + epsilon_{i,t}

    Factor loadings beta_{i,k} are fixed per stock (drawn once from N(0, 1/K)).
    Factor returns f_{k,t} are time-varying; their volatility doubles at the
    midpoint to simulate a factor-volatility regime shift.

    When factor volatilities change the cross-sectional distribution shifts
    in a factor-aligned way — this is what the transport map decomposition
    should recover.

    Args:
        n_dates:   number of trading days
        n_stocks:  number of stocks
        n_factors: number of common factors
        seed:      random seed

    Returns:
        (panel_df, factor_returns, factor_loadings)
            panel_df:        long-format DataFrame [date, ticker, return]
            factor_returns:  (n_dates, n_factors) array of factor realisations
            factor_loadings: (n_stocks, n_factors) array of betas
    """
    rng = np.random.default_rng(seed)

    beta = rng.normal(0.0, 1.0 / np.sqrt(n_factors), size=(n_stocks, n_factors))

    # Factor volatility: low in first half, high in second half
    base_vols = np.linspace(0.01, 0.005, n_factors)
    high_vols = base_vols * 2.0

    factor_vols = np.where(
        np.arange(n_dates)[:, None] < n_dates // 2,
        base_vols[None, :],
        high_vols[None, :],
    )  # (n_dates, n_factors)

    z_factors = rng.standard_normal(size=(n_dates, n_factors))
    factor_returns = z_factors * factor_vols  # (n_dates, n_factors)

    idio_vol = 0.005
    epsilon = rng.normal(0.0, idio_vol, size=(n_dates, n_stocks))

    # (n_dates, n_stocks)
    returns_matrix = factor_returns @ beta.T + epsilon

    dates = pd.bdate_range("2018-01-02", periods=n_dates)
    tickers = [f"STOCK_{i:04d}" for i in range(n_stocks)]

    date_col = np.repeat(dates, n_stocks)
    ticker_col = np.tile(tickers, n_dates)
    return_col = returns_matrix.ravel()

    panel_df = pd.DataFrame(
        {"date": date_col, "ticker": ticker_col, "return": return_col}
    )
    return panel_df, factor_returns, beta


def generate_two_sample_test_data(
    n1: int = 500,
    n2: int = 500,
    shift: float = 0.0,
    scale_ratio: float = 1.0,
    seed: int = None,
) -> tuple:
    """
    Two samples for testing distributional equality.

    Sample 1: N(0, 1)
    Sample 2: N(shift, scale_ratio^2)

    When shift=0 and scale_ratio=1: null hypothesis (same distribution).
    Otherwise: known distributional difference for power analysis.

    Args:
        n1, n2:       sample sizes
        shift:        mean of sample 2
        scale_ratio:  std of sample 2 (sample 1 has std=1)
        seed:         random seed

    Returns:
        (sample1, sample2): two 1-d numpy arrays
    """
    rng = np.random.default_rng(seed)
    sample1 = rng.normal(0.0, 1.0, n1)
    sample2 = rng.normal(shift, scale_ratio, n2)
    return sample1, sample2
