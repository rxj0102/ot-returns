"""
Two-sample distributional tests based on Wasserstein and energy distances.

All permutation tests use the Phipson & Smyth (2010) correction:
    p-value = (1 + #{T_b >= T_obs}) / (1 + B)
ensuring valid (never exactly zero) p-values.

References:
    Székely & Rizzo (2004) — energy distance
    Phipson & Smyth (2010) — permutation p-value correction
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from otreturns.distributions import EmpiricalDistribution, DistributionPanel
from otreturns.distances import wasserstein_1d
from otreturns.transport import transport_cost_decomposition


# ---------------------------------------------------------------------------
# Permutation test — Wasserstein two-sample
# ---------------------------------------------------------------------------

def wasserstein_two_sample_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    p: int = 2,
    n_permutations: int = 1000,
    seed: int = None,
) -> dict:
    """
    Permutation test for equality of distributions using Wasserstein distance.

    H0: mu = nu (same distribution)
    H1: mu != nu

    Test statistic: T = W_p(mu_hat, nu_hat)

    Permutation procedure:
    1. Compute T_obs = W_p(sample1, sample2)
    2. For b = 1, ..., B:
       a. Pool samples and randomly split into two groups of sizes n1, n2
       b. Compute T_b = W_p(group1_b, group2_b)
    3. p-value = (1 + #{T_b >= T_obs}) / (1 + B)

    Args:
        sample1:        1-d array of observations from distribution mu
        sample2:        1-d array of observations from distribution nu
        p:              Wasserstein order (default 2)
        n_permutations: number of permutation draws B
        seed:           random seed for reproducibility

    Returns:
        dict with:
            'statistic':               observed W_p distance
            'p_value':                 permutation p-value in (0, 1]
            'permutation_distribution': array of B permutation statistics
            'reject_h0':               bool, reject at alpha=0.05
    """
    sample1 = np.asarray(sample1, dtype=float).ravel()
    sample2 = np.asarray(sample2, dtype=float).ravel()
    n1, n2 = len(sample1), len(sample2)

    mu = EmpiricalDistribution(sample1)
    nu = EmpiricalDistribution(sample2)
    T_obs = wasserstein_1d(mu, nu, p=p)

    rng = np.random.default_rng(seed)
    pooled = np.concatenate([sample1, sample2])
    perm_stats = np.empty(n_permutations)

    for b in range(n_permutations):
        idx = rng.permutation(n1 + n2)
        g1 = EmpiricalDistribution(pooled[idx[:n1]])
        g2 = EmpiricalDistribution(pooled[idx[n1:]])
        perm_stats[b] = wasserstein_1d(g1, g2, p=p)

    p_value = (1.0 + float(np.sum(perm_stats >= T_obs))) / (1.0 + n_permutations)

    return {
        "statistic": float(T_obs),
        "p_value": float(p_value),
        "permutation_distribution": perm_stats,
        "reject_h0": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# Change-point test
# ---------------------------------------------------------------------------

def wasserstein_change_point_test(
    panel: DistributionPanel,
    candidate_date,
    window: int = 21,
    n_permutations: int = 500,
) -> dict:
    """
    Test whether a distributional change occurred at a specific date.

    Compares the cross-section in the window of `window` dates immediately
    before candidate_date against the window immediately from candidate_date
    onward, using the Wasserstein two-sample permutation test.

    Args:
        panel:          DistributionPanel
        candidate_date: date at which to test for a change
        window:         number of dates on each side
        n_permutations: permutation replicates for the sub-test

    Returns:
        dict with keys: statistic, p_value, permutation_distribution,
                        reject_h0, effect_size (= W_2 distance between
                        the pooled before/after distributions)
    """
    dates = panel.dates
    idx = dates.index(candidate_date)

    before_dates = dates[max(0, idx - window): idx]
    after_dates = dates[idx: min(len(dates), idx + window)]

    if not before_dates or not after_dates:
        raise ValueError(
            f"Not enough dates around {candidate_date} for window={window}"
        )

    before_samples = np.concatenate([panel[d].samples for d in before_dates])
    after_samples = np.concatenate([panel[d].samples for d in after_dates])

    result = wasserstein_two_sample_test(
        before_samples, after_samples,
        p=2, n_permutations=n_permutations,
    )
    result["effect_size"] = result["statistic"]
    return result


# ---------------------------------------------------------------------------
# CUSUM statistic
# ---------------------------------------------------------------------------

def wasserstein_cusum(
    panel: DistributionPanel,
    reference_period: tuple = None,
    p: int = 2,
) -> np.ndarray:
    """
    CUSUM-type statistic for online distributional change detection.

    Procedure:
    1. Build reference distribution mu_ref by pooling all samples from the
       reference period (default: first quarter of the panel).
    2. Estimate drift delta = mean W_p(mu_t, mu_ref) over the reference period.
    3. For each post-reference date t:
           S_t = S_{t-1} + (W_p(mu_t, mu_ref) - delta)
       with S_0 = 0.

    Rising S_t signals a sustained shift away from the reference.
    Flat (near-zero) S_t indicates stability.

    Args:
        panel:            DistributionPanel
        reference_period: (start_date, end_date) inclusive; default: first
                          quarter of dates
        p:                Wasserstein order

    Returns:
        1-d array of CUSUM values, one per post-reference date (length may
        be 0 if the whole panel is consumed by the reference period).
    """
    dates = panel.dates
    n = len(dates)

    if reference_period is None:
        n_ref = max(2, n // 4)
        ref_start, ref_end = dates[0], dates[n_ref - 1]
    else:
        ref_start, ref_end = reference_period[0], reference_period[1]

    ref_dates = [d for d in dates if ref_start <= d <= ref_end]
    post_dates = [d for d in dates if d > ref_end]

    if len(ref_dates) < 2:
        raise ValueError("Reference period must contain at least 2 dates")

    # Reference distribution: pool all reference samples
    ref_samples = np.concatenate([panel[d].samples for d in ref_dates])
    mu_ref = EmpiricalDistribution(ref_samples)

    # Estimate in-sample mean distance (drift under H0)
    ref_distances = np.array([
        wasserstein_1d(panel[d], mu_ref, p=p) for d in ref_dates
    ])
    delta = float(ref_distances.mean())

    # Build CUSUM
    cusum = np.empty(len(post_dates))
    S = 0.0
    for i, d in enumerate(post_dates):
        w = wasserstein_1d(panel[d], mu_ref, p=p)
        S += w - delta
        cusum[i] = S

    return cusum


# ---------------------------------------------------------------------------
# Energy distance test
# ---------------------------------------------------------------------------

def energy_distance_test(
    sample1: np.ndarray,
    sample2: np.ndarray,
    n_permutations: int = 1000,
) -> dict:
    """
    Energy distance two-sample test (Székely & Rizzo, 2004).

    Test statistic:
        E(X, Y) = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'||

    Equivalent (up to scaling) to the maximum mean discrepancy with a
    Laplace kernel, and closely related to W_1.  Faster than W_2 for large
    samples since it requires only pairwise L1 distances.

    Args:
        sample1:        1-d observations from distribution mu
        sample2:        1-d observations from distribution nu
        n_permutations: number of permutation draws

    Returns:
        dict with:
            'statistic':               observed energy distance
            'p_value':                 permutation p-value
            'permutation_distribution': array of permutation statistics
            'reject_h0':               bool at alpha=0.05
    """
    sample1 = np.asarray(sample1, dtype=float).ravel()
    sample2 = np.asarray(sample2, dtype=float).ravel()

    T_obs = _energy_distance(sample1, sample2)

    rng = np.random.default_rng(None)
    n1 = len(sample1)
    pooled = np.concatenate([sample1, sample2])
    N = len(pooled)
    perm_stats = np.empty(n_permutations)

    for b in range(n_permutations):
        idx = rng.permutation(N)
        perm_stats[b] = _energy_distance(pooled[idx[:n1]], pooled[idx[n1:]])

    p_value = (1.0 + float(np.sum(perm_stats >= T_obs))) / (1.0 + n_permutations)

    return {
        "statistic": float(T_obs),
        "p_value": float(p_value),
        "permutation_distribution": perm_stats,
        "reject_h0": p_value < 0.05,
    }


def _energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """E(X,Y) = 2 E|X-Y| - E|X-X'| - E|Y-Y'|."""
    n, m = len(x), len(y)
    cross = np.mean(np.abs(x[:, None] - y[None, :]))
    xx = np.mean(np.abs(x[:, None] - x[None, :]))
    yy = np.mean(np.abs(y[:, None] - y[None, :]))
    return float(2.0 * cross - xx - yy)


# ---------------------------------------------------------------------------
# Rolling distribution stability
# ---------------------------------------------------------------------------

def distribution_stability_test(
    panel: DistributionPanel,
    window: int = 21,
    step: int = 5,
) -> pd.DataFrame:
    """
    Rolling stability analysis comparing each date's window to the previous.

    For each date t spaced `step` apart (starting at index `window`):
        - "previous" window: dates[t_idx - 2*window : t_idx - window]
        - "current"  window: dates[t_idx - window : t_idx]
    Pool samples within each window, then run wasserstein_two_sample_test
    and transport_cost_decomposition between the two pooled distributions.

    Args:
        panel:  DistributionPanel
        window: number of dates per comparison window
        step:   stride between tested dates

    Returns:
        DataFrame with columns:
            date, wasserstein_distance, p_value,
            location_shift, scale_change, shape_change
    """
    dates = panel.dates
    rows = []

    for i in range(2 * window, len(dates), step):
        date = dates[i]
        prev_dates = dates[i - 2 * window: i - window]
        curr_dates = dates[i - window: i]

        prev_samples = np.concatenate([panel[d].samples for d in prev_dates])
        curr_samples = np.concatenate([panel[d].samples for d in curr_dates])

        result = wasserstein_two_sample_test(
            prev_samples, curr_samples, p=2, n_permutations=200
        )

        prev_dist = EmpiricalDistribution(prev_samples)
        curr_dist = EmpiricalDistribution(curr_samples)
        decomp = transport_cost_decomposition(prev_dist, curr_dist)

        rows.append({
            "date": date,
            "wasserstein_distance": result["statistic"],
            "p_value": result["p_value"],
            "location_shift": decomp["location"],
            "scale_change": decomp["scale"],
            "shape_change": decomp["shape"],
        })

    return pd.DataFrame(rows, columns=[
        "date", "wasserstein_distance", "p_value",
        "location_shift", "scale_change", "shape_change",
    ])
