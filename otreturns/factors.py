"""
Transport map decomposition into factor-aligned components.

The optimal transport map T from μ to ν can be approximately decomposed
along factor-loading directions:

    T(x) - x ≈ Σ_k α_k · β̄_k(p)   + residual(p)

where β̄_k(p) is the average loading of factor k for stocks near quantile p
and α_k is the scalar intensity of factor k's contribution to the shift.

Connection to the factor model:
    r_{i,t} = Σ_k β_{i,k} f_{k,t} + ε_{i,t}

When factor returns change between t and t+1, the cross-sectional distribution
shifts in a factor-loading-aligned direction.  The decomposition recovers α_k.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from otreturns.distributions import EmpiricalDistribution, DistributionPanel
from otreturns.distances import wasserstein_1d
from otreturns.barycenters import wasserstein_barycenter_1d


# ---------------------------------------------------------------------------
# Core decomposition
# ---------------------------------------------------------------------------

def transport_factor_decomposition(
    source: EmpiricalDistribution,
    target: EmpiricalDistribution,
    factor_loadings: np.ndarray,
    factor_names: list = None,
    n_quantiles: int = 200,
) -> dict:
    """
    Decompose the optimal transport map into factor-aligned components.

    The displacement field d(p) = F_ν^{-1}(p) - F_μ^{-1}(p) tells how
    quantile p shifted.  Stocks near quantile p in the source have average
    factor loadings β̄(p); we regress d(p) onto β̄(p) across quantile levels.

    Procedure:
    1. Build quantile grid u_1,...,u_M.
    2. Compute displacement field d(u_j) = F_target^{-1}(u_j) - F_source^{-1}(u_j).
    3. Sort source stocks by return to obtain the quantile-loading relationship.
       β̄_k(u_j) = interpolated loading of factor k at quantile u_j.
    4. OLS: d = B @ α + residual, where B[j,k] = β̄_k(u_j).
    5. Factor displacement field for factor k: g_k(u_j) = α_k · β̄_k(u_j).

    Args:
        source:          source distribution μ
        target:          target distribution ν
        factor_loadings: (n_stocks, K) matrix of factor loadings β_{i,k}
        factor_names:    optional list of K factor names
        n_quantiles:     number of quantile grid points (resolution)

    Returns:
        dict with:
            'factor_contributions':      {name: α_k}
            'factor_fraction':           fraction of ||d||² explained by factors
            'residual_fraction':         unexplained fraction
            'displacement_field':        d(u) array of shape (n_quantiles,)
            'factor_displacement_fields':{name: g_k(u)}
            'r_squared':                 OLS R² of regression d ~ β̄ @ α
    """
    factor_loadings = np.asarray(factor_loadings, dtype=float)
    if factor_loadings.ndim == 1:
        factor_loadings = factor_loadings[:, None]
    n_stocks, K = factor_loadings.shape

    if factor_names is None:
        factor_names = [f"factor_{k}" for k in range(K)]
    if len(factor_names) != K:
        raise ValueError(f"factor_names length {len(factor_names)} != K={K}")

    # Quantile grid
    u = (np.arange(n_quantiles) + 0.5) / n_quantiles

    # Displacement field
    Q_src = source.quantile_function(u)   # (M,)
    Q_tgt = target.quantile_function(u)   # (M,)
    displacement = Q_tgt - Q_src          # (M,)

    # Build the factor-loading profile along the quantile axis.
    # Sort source stocks by their return value; assign each a quantile level.
    src_samples = source.samples          # (n_src,)
    n_src = len(src_samples)
    sort_idx = np.argsort(src_samples)
    src_q_levels = (np.arange(n_src) + 0.5) / n_src  # midpoint quantile per stock

    # Align factor_loadings with source stocks.
    # If n_stocks == n_src, assume one-to-one correspondence.
    # Otherwise interpolate loadings from (n_stocks,) onto (n_src,) points.
    if n_stocks == n_src:
        beta_sorted = factor_loadings[sort_idx]          # (n_src, K)
    else:
        beta_sorted = np.zeros((n_src, K))
        stock_q = (np.arange(n_stocks) + 0.5) / n_stocks
        for k in range(K):
            beta_sorted[:, k] = np.interp(src_q_levels, stock_q, factor_loadings[:, k])

    # Interpolate β̄_k onto the quantile grid u
    B = np.zeros((n_quantiles, K))        # (M, K)
    for k in range(K):
        B[:, k] = np.interp(u, src_q_levels, beta_sorted[:, k])

    # OLS: displacement = B @ alpha + residual
    alpha, _, _, _ = np.linalg.lstsq(B, displacement, rcond=None)

    d_hat = B @ alpha                     # (M,)
    residual_vec = displacement - d_hat

    ss_tot = float(np.dot(displacement, displacement))
    ss_res = float(np.dot(residual_vec, residual_vec))
    ss_hat = float(np.dot(d_hat, d_hat))

    r_squared = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-20))
    factor_fraction = ss_hat / max(ss_tot, 1e-20)
    residual_fraction = 1.0 - factor_fraction

    factor_contributions = {name: float(alpha[k]) for k, name in enumerate(factor_names)}
    factor_displacement_fields = {
        name: alpha[k] * B[:, k] for k, name in enumerate(factor_names)
    }

    return {
        "factor_contributions": factor_contributions,
        "factor_fraction": float(np.clip(factor_fraction, 0.0, 1.0)),
        "residual_fraction": float(np.clip(residual_fraction, 0.0, 1.0)),
        "displacement_field": displacement,
        "factor_displacement_fields": factor_displacement_fields,
        "r_squared": float(r_squared),
    }


# ---------------------------------------------------------------------------
# Rolling factor attribution
# ---------------------------------------------------------------------------

def rolling_factor_attribution(
    panel: DistributionPanel,
    factor_loadings_panel: pd.DataFrame,
    window: int = 1,
    factor_names: list = None,
) -> pd.DataFrame:
    """
    For each consecutive pair of dates (t-window, t), decompose the
    distributional shift into factor contributions.

    Args:
        panel:                 DistributionPanel
        factor_loadings_panel: DataFrame of shape (n_stocks, K) giving fixed
                               factor loadings; or a dict {date: (n_stocks, K)}
                               for time-varying loadings.
        window:                step between source and target date (default 1)
        factor_names:          optional list of K factor names

    Returns:
        DataFrame with columns:
            date, total_shift_W2,
            <factor_name>_contribution (one per factor),
            residual, factor_r_squared
    """
    dates = panel.dates

    if isinstance(factor_loadings_panel, pd.DataFrame):
        loadings_static = factor_loadings_panel.values
        if factor_names is None:
            factor_names = list(factor_loadings_panel.columns)
        loadings_dict = None
    elif isinstance(factor_loadings_panel, dict):
        loadings_static = None
        loadings_dict = factor_loadings_panel
    else:
        loadings_static = np.asarray(factor_loadings_panel, dtype=float)
        loadings_dict = None

    K = (loadings_static.shape[1] if loadings_static is not None
         else next(iter(loadings_dict.values())).shape[1])

    if factor_names is None:
        factor_names = [f"factor_{k}" for k in range(K)]

    rows = []
    for i in range(window, len(dates)):
        src_date = dates[i - window]
        tgt_date = dates[i]
        source = panel[src_date]
        target = panel[tgt_date]

        if loadings_dict is not None:
            loadings = loadings_dict.get(tgt_date, loadings_dict.get(src_date))
        else:
            loadings = loadings_static

        w2 = wasserstein_1d(source, target, p=2)
        decomp = transport_factor_decomposition(
            source, target, loadings, factor_names
        )

        row = {
            "date": tgt_date,
            "total_shift_W2": float(w2),
            "residual": decomp["residual_fraction"],
            "factor_r_squared": decomp["r_squared"],
        }
        for name in factor_names:
            row[f"{name}_contribution"] = decomp["factor_contributions"][name]
        rows.append(row)

    contribution_cols = [f"{n}_contribution" for n in factor_names]
    cols = ["date", "total_shift_W2"] + contribution_cols + ["residual", "factor_r_squared"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Wasserstein variance decomposition
# ---------------------------------------------------------------------------

def variance_decomposition_ot(
    panel: DistributionPanel,
    factor_loadings_panel: pd.DataFrame,
    regime_labels: np.ndarray = None,
) -> dict:
    """
    Decompose time-series variation of cross-sectional distributions into
    factor-driven and idiosyncratic components.

    Total variation = (1/T) Σ_t W_2²(μ_t, μ̄)

    If regime_labels given:
        between_regime = Σ_k (T_k/T) W_2²(μ̄_k, μ̄)
        within_regime  = (1/T) Σ_t W_2²(μ_t, μ̄_{label_t})

    Factor-explained fraction = mean R² from transport_factor_decomposition
    applied to each (μ̄, μ_t) displacement.

    Returns:
        dict with 'total_variation', 'between_regime', 'within_regime',
                  'factor_explained_fraction', 'idiosyncratic_fraction',
                  and their fractions of total variation.
    """
    dates = panel.dates
    T = len(dates)
    dists = [panel[d] for d in dates]

    # Overall barycenter
    mu_bar = wasserstein_barycenter_1d(dists)

    # Total variation
    w2_to_bar = np.array([wasserstein_1d(d, mu_bar, p=2) for d in dists])
    total_variation = float(np.mean(w2_to_bar ** 2))

    if isinstance(factor_loadings_panel, pd.DataFrame):
        loadings = factor_loadings_panel.values
    else:
        loadings = np.asarray(factor_loadings_panel, dtype=float)

    # Factor-explained: mean R² from displacement of each μ_t relative to μ̄
    r2_vals = []
    for d in dists:
        dec = transport_factor_decomposition(mu_bar, d, loadings)
        r2_vals.append(dec["r_squared"])
    factor_explained_fraction = float(np.mean(r2_vals))
    idiosyncratic_fraction = 1.0 - factor_explained_fraction

    result = {
        "total_variation": total_variation,
        "factor_explained_fraction": factor_explained_fraction,
        "idiosyncratic_fraction": idiosyncratic_fraction,
    }

    if regime_labels is not None:
        regime_labels = np.asarray(regime_labels, dtype=int)
        unique = np.unique(regime_labels)

        # Regime barycenters
        regime_bary = {}
        for lbl in unique:
            mask = regime_labels == lbl
            regime_bary[lbl] = wasserstein_barycenter_1d(
                [dists[i] for i in range(T) if mask[i]]
            )

        # Within-regime variation
        within_sq = np.array([
            wasserstein_1d(dists[t], regime_bary[regime_labels[t]], p=2) ** 2
            for t in range(T)
        ])
        within_regime = float(np.mean(within_sq))

        # Between-regime variation (weighted by regime size)
        between_regime = 0.0
        for lbl in unique:
            w = float(np.sum(regime_labels == lbl)) / T
            between_regime += w * wasserstein_1d(regime_bary[lbl], mu_bar, p=2) ** 2

        result["between_regime"] = between_regime
        result["within_regime"] = within_regime
        result["between_fraction"] = between_regime / max(total_variation, 1e-20)
        result["within_fraction"] = within_regime / max(total_variation, 1e-20)

    return result


# ---------------------------------------------------------------------------
# Transport PCA (Bigot, Cazelles & Papadakis 2017)
# ---------------------------------------------------------------------------

def transport_pca(
    panel: DistributionPanel,
    n_components: int = 3,
    n_support: int = 200,
) -> dict:
    """
    Principal component analysis in Wasserstein space.

    Method (Bigot, Cazelles & Papadakis 2017):
    1. Compute the Wasserstein barycenter μ̄ of all distributions.
    2. For each date t compute the log-map (tangent vector at μ̄):
           v_t(u) = F_{μ_t}^{-1}(u) - F_{μ̄}^{-1}(u)
    3. Apply standard PCA to the matrix V ∈ R^{T×M}.
    4. The principal components are directions in the tangent space at μ̄.

    Interpretation:
        PC1 typically captures mean shifts (bull vs bear).
        PC2 typically captures dispersion changes (vol regime).
        PC3 typically captures skewness / tail changes.

    Args:
        panel:        DistributionPanel
        n_components: number of principal components to retain
        n_support:    quantile grid resolution

    Returns:
        dict with:
            'components':              (n_comp, n_support) tangent directions
            'explained_variance_ratio': (n_comp,) PCA explained variance
            'scores':                  (T, n_comp) projection of each date
            'barycenter':              reference distribution μ̄
            'reconstructed':           list of T EmpiricalDistributions
                                       reconstructed from top-k PCs
    """
    dates = panel.dates
    T = len(dates)
    dists = [panel[d] for d in dates]

    # Barycenter
    barycenter = wasserstein_barycenter_1d(dists, n_support=n_support)

    # Quantile grid aligned with barycenter
    u = (np.arange(n_support) + 0.5) / n_support
    Q_bar = barycenter.quantile_function(u)   # (M,)

    # Log-map matrix: V[t, :] = quantile_t(u) - Q_bar(u)
    V = np.stack([d.quantile_function(u) - Q_bar for d in dists], axis=0)  # (T, M)

    # PCA
    n_comp = min(n_components, T - 1, n_support)
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(V)             # (T, n_comp)

    # Reconstruct via exp-map: Q_recon = Q_bar + PC_projection
    V_recon = pca.inverse_transform(scores)   # (T, M)
    reconstructed = []
    for t in range(T):
        Q_recon = np.sort(Q_bar + V_recon[t])  # sort ensures valid quantile fn
        reconstructed.append(EmpiricalDistribution(Q_recon))

    return {
        "components": pca.components_,             # (n_comp, M)
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "scores": scores,                           # (T, n_comp)
        "barycenter": barycenter,
        "reconstructed": reconstructed,
    }


# ---------------------------------------------------------------------------
# Legacy stub API (kept for backward compat with placeholder tests)
# ---------------------------------------------------------------------------

def decompose_transport_map(
    T_source: np.ndarray,
    T_target: np.ndarray,
    factor_loadings: np.ndarray,
) -> dict:
    """
    Decompose displacement (T_target - T_source) onto factor loading directions.

    Args:
        T_source:        source quantile grid values, shape (n_quantiles,)
        T_target:        transported quantile values, shape (n_quantiles,)
        factor_loadings: (n_stocks, K) loading matrix

    Returns:
        dict with keys: coefficients, residual_vec, r_squared
    """
    T_source = np.asarray(T_source, dtype=float)
    T_target = np.asarray(T_target, dtype=float)
    factor_loadings = np.asarray(factor_loadings, dtype=float)
    if factor_loadings.ndim == 1:
        factor_loadings = factor_loadings[:, None]

    n_quantiles = len(T_source)
    displacement = T_target - T_source     # (M,)

    # Interpolate loadings onto quantile grid
    n_stocks, K = factor_loadings.shape
    stock_q = (np.arange(n_stocks) + 0.5) / n_stocks
    grid_q = (np.arange(n_quantiles) + 0.5) / n_quantiles
    B = np.zeros((n_quantiles, K))
    for k in range(K):
        B[:, k] = np.interp(grid_q, stock_q, factor_loadings[:, k])

    alpha, _, _, _ = np.linalg.lstsq(B, displacement, rcond=None)
    d_hat = B @ alpha
    residual_vec = displacement - d_hat

    ss_tot = float(np.dot(displacement, displacement))
    ss_res = float(np.dot(residual_vec, residual_vec))
    r_squared = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-20))

    return {
        "coefficients": alpha,
        "residual_vec": residual_vec,
        "r_squared": r_squared,
    }


def factor_contribution_to_shift(
    mu: EmpiricalDistribution,
    nu: EmpiricalDistribution,
    factor_loadings: np.ndarray,
    n_quantiles: int = 1000,
) -> dict:
    """Attribute W_2² transport cost between mu and nu to individual factors."""
    result = transport_factor_decomposition(mu, nu, factor_loadings,
                                            n_quantiles=n_quantiles)
    return result["factor_contributions"]


def fit_factor_model_ot(
    panel: DistributionPanel,
    factor_returns: np.ndarray,
    n_quantiles: int = 500,
) -> dict:
    """Fit transport-factor model across the full panel."""
    raise NotImplementedError(
        "fit_factor_model_ot is superseded by rolling_factor_attribution"
    )
