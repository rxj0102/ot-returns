"""
Transport map decomposition into factor-aligned components.

The optimal transport map T from regime mu to regime nu can be
approximately decomposed along factor-loading directions:

    T(x) - x ≈ sum_{k=1}^{K} alpha_k * phi_k(x)

where phi_k(x) = <x, beta_k> captures the cross-sectional factor loading
and alpha_k is the scalar intensity of the k-th factor's contribution
to the distributional shift.

This connects OT-based regime analysis to the classical factor model:
    r_{i,t} = sum_k beta_{i,k} f_{k,t} + epsilon_{i,t}
"""

from __future__ import annotations

import numpy as np
from otreturns.distributions import EmpiricalDistribution


def decompose_transport_map(T_source: np.ndarray,
                             T_target: np.ndarray,
                             factor_loadings: np.ndarray) -> dict:
    """
    Decompose the displacement map (T_target - T_source) into
    factor-aligned components via least-squares projection.

    Args:
        T_source: source quantile grid values, shape (n_quantiles,)
        T_target: target quantile values T(x), shape (n_quantiles,)
        factor_loadings: (n_stocks, n_factors) loading matrix

    Returns:
        dict with keys: coefficients, residual, r_squared
    """
    raise NotImplementedError("Implemented in Prompt 7")


def factor_contribution_to_shift(mu: EmpiricalDistribution,
                                   nu: EmpiricalDistribution,
                                   factor_loadings: np.ndarray,
                                   n_quantiles: int = 1000) -> dict:
    """
    Attribute the W_2^2 transport cost between mu and nu to individual factors.

    Returns:
        dict mapping factor index -> attributed W_2^2 cost fraction
    """
    raise NotImplementedError("Implemented in Prompt 7")


def fit_factor_model_ot(panel,
                         factor_returns: np.ndarray,
                         n_quantiles: int = 500) -> dict:
    """
    Fit the transport-factor model across the full panel.

    For each consecutive pair of dates (t, t+1), computes the transport map
    and regresses the displacement onto the factor-return changes.

    Returns:
        dict with time-series of factor coefficients and fit quality
    """
    raise NotImplementedError("Implemented in Prompt 7")
