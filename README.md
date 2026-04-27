# ot-returns

A research-grade Python library for applying **optimal transport** theory to
cross-sectional equity return distributions.

## Mathematical Overview

At each date $t$, the cross-section of stock returns

$$\mu_t = \frac{1}{N_t} \sum_{i=1}^{N_t} \delta_{r_{i,t}}$$

is a discrete probability measure. As market conditions evolve this measure
shifts, stretches, and changes shape. Optimal transport provides the
geometrically natural framework for quantifying and decomposing those changes.

### Wasserstein Distance

For $p \ge 1$ the $p$-Wasserstein distance between two measures $\mu$ and $\nu$
on $\mathbb{R}$ is

$$W_p(\mu, \nu) = \left( \inf_{\gamma \in \Pi(\mu,\nu)} \int |x - y|^p \, d\gamma(x,y) \right)^{1/p}$$

In one dimension the infimum is attained by the **quantile coupling**:

$$W_p^p(\mu, \nu) = \int_0^1 |F_\mu^{-1}(u) - F_\nu^{-1}(u)|^p \, du$$

making computation exact and $O(n \log n)$.

### Transport Maps and Factor Decomposition

The optimal transport map $T: \text{supp}(\mu) \to \text{supp}(\nu)$ pushes
$\mu$ forward to $\nu$ while minimising the total squared displacement.
In one dimension $T = F_\nu^{-1} \circ F_\mu$.

For multi-period data we decompose $T$ into factor-aligned components:

$$T(x) \approx \sum_{k=1}^{K} \langle T, \phi_k \rangle \phi_k(x)$$

where $\phi_k$ are factor-loading directions, linking distributional shifts
to Fama–French-style factor exposures.

### Wasserstein Barycenters

The Wasserstein barycenter of $\{\mu_t\}_{t \in \mathcal{T}}$ is

$$\bar{\mu} = \arg\min_\nu \sum_{t \in \mathcal{T}} \lambda_t W_2^2(\nu, \mu_t)$$

Used as a canonical "average market state" across a regime, with displacement
interpolation (McCann 1997) tracing the geodesic between regimes.

### Regime Detection

Wasserstein distances form a dissimilarity matrix $D_{s,t} = W_2(\mu_s, \mu_t)$.
Spectral clustering on $D$ yields distributional regimes without assuming
parametric structure on the return distribution.

## Installation

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
import pandas as pd
from otreturns.distributions import DistributionPanel
from data.synthetic import generate_regime_switching_panel

# Generate synthetic data with three regimes
panel_df, regime_labels = generate_regime_switching_panel(
    n_dates=504, n_stocks=500, seed=42
)

# Build distribution panel
panel = DistributionPanel.from_panel(panel_df, min_stocks=100, winsorize=0.005)

# Inspect a single cross-sectional distribution
date = panel.dates[100]
dist = panel[date]
print(dist.descriptive_stats())
```

## Module Reference

| Module | Description |
|---|---|
| `otreturns.distributions` | `EmpiricalDistribution` and `DistributionPanel` |
| `otreturns.distances` | Wasserstein and Sinkhorn distances |
| `otreturns.transport` | Optimal transport maps and plans |
| `otreturns.barycenters` | Wasserstein barycenters |
| `otreturns.interpolation` | Displacement interpolation (McCann) |
| `otreturns.testing` | Two-sample tests, bootstrap inference |
| `otreturns.regimes` | Wasserstein-based regime detection |
| `otreturns.factors` | Transport map ↔ factor decomposition |
| `otreturns.utils` | Grid utilities, numerical helpers |

## Target Venues

- Journal of Financial Econometrics
- Quantitative Finance
- Annals of Applied Statistics
