# ot-returns

A research-grade Python library for applying **optimal transport** theory to cross-sectional equity return distributions.

> Target venues: *Journal of Financial Econometrics*, *Quantitative Finance*, *Annals of Applied Statistics*

---

## What is this?

At each trading date $t$, the cross-section of stock returns

$$\mu_t = \frac{1}{N_t} \sum_{i=1}^{N_t} \delta_{r_{i,t}}$$

is a discrete probability measure.  As market conditions evolve this measure shifts, stretches, and changes shape.  **Optimal transport** provides the geometrically natural framework for quantifying and decomposing those changes.

`ot-returns` implements the full pipeline:

```
raw panel data
    → EmpiricalDistribution per date
    → pairwise Wasserstein distance matrix
    → spectral regime detection
    → regime barycenters
    → displacement interpolation between regimes
    → factor decomposition of transport maps
    → statistical tests for distributional change
```

---

## Mathematical Summary

### Wasserstein Distance

For $p \ge 1$ the $p$-Wasserstein distance between empirical measures on $\mathbb{R}$:

$$W_p^p(\mu, \nu) = \int_0^1 \bigl|F_\mu^{-1}(u) - F_\nu^{-1}(u)\bigr|^p \, du$$

Exact computation is $O(n \log n)$.  For two Gaussians:
$W_2^2(\mathcal{N}(\mu_1,\sigma_1^2), \mathcal{N}(\mu_2,\sigma_2^2)) = (\mu_1-\mu_2)^2 + (\sigma_1-\sigma_2)^2$

Note: $W_2$ averages **standard deviations**, not variances.

### Transport Maps (Brenier 1991)

The unique $W_2$-optimal map is the monotone rearrangement: $T = F_\nu^{-1} \circ F_\mu$.
For Gaussians it is affine: $T(x) = \mu_\nu + (\sigma_\nu/\sigma_\mu)(x - \mu_\mu)$.

### Wasserstein Barycenters (Agueh & Carlier 2011)

$$\bar\mu = \arg\min_\nu \sum_t \lambda_t W_2^2(\nu, \mu_t)$$

**1-d closed form**: $F_{\bar\mu}^{-1}(u) = \sum_t \lambda_t F_{\mu_t}^{-1}(u)$ — the barycenter quantile function is the weighted average of the input quantile functions.  $O(T \cdot Q)$ computation.

### McCann Displacement Interpolation (McCann 1997)

The W₂-geodesic has constant speed and a closed form in 1-d:

$$F_{\mu_t}^{-1}(u) = (1-t) F_{\mu_0}^{-1}(u) + t F_{\mu_1}^{-1}(u), \quad W_2(\mu_0, \mu_t) = t \cdot W_2(\mu_0, \mu_1)$$

### Tangent-Space PCA (Bigot, Cazelles & Papadakis 2017)

Log-map at barycenter $\bar\mu$: $v_t(u) = F_{\mu_t}^{-1}(u) - F_{\bar\mu}^{-1}(u)$.
PCA on the matrix $V \in \mathbb{R}^{T \times Q}$ finds dominant modes of distributional variation
(mean shift, vol change, skewness, ...).

### Factor Decomposition

Under $r_i = \sum_k \beta_{ik} f_k + \varepsilon_i$, the displacement field
$d(u) = F_\nu^{-1}(u) - F_\mu^{-1}(u)$ is decomposed as:

$$d(u) \approx \sum_k \alpha_k \bar\beta_k(u) + \varepsilon(u)$$

where $\bar\beta_k(u)$ is the average loading of factor $k$ at quantile $u$.
OLS gives factor intensities $\alpha_k$ and goodness-of-fit $R^2$.

---

## Installation

```bash
git clone https://github.com/rxj0102/ot-returns
cd ot-returns
pip install -e ".[dev]"
```

**Requirements**: Python ≥ 3.9, NumPy, SciPy, pandas, scikit-learn, matplotlib, POT (`python-ot`).

---

## Quickstart — Regime Detection in ~15 lines

```python
import sys; sys.path.insert(0, ".")
import numpy as np
from data.synthetic import generate_regime_switching_panel
from otreturns.distributions import DistributionPanel
from otreturns.regimes import WassersteinRegimeDetector
from otreturns.barycenters import regime_barycenters
from otreturns.interpolation import interpolation_path
from sklearn.metrics import adjusted_rand_score

# 1. Load data
panel_df, true_labels = generate_regime_switching_panel(n_dates=120, n_stocks=300, seed=42)
panel = DistributionPanel.from_panel(panel_df, min_stocks=50, winsorize=0.005)

# 2. Detect regimes
detector = WassersteinRegimeDetector(n_regimes=3)
pred_labels = detector.fit_predict(panel)
print(f"ARI = {adjusted_rand_score(true_labels[:len(panel.dates)], pred_labels):.3f}")

# 3. Compute regime barycenters
bary = regime_barycenters(panel, pred_labels, method="1d", n_support=300)

# 4. Interpolate between calm (0) and crisis (1) regime
path = interpolation_path(bary[0], bary[1], n_steps=10, n_support=300)
print(f"Path has {len(path)} distributions; "
      f"mean evolves from {bary[0].moments(1)['mean']:.4f} "
      f"to {bary[1].moments(1)['mean']:.4f}")
```

---

## Feature List

| Feature | Function / Class | Notes |
|---|---|---|
| Wasserstein distance (1-d) | `wasserstein_1d` | Exact, $O(n\log n)$ |
| Wasserstein distance (n-d) | `wasserstein_nd` | via POT EMD |
| Log-domain Sinkhorn | `sinkhorn_distance` | Numerical stable |
| Sliced Wasserstein | `sliced_wasserstein` | Fast approx for high-d |
| Pairwise distance matrix | `wasserstein_distance_matrix` | |
| OT map (1-d) | `optimal_transport_map_1d` | $T = F_\nu^{-1} \circ F_\mu$ |
| OT plan (general) | `optimal_transport_plan` | EMD or Sinkhorn |
| Transport cost decomposition | `transport_cost_decomposition` | Location/scale/shape |
| Transport path | `TransportPath` | Time series of maps |
| Barycenter (1-d exact) | `wasserstein_barycenter_1d` | Agueh-Carlier |
| Barycenter (Sinkhorn) | `wasserstein_barycenter_sinkhorn` | Fixed-support, any dim |
| Regime barycenters | `regime_barycenters` | Per-cluster Fréchet mean |
| McCann interpolation | `mccann_interpolation` | W₂-geodesic step |
| Interpolation path | `interpolation_path` | Full geodesic |
| Multi-marginal interpolation | `multi_marginal_interpolation` | Piecewise geodesic |
| Two-sample test | `wasserstein_two_sample_test` | Permutation, Phipson-Smyth |
| Change-point test | `wasserstein_change_point_test` | Before/after window |
| CUSUM monitoring | `wasserstein_cusum` | Online detection |
| Energy distance test | `energy_distance_test` | Székely-Rizzo |
| Rolling stability | `distribution_stability_test` | DataFrame output |
| Regime detection | `WassersteinRegimeDetector` | Spectral/hierarchical/kmedoids |
| Rolling regime detection | `rolling_regime_detection` | No lookahead |
| Silhouette analysis | `.silhouette_analysis()` | Wasserstein silhouette |
| Optimal k selection | `.optimal_n_regimes()` | Silhouette + gap stat |
| Factor decomposition | `transport_factor_decomposition` | OLS on displacement field |
| Rolling attribution | `rolling_factor_attribution` | DataFrame output |
| Variance decomposition | `variance_decomposition_ot` | W²-ANOVA |
| Transport PCA | `transport_pca` | Bigot et al. 2017 |

---

## Project Structure

```
ot-returns/
├── otreturns/           # Core library
│   ├── distributions.py # EmpiricalDistribution, DistributionPanel
│   ├── distances.py     # Wasserstein, Sinkhorn, sliced
│   ├── transport.py     # OT maps, plans, TransportPath
│   ├── barycenters.py   # 1-d exact & Sinkhorn barycenters
│   ├── interpolation.py # McCann interpolation, paths
│   ├── testing.py       # Permutation tests, CUSUM
│   ├── regimes.py       # WassersteinRegimeDetector
│   └── factors.py       # Factor decomposition, transport PCA
├── data/
│   ├── synthetic.py     # Controlled synthetic generators
│   └── loader.py        # CSV/Parquet loaders
├── experiments/         # Runnable experiment scripts
├── notebooks/           # Jupyter notebooks (01–04)
├── tests/               # 365+ pytest tests
└── docs/
    ├── math_background.md
    └── algorithms.md
```

---

## Running Tests

```bash
pytest --tb=short -q      # all 365 tests
pytest tests/test_regimes.py -v   # specific module
```

## Running Experiments

```bash
python experiments/run_regime_detection.py
python experiments/run_barycenter_analysis.py
python experiments/run_factor_decomposition.py
python experiments/run_two_sample_tests.py
python experiments/run_interpolation_visualization.py
```

Results and figures are saved to `results/` and `figures/`.

---

## References

- Villani, C. (2003). *Topics in Optimal Transportation*. AMS Graduate Studies in Mathematics.
- Peyré, G. & Cuturi, M. (2019). Computational Optimal Transport. *Foundations and Trends in Machine Learning* 11(5–6).
- Agueh, M. & Carlier, G. (2011). Barycenters in the Wasserstein space. *SIAM Journal on Mathematical Analysis* 43(2), 904–924.
- Bigot, J., Cazelles, E. & Papadakis, N. (2017). Geodesic PCA in the Wasserstein space by convex PCA. *ESAIM: Probability and Statistics* 21, 35–52.
- Székely, G.J. & Rizzo, M.L. (2004). Testing for equal distributions in high dimension. *InterStat* 5.
- Benamou, J.-D., Carlier, G., Cuturi, M., Nenna, L. & Peyré, G. (2015). Iterative Bregman projections for regularized transportation problems. *SIAM Journal on Scientific Computing* 37(2), A1111–A1138.
- Phipson, B. & Smyth, G.K. (2010). Permutation p-values should never be zero: calculating exact p-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology* 9(1).
- Flamary, R. et al. (2021). POT: Python Optimal Transport. *Journal of Machine Learning Research* 22(78), 1–8.
- McCann, R.J. (1997). A convexity principle for interacting gases. *Advances in Mathematics* 128(1), 153–179.
- Gatheral, J., Jaisson, T. & Rosenbaum, M. (2018). Volatility is rough. *Quantitative Finance* 18(6), 933–949.
