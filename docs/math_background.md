# Mathematical Background

## 1. Optimal Transport

### The Kantorovich Problem

Given probability measures $\mu$ on $\mathcal{X}$ and $\nu$ on $\mathcal{Y}$,
the **optimal transport problem** (Kantorovich 1942) is:

$$W_p^p(\mu, \nu) = \inf_{\gamma \in \Pi(\mu,\nu)} \int_{\mathcal{X} \times \mathcal{Y}} c(x,y) \, d\gamma(x,y)$$

where $\Pi(\mu,\nu)$ is the set of **couplings** (joint distributions with
marginals $\mu$ and $\nu$) and $c(x,y) = |x-y|^p$ is the ground cost.

The infimum is the **$p$-Wasserstein distance** $W_p(\mu,\nu)$.

### One-Dimensional Case

In one dimension, $\Pi(\mu,\nu)$ has a unique optimal element: the
**quantile coupling** $(F_\mu^{-1}(U), F_\nu^{-1}(U))$ for $U \sim \text{Uniform}[0,1]$.
This gives:

$$W_p^p(\mu,\nu) = \int_0^1 |F_\mu^{-1}(u) - F_\nu^{-1}(u)|^p \, du$$

which can be computed exactly in $O(n \log n)$ after sorting.

The **optimal transport map** is $T = F_\nu^{-1} \circ F_\mu$, pushing $\mu$
to $\nu$ while minimising $\int |x - T(x)|^p \, d\mu(x)$.

### Wasserstein Space

The space of probability measures with finite $p$-th moment, equipped with
$W_p$, forms a metric space $(\mathcal{P}_p(\mathcal{X}), W_p)$ called
**Wasserstein space**. Its geometry is richer than the $L^2$ geometry on
densities: it respects the support structure of the measures.

## 2. Wasserstein Barycenters

The **Wasserstein barycenter** (Agueh & Carlier 2011) of measures
$\{\mu_1, \ldots, \mu_T\}$ with weights $\lambda_t \geq 0$, $\sum \lambda_t = 1$:

$$\bar{\mu} = \arg\min_{\nu \in \mathcal{P}_2} \sum_{t=1}^T \lambda_t W_2^2(\nu, \mu_t)$$

**1-d formula**: $F_{\bar\mu}^{-1}(u) = \sum_t \lambda_t F_{\mu_t}^{-1}(u)$.
The barycenter quantile function is the weighted average of individual quantile functions.

## 3. Displacement Interpolation

**McCann's interpolation** (McCann 1997) defines the geodesic in
$(\mathcal{P}_2, W_2)$ between $\mu_0$ and $\mu_1$:

$$\mu_t = \big((1-t)\text{Id} + t T\big)_\# \mu_0, \quad t \in [0,1]$$

where $T$ is the optimal transport map from $\mu_0$ to $\mu_1$.

In 1-d: $F_{\mu_t}^{-1}(u) = (1-t) F_{\mu_0}^{-1}(u) + t F_{\mu_1}^{-1}(u)$.

This traces the shortest path between two distributions in Wasserstein space.

## 4. Cross-Sectional Return Distributions

At each date $t$, define the **cross-sectional distribution**:

$$\mu_t = \frac{1}{N_t} \sum_{i=1}^{N_t} \delta_{r_{i,t}}$$

This is an empirical measure. The panel $\{\mu_t\}_{t=1}^T$ is a time series
in Wasserstein space.

**Regime detection**: cluster dates by the pairwise distance matrix
$D_{s,t} = W_2(\mu_s, \mu_t)$ using spectral methods.

**Factor structure**: under the factor model
$r_{i,t} = \sum_k \beta_{i,k} f_{k,t} + \varepsilon_{i,t}$,
when factor volatilities change the cross-sectional distribution shifts
along factor-loading directions. The OT map between consecutive distributions
should be approximately factor-aligned.

## 5. Entropic Regularisation (Sinkhorn)

For large $n$, exact OT is $O(n^3)$. The regularised problem

$$S_\varepsilon(\mu, \nu) = \inf_{\gamma \in \Pi(\mu,\nu)} \int c \, d\gamma + \varepsilon \, \text{KL}(\gamma \| \mu \otimes \nu)$$

is solved by the **Sinkhorn–Knopp algorithm** (matrix scaling), converging
in $O(n^2 / \varepsilon)$ iterations (Cuturi 2013). As $\varepsilon \to 0$,
$S_\varepsilon \to W_2^2$.

## References

- Villani, C. (2008). *Optimal Transport: Old and New*. Springer.
- Agueh, M. & Carlier, G. (2011). Barycenters in the Wasserstein space. *SIAM J. Math. Anal.*
- McCann, R.J. (1997). A convexity principle for interacting gases. *Advances in Mathematics*.
- Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport. *NeurIPS*.
- Flamary, R. et al. (2021). POT: Python Optimal Transport. *JMLR*.
