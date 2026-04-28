# Mathematical Background

This document covers the theoretical foundations underlying the `ot-returns` library.

---

## 1. Optimal Transport

### The Monge Problem

The classical **Monge problem** (Monge 1781) asks: given two measures $\mu$ on $\mathcal{X}$ and $\nu$ on $\mathcal{Y}$, find the map $T: \mathcal{X} \to \mathcal{Y}$ satisfying $T_\#\mu = \nu$ (pushing $\mu$ forward to $\nu$) that minimises the total transport cost:

$$\inf_{T: T_\#\mu = \nu} \int_{\mathcal{X}} c(x, T(x)) \, d\mu(x)$$

This is ill-posed in general (no $T$ may exist) and non-convex.

### The Kantorovich Relaxation

**Kantorovich (1942)** relaxed the Monge problem to a linear programme over **couplings** — joint distributions with prescribed marginals:

$$W_p^p(\mu, \nu) = \inf_{\gamma \in \Pi(\mu,\nu)} \int_{\mathcal{X} \times \mathcal{Y}} c(x,y) \, d\gamma(x,y)$$

where $\Pi(\mu,\nu) = \{\gamma \in \mathcal{P}(\mathcal{X} \times \mathcal{Y}) : (\pi_x)_\#\gamma = \mu,\; (\pi_y)_\#\gamma = \nu\}$ and $c(x,y) = |x-y|^p$.

The infimum is the **$p$-Wasserstein distance** $W_p(\mu,\nu)$.  This is always attained (Kantorovich duality), and when $\mu$ is absolutely continuous Brenier's theorem guarantees a unique optimal $T$.

### Brenier's Theorem

For $p=2$ and $\mu$ absolutely continuous on $\mathbb{R}^d$, the unique optimal transport plan is supported on the graph of a gradient: $T = \nabla \phi$ for some convex $\phi$ (Brenier 1991).  The map $T$ is the unique solution to the **Monge-Ampère equation**:

$$\det(D^2\phi(x)) = \frac{\mu(x)}{\nu(\nabla\phi(x))}$$

In **1-d**, the unique convex gradient is $T = F_\nu^{-1} \circ F_\mu$ — the quantile rearrangement.

### One-Dimensional Closed Form

In 1-d, $\Pi(\mu,\nu)$ has a unique optimal element — the **monotone coupling** $(F_\mu^{-1}(U), F_\nu^{-1}(U))$ for $U \sim \mathrm{Uniform}[0,1]$:

$$W_p^p(\mu,\nu) = \int_0^1 \bigl|F_\mu^{-1}(u) - F_\nu^{-1}(u)\bigr|^p \, du$$

For $p=2$ and two Gaussians $\mathcal{N}(\mu_i, \sigma_i^2)$:

$$W_2^2 = (\mu_1 - \mu_2)^2 + (\sigma_1 - \sigma_2)^2$$

Crucially, $W_2$ penalises differences in **standard deviations**, not variances.  The Wasserstein barycenter of $\mathcal{N}(0,\sigma_1^2)$ and $\mathcal{N}(0,\sigma_2^2)$ has standard deviation $(\sigma_1+\sigma_2)/2$, not $\sqrt{(\sigma_1^2+\sigma_2^2)/2}$.

### Wasserstein Space

$(\mathcal{P}_p(\mathbb{R}^d), W_p)$ is a complete separable metric space.  Its geometry encodes the mass-transport structure:

- **Metrises weak convergence**: $W_p(\mu_n,\mu) \to 0$ iff $\mu_n \rightharpoonup \mu$ and $p$-th moments converge.
- **Geodesics**: the W₂-geodesic between $\mu$ and $\nu$ is the displacement interpolation (Section 3).
- **Comparison to other divergences**: KL divergence is asymmetric, infinite for non-absolutely-continuous pairs, and does not metrize weak convergence.

---

## 2. Wasserstein Barycenters (Agueh & Carlier 2011)

The **Wasserstein barycenter** of $\{\mu_t\}_{t=1}^T$ with weights $\lambda_t \geq 0$, $\sum\lambda_t=1$:

$$\bar\mu = \arg\min_{\nu \in \mathcal{P}_2} \sum_{t=1}^T \lambda_t W_2^2(\nu, \mu_t)$$

**Existence and uniqueness**: guaranteed when at least one $\mu_t$ is absolutely continuous (Agueh & Carlier 2011).

**1-d closed form**: The Fréchet mean in $(\mathcal{P}_2(\mathbb{R}), W_2)$ has quantile function equal to the weighted average of the input quantile functions:

$$F_{\bar\mu}^{-1}(u) = \sum_{t=1}^T \lambda_t F_{\mu_t}^{-1}(u)$$

This remarkable result makes exact computation $O(T \cdot Q)$ where $Q$ is the quantile grid resolution.

**Iterative Bregman projections** (Benamou, Carlier, Cuturi, Nenna & Peyré 2015) extend this to multi-dimensional settings or fixed-support approximations via Sinkhorn.

---

## 3. McCann Displacement Interpolation (McCann 1997)

The **geodesic** in $(\mathcal{P}_2, W_2)$ from $\mu_0$ to $\mu_1$ is:

$$\mu_t = \bigl((1-t)\,\mathrm{Id} + t\,T\bigr)_\# \mu_0, \quad t \in [0,1]$$

where $T$ is the $W_2$-optimal map from $\mu_0$ to $\mu_1$.  Equivalently:

$$F_{\mu_t}^{-1}(u) = (1-t)\,F_{\mu_0}^{-1}(u) + t\,F_{\mu_1}^{-1}(u) \quad (1\text{-d})$$

**Constant speed**: $W_2(\mu_0, \mu_t) = t \cdot W_2(\mu_0, \mu_1)$.

**Generalisation**: piecewise geodesics over $K$ marginals at times $t_1 < \cdots < t_K$ are obtained by applying the McCann interpolant on each sub-interval.

---

## 4. Entropic Regularisation and Sinkhorn

For large $n$, exact OT is $O(n^3)$ via linear programming.  The **entropically regularised** problem

$$S_\varepsilon(\mu,\nu) = \inf_{\gamma \in \Pi(\mu,\nu)} \int c\,d\gamma + \varepsilon\,\mathrm{KL}(\gamma \| \mu\otimes\nu)$$

has a unique solution $\gamma_\varepsilon^* = u \otimes v \cdot K$ where $K_{ij} = e^{-c_{ij}/\varepsilon}$, and the scalings $u, v$ satisfy a fixed-point equation solved by the **Sinkhorn–Knopp algorithm** (matrix scaling).

The **log-domain** iteration (used in this library for numerical stability):

$$\log u_i \leftarrow \log a_i - \log\sum_j K_{ij} v_j, \quad \log v_j \leftarrow \log b_j - \log\sum_i K_{ij} u_i$$

As $\varepsilon \to 0$, $S_\varepsilon \to W_2^2$; larger $\varepsilon$ smooths the transport plan.

---

## 5. Tangent-Space PCA (Bigot, Cazelles & Papadakis 2017)

The Riemannian exponential map at $\bar\mu$ sends a tangent vector $v$ (a function of quantile levels) to:

$$\exp_{\bar\mu}(v) = (F_{\bar\mu}^{-1} + v)_\# \mathrm{Uniform}[0,1]$$

The **log-map** (inverse) of $\mu_t$ at $\bar\mu$ is the tangent vector:

$$v_t(u) = F_{\mu_t}^{-1}(u) - F_{\bar\mu}^{-1}(u)$$

**Wasserstein PCA**: apply standard PCA to the matrix $V \in \mathbb{R}^{T \times Q}$ with rows $v_t$.  The principal directions lie in the tangent space at the barycenter; they capture:

- **PC1**: mean shifts (translational modes, bull vs bear).
- **PC2**: dispersion changes (scale modes, vol regimes).
- **PC3**: skewness/tail changes (shape modes, crash risk).

Reconstructed distributions: $\hat\mu_t = \exp_{\bar\mu}(\hat v_t)$ where $\hat v_t$ is the low-rank approximation.

---

## 6. Factor Decomposition of Transport Maps

Under the cross-sectional factor model $r_i = \sum_k \beta_{ik} f_k + \varepsilon_i$, sorting stocks by return $r_i$ induces a quantile ordering that is approximately the same as sorting by the dominant factor exposure.

Define the **factor-loading profile** at quantile level $u$:
$$\bar\beta_k(u) = \mathbb{E}[\beta_{ik} \mid F_\mu(r_i) \approx u]$$

The displacement field $d(u) = F_\nu^{-1}(u) - F_\mu^{-1}(u)$ is decomposed by OLS:

$$d(u) \approx \sum_k \alpha_k \bar\beta_k(u) + \varepsilon(u)$$

The coefficient $\alpha_k$ measures how much factor $k$ contributed to the distributional shift.  High $R^2$ indicates a factor-driven transition; low $R^2$ indicates idiosyncratic reshuffling.

---

## 7. Connection to Financial Applications

| OT concept | Financial interpretation |
|---|---|
| $W_2(\mu_t, \mu_{t+1})$ | Speed of distributional change |
| OT map $T$ | Return quantile rearrangement between dates |
| Wasserstein barycenter | Canonical "regime distribution" |
| Displacement interpolation | Distribution morphing from calm to crisis |
| Tangent vector $v_t$ | Deviation of today's cross-section from the mean |
| Factor decomposition | Attribution of distributional shift to risk factors |
| CUSUM on $W_2$ | Early warning for regime transitions |
| Permutation test | Formal test for distributional change |

---

## References

- Monge, G. (1781). Mémoire sur la théorie des déblais et des remblais. *Histoire de l'Académie Royale des Sciences*.
- Kantorovich, L. (1942). On the translocation of masses. *Doklady Akademii Nauk USSR*.
- Brenier, Y. (1991). Polar factorization and monotone rearrangement. *Comm. Pure Appl. Math.*
- McCann, R.J. (1997). A convexity principle for interacting gases. *Advances in Mathematics*.
- Villani, C. (2003). *Topics in Optimal Transportation*. AMS.
- Agueh, M. & Carlier, G. (2011). Barycenters in the Wasserstein space. *SIAM J. Math. Anal.* 43(2).
- Benamou, J.-D., Carlier, G., Cuturi, M., Nenna, L. & Peyré, G. (2015). Iterative Bregman projections for regularized transportation problems. *SIAM J. Sci. Comput.*
- Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of optimal transport. *NeurIPS 26*.
- Bigot, J., Cazelles, E. & Papadakis, N. (2017). Geodesic PCA in the Wasserstein space. *ESAIM: Probability and Statistics*.
- Székely, G.J. & Rizzo, M.L. (2004). Testing for equal distributions in high dimension. *InterStat*.
- Peyré, G. & Cuturi, M. (2019). Computational optimal transport. *Foundations and Trends in ML*.
- Flamary, R. et al. (2021). POT: Python Optimal Transport. *JMLR 22*(78).
