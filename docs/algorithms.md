# Algorithm Reference

## Wasserstein Distance (1-d, exact)

**Input**: samples $\{x_i\}_{i=1}^n$ with weights $\{a_i\}$,
           samples $\{y_j\}_{j=1}^m$ with weights $\{b_j\}$.

1. Sort both sets by value.
2. Merge support points onto a common sorted grid.
3. Compute cumulative weight profiles $A(x)$, $B(y)$.
4. Integrate $|A^{-1}(u) - B^{-1}(u)|^p$ over $u \in [0,1]$ via the
   merged grid (piecewise-constant integrand).

**Complexity**: $O((n+m)\log(n+m))$.

## Sinkhorn–Knopp (regularised OT)

**Input**: weight vectors $a \in \Delta^n$, $b \in \Delta^m$,
           cost matrix $C \in \mathbb{R}^{n \times m}$,
           regularisation $\varepsilon > 0$.

1. Initialise $u = \mathbf{1}_n$, $K = \exp(-C/\varepsilon)$.
2. Repeat until convergence:
   - $u \leftarrow a / (K v)$
   - $v \leftarrow b / (K^\top u)$
3. Optimal plan: $\gamma^* = \text{diag}(u) K \text{diag}(v)$.
4. Cost: $\langle C, \gamma^* \rangle$.

**Complexity per iteration**: $O(nm)$. Use log-domain for numerical stability.

## Wasserstein Barycenter (1-d)

**Input**: $T$ distributions $\{\mu_t\}$ with barycenter weights $\lambda_t$.

1. Build uniform quantile grid $u_1, \ldots, u_Q \in (0,1)$.
2. Evaluate each $F_{\mu_t}^{-1}(u_k)$ by interpolation.
3. Barycenter quantile: $F_{\bar\mu}^{-1}(u_k) = \sum_t \lambda_t F_{\mu_t}^{-1}(u_k)$.
4. Return as `EmpiricalDistribution` with $Q$ support points and uniform weights.

**Complexity**: $O(T \cdot Q)$ (after the $O(n \log n)$ sorting per distribution).

## Regime Detection (Spectral Clustering on $W_2$)

1. Compute $T \times T$ distance matrix $D$ (pairwise $W_2$).
2. Build affinity: $A_{st} = \exp(-D_{st}^2 / (2\sigma^2))$,
   where $\sigma = \text{median}(D)$ (self-tuning bandwidth).
3. Normalised graph Laplacian: $L = I - D^{-1/2} A D^{-1/2}$.
4. Compute the top $K$ eigenvectors of $L$.
5. $K$-means on the eigenvector rows.
6. Smooth label sequence with sliding-window majority vote.

**Complexity**: $O(T^2)$ for the distance matrix; $O(T^3)$ for eigendecomposition
(use randomised SVD for large $T$).

## Transport Map Decomposition

**Input**: Quantile maps $x_k = F_\mu^{-1}(u_k)$,
           $T_k = F_\nu^{-1}(u_k)$ (the transport map values),
           factor-loading matrix $\beta \in \mathbb{R}^{N \times K}$.

Displacement: $d_k = T_k - x_k$.

Projection onto factor directions (constructed from $\beta$):
1. Compute factor quantile functions $\phi_k(u) = F_{\beta_k r}^{-1}(u)$
   where $\beta_k r$ is the cross-sectional factor-$k$ return.
2. Ordinary least squares: $\alpha = (\Phi^\top \Phi)^{-1} \Phi^\top d$
   where $\Phi \in \mathbb{R}^{Q \times K}$ has columns $\phi_k$.
3. Residual: $\epsilon = d - \Phi \alpha$.
4. $R^2 = 1 - \|\epsilon\|^2 / \|d\|^2$.

## Two-Sample Permutation Test

**Input**: Samples $X = \{x_1,\ldots,x_n\}$, $Y = \{y_1,\ldots,y_m\}$.

1. Compute observed statistic: $W_{\text{obs}} = W_2(\hat\mu_X, \hat\mu_Y)$.
2. Pool $Z = X \cup Y$.
3. For $b = 1, \ldots, B$:
   - Draw permutation: split $Z$ randomly into groups of size $n$ and $m$.
   - Compute $W_b = W_2(\hat\mu_{Z_1}, \hat\mu_{Z_2})$.
4. $p\text{-value} = \#\{b : W_b \geq W_{\text{obs}}\} / B$.

**Note**: Under the null $H_0: \mu = \nu$, all $(n+m)!/(n!\,m!)$ splits
are equally likely.
