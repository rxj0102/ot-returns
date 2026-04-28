# Algorithm Reference

This document describes each major algorithm implemented in `ot-returns`, with pseudocode, complexity, and stability notes.

---

## 1. Wasserstein Distance (1-d, exact)

**Input**: samples $\{x_i\}_{i=1}^n$ with weights $\{a_i\}$, samples $\{y_j\}_{j=1}^m$ with weights $\{b_j\}$, order $p$.

**Fast path** (equal sizes, uniform weights):
1. Sort both arrays: $x_{(1)} \le \cdots \le x_{(n)}$, $y_{(1)} \le \cdots \le y_{(n)}$.
2. $W_p^p = \frac{1}{n}\sum_i |x_{(i)} - y_{(i)}|^p$.

**General path**:
1. Merge support points onto a common sorted grid $z_1 < \cdots < z_{n+m}$.
2. Compute cumulative weight profiles $A(z_k) = \sum_{x_i \le z_k} a_i$, $B(z_k) = \sum_{y_j \le z_k} b_j$.
3. Integrate $\int |A^{-1}(u) - B^{-1}(u)|^p \, du$ exactly over the piecewise-constant intervals.

**Complexity**: $O((n+m)\log(n+m))$ dominated by sorting.  
**Accuracy**: exact (no discretisation error).

---

## 2. Log-Domain Sinkhorn (regularised OT)

**Input**: weights $a \in \Delta^n$, $b \in \Delta^m$, cost matrix $C \in \mathbb{R}^{n\times m}$, regularisation $\varepsilon > 0$.

**Algorithm** (log-domain for stability):
1. Initialise log-scalings $f = \log a$, $g = \mathbf{0}_m$; set $\log K_{ij} = -C_{ij}/\varepsilon$.
2. Repeat until $\|P\mathbf{1}_m - a\|_1 < \delta$:
   - $f_i \leftarrow \log a_i - \log \sum_j \exp(\log K_{ij} + g_j)$ &nbsp;&nbsp; (log-softmax)
   - $g_j \leftarrow \log b_j - \log \sum_i \exp(\log K_{ij} + f_i)$
3. Plan: $P_{ij} = \exp(f_i + \log K_{ij} + g_j)$.
4. Cost: $\langle C, P \rangle$.

**Why log-domain?** For small $\varepsilon$ or large costs, $K_{ij}$ underflows to 0.  Working in log-space prevents this.

**Convergence**: linear rate $O(\exp(-2/\varepsilon))$ — slow for very small $\varepsilon$.  Use $\varepsilon \geq$ median squared cost / 10 as a practical guideline.

**Complexity**: $O(nm)$ per iteration; typically 20–100 iterations.

---

## 3. Wasserstein Barycenter (1-d, exact)

**Input**: $T$ distributions $\{\mu_t\}$ with barycenter weights $\lambda_t$, grid size $Q$.

1. Build midpoint quantile grid: $u_k = (k + 0.5)/Q$ for $k = 0, \ldots, Q-1$.
2. For each $t$: evaluate $q_{tk} = F_{\mu_t}^{-1}(u_k)$ using `quantile_function`.
3. Barycenter quantile values: $\bar q_k = \sum_t \lambda_t q_{tk}$.
4. Return `EmpiricalDistribution(`$\bar q$`)` — an equally-weighted distribution on the $Q$ support points.

**Complexity**: $O(T \cdot Q)$ after sorting each distribution once in $O(n_t \log n_t)$.  
**No discretisation error**: the barycenter quantile function is exact.

**Multi-d extension** (Bregman projections, Benamou et al. 2015): project each $\mu_t$ onto a shared histogram grid, then iterate Sinkhorn barycenter updates.  Implemented via `ot.bregman.barycenter` from POT.

---

## 4. Spectral Clustering on Wasserstein Distance Matrix

**Input**: panel of $T$ distributions.

1. Compute $T \times T$ distance matrix: $D_{st} = W_2(\mu_s, \mu_t)$.  Complexity: $O(T^2 n\log n)$.
2. Build Gaussian affinity: $A_{st} = \exp\bigl(-D_{st}^2 / (2\sigma^2)\bigr)$, where $\sigma = \mathrm{median}_{s \ne t}(D_{st})$ (median heuristic for self-tuning bandwidth).
3. Apply `sklearn.cluster.SpectralClustering` on $A$ with `affinity='precomputed'`.  Internally: normalised graph Laplacian → top-$k$ eigenvectors → $k$-means.
4. Assign each date to a regime label $\{0, \ldots, k-1\}$.

**Alternative backends**:
- **Hierarchical** (`AgglomerativeClustering`, average linkage on $D$): $O(T^2)$, no random initialisation.
- **$k$-medoids** (custom implementation on $D$): more interpretable centres; medoid is the actual date that best represents its cluster.

**Bandwidth sensitivity**: the median heuristic is robust but coarse.  For production use, cross-validate $\sigma$ via silhouette score.

**Scalability**: for $T > 500$, consider approximate $W_2$ (sliced Wasserstein) or random projection of the distance matrix.

---

## 5. Permutation Test (Phipson & Smyth 2010)

**Input**: samples $X = \{x_1,\ldots,x_n\}$, $Y = \{y_1,\ldots,y_m\}$, number of permutations $B$.

1. Compute observed statistic: $T_\mathrm{obs} = W_p(\hat\mu_X, \hat\mu_Y)$.
2. Pool: $Z = X \cup Y$ (length $n+m$).
3. For $b = 1, \ldots, B$:  
   a. Draw uniform random permutation of $\{1,\ldots,n+m\}$.  
   b. Split into $Z_1$ (first $n$ indices) and $Z_2$ (remaining $m$).  
   c. Compute $T_b = W_p(\hat\mu_{Z_1}, \hat\mu_{Z_2})$.
4. $p\text{-value} = \frac{1 + \#\{b : T_b \ge T_\mathrm{obs}\}}{1 + B}$.

**Why `+1` correction?** (Phipson & Smyth 2010): including the observed data as one of the permutations guarantees $p \ge 1/(B+1) > 0$, making the p-value a valid probability even when $T_\mathrm{obs}$ is the maximum.

**Power**: W₂ has greater power than KS against scale changes (W₂ is sensitive to both location and scale) but similar power against pure location shifts.  Energy distance has comparable power with lower computational cost for very large samples.

**Type I error**: under $H_0: \mu=\nu$, the test is exactly level $\alpha$ for any finite $B$.

---

## 6. Transport Map Factor Decomposition

**Input**: source $\mu$, target $\nu$, factor-loading matrix $\beta \in \mathbb{R}^{N \times K}$, grid size $Q$.

1. Build quantile grid: $u_k = (k+0.5)/Q$.
2. Compute displacement: $d_k = F_\nu^{-1}(u_k) - F_\mu^{-1}(u_k)$.
3. Sort source stocks by return to get quantile ordering; assign each stock $i$ quantile level $p_i = (\mathrm{rank}(r_i) - 0.5)/N$.
4. Interpolate factor loadings onto the grid: $\bar\beta_k(u_j) = \mathrm{interp}(u_j;\, p_i,\, \beta_{ik})$.
5. Form design matrix $B \in \mathbb{R}^{Q \times K}$ with $B_{jk} = \bar\beta_k(u_j)$.
6. Solve OLS: $\hat\alpha = (B^\top B)^{-1} B^\top d$.
7. $R^2 = 1 - \|\hat\varepsilon\|^2 / \|d\|^2$ where $\hat\varepsilon = d - B\hat\alpha$.

**Interpretation**: $\hat\alpha_k$ is the intensity of factor $k$'s contribution; high $R^2$ means the shift was factor-driven.  Works best when the factor signal dominates idiosyncratic noise in the sorting step.

---

## 7. Transport PCA (Bigot et al. 2017)

**Input**: panel $\{\mu_t\}_{t=1}^T$, grid size $Q$, $K$ components.

1. Compute Wasserstein barycenter $\bar\mu$ via 1-d exact formula.
2. Evaluate $Q_\mathrm{bar}(u) = F_{\bar\mu}^{-1}(u)$ on the midpoint grid.
3. For each $t$: $V[t, :] = F_{\mu_t}^{-1}(u) - Q_\mathrm{bar}(u)$.  (log-map)
4. PCA on $V \in \mathbb{R}^{T \times Q}$: compute top-$K$ left singular vectors via SVD.
5. Reconstruct: $\hat V[t, :] = \mathrm{PC\,projection}$; $\hat F_t^{-1}(u) = Q_\mathrm{bar}(u) + \hat V[t, :]$.
6. Sort each reconstructed quantile function to ensure monotonicity.

**Complexity**: $O(TQ^2)$ for SVD if $T < Q$; $O(T^2 Q)$ otherwise.  Typically $Q=200$ and $T \le 500$ are well within range.

**Interpretation**: the $i$-th principal component $e_i(u)$ is a tangent direction at $\bar\mu$.  Adding $\lambda e_i$ to $Q_\mathrm{bar}$ gives a family of distributions varying along that mode.

---

## 8. CUSUM Distributional Monitoring

**Input**: panel $\{\mu_t\}$, reference period $[t_0, t_\mathrm{ref}]$, order $p$.

1. Pool reference samples to form $\hat\mu_\mathrm{ref}$.
2. Estimate drift: $\hat\delta = \frac{1}{|\mathcal{R}|} \sum_{t \in \mathcal{R}} W_p(\mu_t, \hat\mu_\mathrm{ref})$.
3. For $t > t_\mathrm{ref}$:
   $$S_t = S_{t-1} + \bigl(W_p(\mu_t, \hat\mu_\mathrm{ref}) - \hat\delta\bigr), \quad S_{t_\mathrm{ref}} = 0$$
4. Signal when $S_t > h$ (threshold $h$ calibrated by simulation under $H_0$).

Rising $S_t$ indicates a sustained shift above the in-sample baseline.  Flat $S_t$ indicates the process is stationary around its reference distribution.

---

## 9. Numerical Stability Notes

| Issue | Symptom | Fix |
|---|---|---|
| Sinkhorn divergence | NaN or Inf transport plan | Use log-domain; increase $\varepsilon$ |
| Zero-weight bins | Division by zero in barycenter | Add small floor $10^{-10}$ before normalising |
| Spectral clustering degeneracy | All dates in one cluster | Widen bandwidth $\sigma$; check $D$ for near-identical rows |
| OLS collinearity in factor decomp | Large $\|\\hat\alpha\|$ | Regularise with ridge if condition number of $B^\top B$ is large |
| Quantile interpolation boundary | Quantiles outside training range | Clamp to boundary values (already done in `quantile_function`) |
