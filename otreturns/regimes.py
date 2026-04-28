"""
Wasserstein-based regime detection for cross-sectional return distributions.

Algorithm:
1. Compute the T×T pairwise Wasserstein distance matrix D from DistributionPanel.
2. Convert D to affinity A = exp(-D² / (2σ²)), σ = median(D).
3. Apply spectral / hierarchical / Wasserstein-k-means clustering.
4. Assign each date to a regime; optionally compute barycenters, transitions,
   silhouette scores, and optimal-K analysis.

The key advantage over return-based regime models (HMM on index returns) is
that we use the full cross-sectional distribution — capturing shifts in
skewness, tail heaviness, and multi-modality, not just mean/variance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_samples
import scipy.cluster.hierarchy as sch

from otreturns.distributions import EmpiricalDistribution, DistributionPanel
from otreturns.distances import wasserstein_1d
from otreturns.barycenters import wasserstein_barycenter_1d


# ---------------------------------------------------------------------------
# WassersteinRegimeDetector
# ---------------------------------------------------------------------------

class WassersteinRegimeDetector:
    """
    Detect market regimes from the evolution of cross-sectional distributions.

    Method:
    1. Compute the pairwise Wasserstein distance matrix D between all dates.
    2. Apply spectral / hierarchical / Wasserstein-k-means clustering on D.
    3. Assign regime labels to each date.

    The key insight: dates with similar cross-sectional structure will be
    close in Wasserstein distance, forming natural clusters = regimes.
    """

    def __init__(
        self,
        n_regimes: int = 3,
        clustering_method: str = "spectral",
        distance_method: str = "1d",
        window: int = None,
    ):
        """
        Args:
            n_regimes:          number of regimes to detect
            clustering_method:  'spectral', 'hierarchical', or
                                'kmeans_wasserstein'
            distance_method:    '1d' (fast quantile formula) or 'sinkhorn'
            window:             if not None, use rolling-window pooled
                                distributions (smoother but slower)
        """
        if n_regimes < 2:
            raise ValueError(f"n_regimes must be >= 2, got {n_regimes}")
        self.n_regimes = n_regimes
        self.clustering_method = clustering_method
        self.distance_method = distance_method
        self.window = window

        self._labels: np.ndarray | None = None
        self._D: np.ndarray | None = None
        self._panel: DistributionPanel | None = None
        self._barycenters: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, panel: DistributionPanel) -> "WassersteinRegimeDetector":
        """
        Fit the regime detector.

        Steps:
        1. Build (possibly rolling-window) distributions for each date.
        2. Compute distance matrix D_{st} = W_2(μ_s, μ_t).
        3. Convert to affinity A_{st} = exp(-D²_{st} / (2σ²)),
           where σ = median(D) (median heuristic for kernel bandwidth).
        4. Apply clustering to A (spectral) or D (hierarchical).
        5. Store regime labels and compute barycenters.
        """
        self._panel = panel
        dists = self._get_distributions(panel)
        self._D = _pairwise_wasserstein(dists)
        self._labels = self._cluster(self._D)
        self._barycenters = None   # computed lazily
        return self

    def predict(self, new_distribution: EmpiricalDistribution) -> int:
        """
        Assign a regime label to a new distribution.

        Compares W_2 distance to each regime barycenter; returns the label
        of the nearest barycenter.
        """
        self._require_fitted()
        barycenters = self.regime_barycenters
        best_label, best_dist = -1, float("inf")
        for label, bary in barycenters.items():
            d = wasserstein_1d(new_distribution, bary, p=2)
            if d < best_dist:
                best_dist = d
                best_label = label
        return int(best_label)

    def fit_predict(self, panel: DistributionPanel) -> np.ndarray:
        """Fit and return regime labels (integer array, length = len(dates))."""
        self.fit(panel)
        return self._labels.copy()

    @property
    def regime_barycenters(self) -> dict:
        """Wasserstein barycenters for each detected regime (computed once)."""
        self._require_fitted()
        if self._barycenters is None:
            dists = self._get_distributions(self._panel)
            labels = self._labels
            unique = np.unique(labels)
            self._barycenters = {}
            for lbl in unique:
                mask = labels == lbl
                group = [dists[i] for i in range(len(dists)) if mask[i]]
                self._barycenters[int(lbl)] = wasserstein_barycenter_1d(group)
        return self._barycenters

    @property
    def distance_matrix(self) -> np.ndarray:
        """The computed pairwise Wasserstein distance matrix (T × T)."""
        self._require_fitted()
        return self._D.copy()

    def transition_matrix(self) -> np.ndarray:
        """
        Empirical regime transition matrix.

        P_{ij} = P(regime_{t+1} = j | regime_t = i)
        Estimated from the regime label sequence by counting consecutive pairs.

        Returns:
            (K, K) row-stochastic matrix; rows with no transitions are uniform.
        """
        self._require_fitted()
        K = self.n_regimes
        P = np.zeros((K, K))
        for t in range(len(self._labels) - 1):
            i, j = int(self._labels[t]), int(self._labels[t + 1])
            P[i, j] += 1
        # Normalize rows
        row_sums = P.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)  # avoid /0
        return P / row_sums

    def silhouette_analysis(self) -> dict:
        """
        Compute Wasserstein silhouette scores for cluster quality.

        For each date t with regime label k:
            s_t = (b_t - a_t) / max(a_t, b_t)

        where:
            a_t = mean W_2 distance to other dates in the same regime
            b_t = min over k' != k of mean W_2 distance to dates in k'

        Returns:
            dict with 'scores' (per-date array), 'mean_score' (float),
            'per_regime_scores' (dict label → mean score)
        """
        self._require_fitted()
        D = self._D
        labels = self._labels
        n = len(labels)
        unique = np.unique(labels)

        scores = np.empty(n)
        for i in range(n):
            k = labels[i]
            same = [j for j in range(n) if labels[j] == k and j != i]
            a_i = float(D[i, same].mean()) if same else 0.0

            b_i = float("inf")
            for k2 in unique:
                if k2 == k:
                    continue
                other = [j for j in range(n) if labels[j] == k2]
                if other:
                    b_i = min(b_i, float(D[i, other].mean()))

            denom = max(a_i, b_i)
            scores[i] = (b_i - a_i) / denom if denom > 0 else 0.0

        per_regime = {}
        for lbl in unique:
            mask = labels == lbl
            per_regime[int(lbl)] = float(scores[mask].mean())

        return {
            "scores": scores,
            "mean_score": float(scores.mean()),
            "per_regime_scores": per_regime,
        }

    def optimal_n_regimes(
        self,
        panel: DistributionPanel,
        max_regimes: int = 8,
    ) -> dict:
        """
        Determine the optimal number of regimes.

        For k = 2, ..., max_regimes:
          1. Fit a detector with k regimes.
          2. Compute the mean Wasserstein silhouette score.

        Also computes a simple gap statistic: the score relative to a null
        (random permutation of the same distance matrix, averaged over 10
        random shuffles).

        Returns:
            dict with:
                'k_values':          list of k tested
                'silhouette_scores': silhouette score per k
                'gap_statistics':    gap = score(k) - null_score(k) per k
                'recommended_k':     k with the highest silhouette score
        """
        # Compute the distance matrix once for all k
        dists = self._get_distributions(panel)
        D = _pairwise_wasserstein(dists)

        k_values = list(range(2, max_regimes + 1))
        sil_scores: list[float] = []
        gap_stats: list[float] = []

        rng = np.random.default_rng(0)

        for k in k_values:
            det = WassersteinRegimeDetector(
                n_regimes=k,
                clustering_method=self.clustering_method,
                distance_method=self.distance_method,
            )
            det._panel = panel
            det._D = D
            det._labels = det._cluster(D)
            sil = det.silhouette_analysis()["mean_score"]
            sil_scores.append(sil)

            # Null: permute rows AND columns of D (preserves symmetry)
            null_scores = []
            for _ in range(10):
                perm = rng.permutation(len(D))
                D_null = D[np.ix_(perm, perm)]
                det_null = WassersteinRegimeDetector(
                    n_regimes=k,
                    clustering_method=self.clustering_method,
                )
                det_null._panel = panel
                det_null._D = D_null
                det_null._labels = det_null._cluster(D_null)
                null_scores.append(det_null.silhouette_analysis()["mean_score"])
            gap_stats.append(sil - float(np.mean(null_scores)))

        best_idx = int(np.argmax(sil_scores))
        return {
            "k_values": k_values,
            "silhouette_scores": sil_scores,
            "gap_statistics": gap_stats,
            "recommended_k": k_values[best_idx],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        if self._labels is None:
            raise RuntimeError("Call fit() before using this method.")

    def _get_distributions(self, panel: DistributionPanel) -> list:
        """Return one EmpiricalDistribution per date (rolling-window if set)."""
        dates = panel.dates
        if self.window is None:
            return [panel[d] for d in dates]
        # Rolling window: pool last `window` dates up to and including t
        result = []
        for i, d in enumerate(dates):
            start = max(0, i - self.window + 1)
            window_dates = dates[start: i + 1]
            pooled = np.concatenate([panel[dd].samples for dd in window_dates])
            result.append(EmpiricalDistribution(pooled))
        return result

    def _cluster(self, D: np.ndarray) -> np.ndarray:
        """Dispatch to the chosen clustering algorithm."""
        method = self.clustering_method
        if method == "spectral":
            return _spectral_cluster(D, self.n_regimes)
        elif method == "hierarchical":
            return _hierarchical_cluster(D, self.n_regimes)
        elif method == "kmeans_wasserstein":
            return _wasserstein_kmeans(D, self.n_regimes)
        else:
            raise ValueError(
                f"Unknown clustering_method '{method}'. "
                "Use 'spectral', 'hierarchical', or 'kmeans_wasserstein'."
            )


# ---------------------------------------------------------------------------
# Rolling regime detection (lookahead-free)
# ---------------------------------------------------------------------------

def rolling_regime_detection(
    panel: DistributionPanel,
    lookback: int = 126,
    n_regimes: int = 3,
    step: int = 21,
) -> pd.DataFrame:
    """
    Online/rolling regime detection — no lookahead bias.

    At each date t (every `step` dates), fit a WassersteinRegimeDetector on
    data from [t - lookback, t] only, then assign the label for date t and
    record a confidence score.

    Confidence:
        1 - (distance to assigned barycenter) / (distance to furthest barycenter)

    Args:
        panel:    DistributionPanel
        lookback: number of historical dates to use per window
        n_regimes: number of regimes
        step:     number of dates between successive detections

    Returns:
        DataFrame with columns ['date', 'regime', 'confidence']
    """
    dates = panel.dates
    rows = []

    for i in range(lookback, len(dates), step):
        window_dates = dates[i - lookback: i + 1]

        # Build sub-panel for the lookback window
        sub_dists = {d: panel[d] for d in window_dates}
        sub_panel = DistributionPanel(sub_dists, window_dates)

        detector = WassersteinRegimeDetector(n_regimes=n_regimes)
        detector.fit(sub_panel)

        # Label for the last date in the window
        current_date = window_dates[-1]
        current_dist = panel[current_date]
        regime = detector.predict(current_dist)

        # Confidence
        barycenters = detector.regime_barycenters
        dists_to_bary = {
            lbl: wasserstein_1d(current_dist, bary, p=2)
            for lbl, bary in barycenters.items()
        }
        d_assigned = dists_to_bary[regime]
        d_max = max(dists_to_bary.values())
        confidence = 1.0 - (d_assigned / d_max) if d_max > 0 else 1.0

        rows.append({
            "date": current_date,
            "regime": regime,
            "confidence": float(confidence),
        })

    return pd.DataFrame(rows, columns=["date", "regime", "confidence"])


# ---------------------------------------------------------------------------
# Legacy functional API (kept for backward compatibility with stub callers)
# ---------------------------------------------------------------------------

def compute_distance_matrix(
    panel: DistributionPanel,
    p: int = 2,
    n_quantiles: int = 500,
) -> np.ndarray:
    """Compute the T×T pairwise W_p distance matrix for all dates."""
    dists = [panel[d] for d in panel.dates]
    return _pairwise_wasserstein(dists, p=p)


def detect_regimes(
    panel: DistributionPanel,
    n_regimes: int = 3,
    p: int = 2,
    smoothing_window: int = 5,
    seed: int = None,
) -> np.ndarray:
    """Detect distributional regimes via spectral clustering."""
    det = WassersteinRegimeDetector(n_regimes=n_regimes, distance_method="1d")
    return det.fit_predict(panel)


def regime_transition_matrix(labels: np.ndarray) -> np.ndarray:
    """Estimate the empirical transition probability matrix."""
    labels = np.asarray(labels, dtype=int)
    K = int(labels.max()) + 1
    P = np.zeros((K, K))
    for t in range(len(labels) - 1):
        P[labels[t], labels[t + 1]] += 1
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return P / row_sums


# ---------------------------------------------------------------------------
# Clustering back-ends
# ---------------------------------------------------------------------------

def _pairwise_wasserstein(dists: list, p: int = 2) -> np.ndarray:
    """Compute symmetric T×T distance matrix via lower-triangular enumeration."""
    T = len(dists)
    D = np.zeros((T, T))
    for i in range(T):
        for j in range(i + 1, T):
            d = wasserstein_1d(dists[i], dists[j], p=p)
            D[i, j] = d
            D[j, i] = d
    return D


def _affinity_from_distance(D: np.ndarray) -> np.ndarray:
    """Gaussian affinity with median-heuristic bandwidth."""
    flat = D[D > 0]
    sigma = float(np.median(flat)) if len(flat) > 0 else 1.0
    sigma = max(sigma, 1e-8)
    return np.exp(-D ** 2 / (2.0 * sigma ** 2))


def _spectral_cluster(D: np.ndarray, k: int) -> np.ndarray:
    A = _affinity_from_distance(D)
    sc = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=0,
        n_init=10,
    )
    return sc.fit_predict(A)


def _hierarchical_cluster(D: np.ndarray, k: int) -> np.ndarray:
    # Ward linkage requires Euclidean; use average linkage on the distance matrix
    hc = AgglomerativeClustering(
        n_clusters=k,
        metric="precomputed",
        linkage="average",
    )
    return hc.fit_predict(D)


def _wasserstein_kmeans(D: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
    """
    K-medoids on the precomputed distance matrix.

    Uses the Wasserstein distance directly: each cluster center is the
    medoid (the date minimising total within-cluster distance).
    """
    rng = np.random.default_rng(0)
    n = len(D)
    # Initialise medoids with k-means++ style on the distance matrix
    medoids = [int(rng.integers(n))]
    for _ in range(k - 1):
        # Distance to nearest medoid for each point
        min_d = np.min(D[:, medoids], axis=1)
        probs = min_d ** 2
        probs /= probs.sum()
        medoids.append(int(rng.choice(n, p=probs)))

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign
        new_labels = np.argmin(D[:, medoids], axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update medoids
        for c in range(k):
            members = np.where(labels == c)[0]
            if len(members) == 0:
                continue
            intra = D[np.ix_(members, members)].sum(axis=1)
            medoids[c] = int(members[np.argmin(intra)])

    return labels
