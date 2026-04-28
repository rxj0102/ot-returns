"""
Experiment: Wasserstein-based regime detection on a synthetic panel.

Usage:
    python experiments/run_regime_detection.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic import generate_regime_switching_panel
from otreturns.distributions import DistributionPanel
from otreturns.regimes import WassersteinRegimeDetector
from sklearn.metrics import adjusted_rand_score


def main():
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    figures_dir = Path(cfg["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg["output_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Wasserstein Regime Detection Experiment")
    print("=" * 60)

    n_dates, n_stocks = 120, 300
    print(f"\nGenerating regime-switching panel  (n_dates={n_dates}, n_stocks={n_stocks})...")
    panel_df, true_labels = generate_regime_switching_panel(
        n_dates=n_dates, n_stocks=n_stocks, seed=cfg["data"]["seed"]
    )

    panel = DistributionPanel.from_panel(
        panel_df, min_stocks=50, winsorize=cfg["data"]["winsorize"]
    )
    dates = panel.dates
    T = len(dates)
    true_labels = true_labels[:T]
    print(f"Panel: {T} dates  |  regime sizes: {np.bincount(true_labels).tolist()}")

    # ------------------------------------------------------------------ fit
    print("\nFitting WassersteinRegimeDetector (spectral, k=3)...")
    detector = WassersteinRegimeDetector(n_regimes=3, clustering_method="spectral")
    pred_labels = detector.fit_predict(panel)

    ari = adjusted_rand_score(true_labels, pred_labels)
    print(f"Adjusted Rand Index : {ari:.4f}  (>0.7 = good recovery)")

    # Transition matrix
    P = detector.transition_matrix()
    print(f"\nTransition matrix (rows = from, cols = to):")
    for row in P:
        print("  " + "  ".join(f"{v:.3f}" for v in row))

    # Silhouette
    sil = detector.silhouette_analysis()
    print(f"\nSilhouette scores:")
    print(f"  mean  : {sil['mean_score']:.4f}")
    for lbl, sc in sorted(sil["per_regime_scores"].items()):
        print(f"  k={lbl} : {sc:.4f}")

    # Optimal k
    print("\nOptimal-k sweep (k = 2..5)...")
    opt = detector.optimal_n_regimes(panel, max_regimes=5)
    for k, s, g in zip(opt["k_values"], opt["silhouette_scores"], opt["gap_statistics"]):
        tag = " <-- recommended" if k == opt["recommended_k"] else ""
        print(f"  k={k}: silhouette={s:.4f}  gap={g:.4f}{tag}")

    # ---------------------------------------------------------------- plots
    colors_true = true_labels
    colors_pred = pred_labels

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].scatter(range(T), true_labels, c=colors_true, cmap="tab10",
                    vmin=0, vmax=9, s=9, alpha=0.85)
    axes[0].set_ylabel("Regime"); axes[0].set_title("True regime labels")

    axes[1].scatter(range(T), pred_labels, c=colors_pred, cmap="tab10",
                    vmin=0, vmax=9, s=9, alpha=0.85)
    axes[1].set_ylabel("Regime")
    axes[1].set_title(f"Detected regimes  (ARI = {ari:.3f})")
    axes[1].set_xlabel("Date index")
    plt.tight_layout()
    fig.savefig(figures_dir / "regime_assignments.png", dpi=120)
    plt.close(fig)

    # Barycenter quantile functions
    bary = detector.regime_barycenters
    u = np.linspace(0.01, 0.99, 300)
    fig, ax = plt.subplots(figsize=(8, 5))
    palette = ["steelblue", "darkorange", "forestgreen", "crimson", "purple"]
    for lbl, b in sorted(bary.items()):
        ax.plot(b.quantile_function(u), u, color=palette[lbl % len(palette)],
                lw=2.5, label=f"Regime {lbl}")
    ax.set_xlabel("Return"); ax.set_ylabel("Quantile u")
    ax.set_title("Regime barycenter quantile functions")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "regime_barycenters.png", dpi=120)
    plt.close(fig)

    # Distance matrix heatmap
    D = detector.distance_matrix
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(D, aspect="auto", cmap="viridis_r")
    plt.colorbar(im, ax=ax, label="W₂ distance")
    ax.set_title("Pairwise Wasserstein distance matrix")
    ax.set_xlabel("Date"); ax.set_ylabel("Date")
    plt.tight_layout()
    fig.savefig(figures_dir / "distance_matrix.png", dpi=120)
    plt.close(fig)

    pd.DataFrame({"date_index": range(T), "true": true_labels,
                  "detected": pred_labels}).to_csv(
        results_dir / "regime_assignments.csv", index=False)
    print(f"\nFigures → {figures_dir}")
    print(f"Results → {results_dir}")


if __name__ == "__main__":
    main()
