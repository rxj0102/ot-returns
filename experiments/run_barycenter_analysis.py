"""
Experiment: Wasserstein barycenters and displacement interpolation.

Computes regime barycenters on a synthetic regime-switching panel,
then visualises the W₂-geodesic (displacement interpolation) between
the "calm" and "crisis" regime barycenters.

Usage:
    python experiments/run_barycenter_analysis.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic import generate_regime_switching_panel
from otreturns.distributions import DistributionPanel
from otreturns.regimes import WassersteinRegimeDetector
from otreturns.barycenters import wasserstein_barycenter_1d, regime_barycenters
from otreturns.interpolation import interpolation_path
from otreturns.distances import wasserstein_1d


def main():
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    figures_dir = Path(cfg["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Barycenter Analysis & Displacement Interpolation")
    print("=" * 60)

    n_dates, n_stocks = 120, 300
    print(f"\nGenerating panel (n_dates={n_dates}, n_stocks={n_stocks})...")
    panel_df, true_labels = generate_regime_switching_panel(
        n_dates=n_dates, n_stocks=n_stocks, seed=cfg["data"]["seed"]
    )
    panel = DistributionPanel.from_panel(panel_df, min_stocks=50,
                                         winsorize=cfg["data"]["winsorize"])
    dates = panel.dates
    T = len(dates)
    true_labels = true_labels[:T]

    # ---------------------------------------------------------------- detect regimes
    print("Detecting regimes...")
    detector = WassersteinRegimeDetector(n_regimes=3)
    pred_labels = detector.fit_predict(panel)

    # ---------------------------------------------------------------- barycenters
    print("Computing regime barycenters...")
    bary_dict = regime_barycenters(panel, pred_labels, method="1d", n_support=300)

    u = np.linspace(0.01, 0.99, 300)
    palette = ["steelblue", "darkorange", "forestgreen"]

    for lbl, b in sorted(bary_dict.items()):
        m = b.moments(2)
        print(f"  Regime {lbl}: mean={m['mean']:.4f}  std={np.sqrt(m['variance']):.4f}  n_support={b.n}")

    # W2 distances between barycenters
    labels_sorted = sorted(bary_dict.keys())
    print("\nW₂ distances between regime barycenters:")
    for i in labels_sorted:
        for j in labels_sorted:
            if j > i:
                d = wasserstein_1d(bary_dict[i], bary_dict[j], p=2)
                print(f"  W₂(regime {i}, regime {j}) = {d:.4f}")

    # ---------------------------------------------------------------- interpolation
    # Choose the two most different barycenters by W2 distance
    pairs = [(i, j, wasserstein_1d(bary_dict[i], bary_dict[j]))
             for i in labels_sorted for j in labels_sorted if j > i]
    src_lbl, tgt_lbl, _ = max(pairs, key=lambda x: x[2])

    src_bary = bary_dict[src_lbl]
    tgt_bary = bary_dict[tgt_lbl]
    n_steps = 10
    path = interpolation_path(src_bary, tgt_bary, n_steps=n_steps, n_support=300)
    print(f"\nInterpolation path: regime {src_lbl} → regime {tgt_lbl} ({n_steps} steps)")
    for k, dist in enumerate(path):
        t = k / n_steps
        m = dist.moments(2)
        print(f"  t={t:.1f}: mean={m['mean']:.4f}  std={np.sqrt(m['variance']):.4f}")

    # ---------------------------------------------------------------- plots
    # 1. Barycenter quantile functions
    fig, ax = plt.subplots(figsize=(8, 5))
    for lbl, b in sorted(bary_dict.items()):
        ax.plot(b.quantile_function(u), u, color=palette[lbl % len(palette)],
                lw=2.5, label=f"Regime {lbl}")
    ax.set_xlabel("Return"); ax.set_ylabel("Quantile u")
    ax.set_title("Regime barycenter quantile functions")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "barycenter_quantiles.png", dpi=120)
    plt.close(fig)

    # 2. Displacement interpolation — quantile functions
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = cm.coolwarm
    for k, dist in enumerate(path):
        t = k / n_steps
        ax.plot(dist.quantile_function(u), u, color=cmap(t), lw=1.8, alpha=0.85)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Interpolation parameter t")
    ax.set_xlabel("Return"); ax.set_ylabel("Quantile u")
    ax.set_title(f"W₂-geodesic: regime {src_lbl} → regime {tgt_lbl}")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "interpolation_path_quantiles.png", dpi=120)
    plt.close(fig)

    # 3. Moment evolution along path
    t_vals = np.linspace(0, 1, len(path))
    means = [d.moments(2)["mean"] for d in path]
    stds  = [np.sqrt(d.moments(2)["variance"]) for d in path]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(t_vals, means, "o-", color="steelblue")
    axes[0].set_xlabel("t"); axes[0].set_ylabel("Mean")
    axes[0].set_title("Mean along interpolation path")
    axes[0].grid(alpha=0.3)

    axes[1].plot(t_vals, stds, "o-", color="darkorange")
    axes[1].set_xlabel("t"); axes[1].set_ylabel("Std dev")
    axes[1].set_title("Std dev along interpolation path (linear in W₂)")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "interpolation_moments.png", dpi=120)
    plt.close(fig)

    print(f"\nFigures → {figures_dir}")


if __name__ == "__main__":
    main()
