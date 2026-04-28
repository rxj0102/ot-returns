"""
Experiment: Displacement interpolation visualisation.

Generates a sequence of frames showing the cross-sectional return
distribution "morphing" between a calm regime and a crisis regime
along the W₂-geodesic, and saves them as individual PNGs plus an
HTML figure (static multi-panel grid, no dependencies beyond matplotlib).

Usage:
    python experiments/run_interpolation_visualization.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import numpy as np
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic import generate_regime_switching_panel
from otreturns.distributions import EmpiricalDistribution, DistributionPanel
from otreturns.regimes import WassersteinRegimeDetector
from otreturns.barycenters import regime_barycenters
from otreturns.interpolation import interpolation_path, multi_marginal_interpolation
from otreturns.distances import wasserstein_1d


def _kde(samples, x_grid, bw=None):
    """Gaussian KDE for visualisation."""
    if bw is None:
        bw = 1.06 * np.std(samples) * len(samples) ** (-0.2)
    bw = max(bw, 1e-6)
    diff = (x_grid[:, None] - samples[None, :]) / bw
    return np.mean(np.exp(-0.5 * diff ** 2), axis=1) / (bw * np.sqrt(2 * np.pi))


def main():
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    figures_dir = Path(cfg["figures_dir"])
    frames_dir  = figures_dir / "interpolation_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Displacement Interpolation Visualisation")
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

    # Detect regimes and compute barycenters
    print("Detecting regimes and computing barycenters...")
    detector = WassersteinRegimeDetector(n_regimes=3)
    pred_labels = detector.fit_predict(panel)
    bary_dict = regime_barycenters(panel, pred_labels, method="1d", n_support=300)

    # Pick the two most distant barycenters
    labels_sorted = sorted(bary_dict.keys())
    pairs = [(i, j, wasserstein_1d(bary_dict[i], bary_dict[j]))
             for i in labels_sorted for j in labels_sorted if j > i]
    src_lbl, tgt_lbl, w2_total = max(pairs, key=lambda x: x[2])
    print(f"Interpolating regime {src_lbl} → regime {tgt_lbl}  "
          f"(W₂ = {w2_total:.4f})")

    src_bary = bary_dict[src_lbl]
    tgt_bary = bary_dict[tgt_lbl]

    n_steps = 12
    path = interpolation_path(src_bary, tgt_bary, n_steps=n_steps, n_support=300)

    # x grid for KDE
    all_samples = np.concatenate([src_bary.samples, tgt_bary.samples])
    x_lo = np.percentile(all_samples, 1) - 0.5
    x_hi = np.percentile(all_samples, 99) + 0.5
    x_grid = np.linspace(x_lo, x_hi, 400)
    u = np.linspace(0.01, 0.99, 300)

    # ---------------------------------------------------------------- frames
    print(f"Saving {n_steps + 1} interpolation frames...")
    cmap = cm.RdYlGn_r
    for k, dist in enumerate(path):
        t = k / n_steps
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))

        # PDF (KDE)
        dens = _kde(dist.samples, x_grid)
        axes[0].fill_between(x_grid, dens, alpha=0.35, color=cmap(t))
        axes[0].plot(x_grid, dens, color=cmap(t), lw=2)
        axes[0].set_xlabel("Cross-sectional return"); axes[0].set_ylabel("Density")
        axes[0].set_title(f"PDF at t={t:.2f}  (W₂-geodesic)")
        axes[0].set_xlim(x_lo, x_hi); axes[0].grid(alpha=0.3)

        # Quantile function
        q_vals = dist.quantile_function(u)
        axes[1].plot(u, q_vals, color=cmap(t), lw=2)
        axes[1].set_xlabel("Quantile u"); axes[1].set_ylabel("Return")
        axes[1].set_title("Quantile function")
        axes[1].grid(alpha=0.3)

        m = dist.moments(2)
        mean_val = m["mean"]
        std_val  = np.sqrt(m["variance"])
        fig.suptitle(
            f"Regime {src_lbl} → Regime {tgt_lbl}  |  t={t:.2f}  "
            f"|  mean={mean_val:.3f}  std={std_val:.3f}",
            fontsize=11,
        )
        plt.tight_layout()
        fig.savefig(frames_dir / f"frame_{k:03d}.png", dpi=100)
        plt.close(fig)

    # ---------------------------------------------------------------- summary mosaic
    n_show = min(9, len(path))
    step = max(1, len(path) // n_show)
    idx_show = list(range(0, len(path), step))[:n_show]

    ncols = 3
    nrows = (len(idx_show) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5),
                              sharex=True, sharey=False)
    axes = np.array(axes).ravel()

    for ax_idx, k in enumerate(idx_show):
        t = k / n_steps
        dist = path[k]
        dens = _kde(dist.samples, x_grid)
        color = cmap(t)
        axes[ax_idx].fill_between(x_grid, dens, alpha=0.3, color=color)
        axes[ax_idx].plot(x_grid, dens, color=color, lw=2)
        m = dist.moments(2)
        axes[ax_idx].set_title(
            f"t={t:.2f}  μ={m['mean']:.3f}  σ={np.sqrt(m['variance']):.3f}",
            fontsize=9,
        )
        axes[ax_idx].grid(alpha=0.3)
        axes[ax_idx].set_xlabel("Return", fontsize=8)

    for ax in axes[len(idx_show):]:
        ax.set_visible(False)

    fig.suptitle(
        f"W₂-geodesic: regime {src_lbl} (calm) → regime {tgt_lbl} (crisis)\n"
        f"W₂ = {w2_total:.4f}",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(figures_dir / "interpolation_mosaic.png", dpi=120)
    plt.close(fig)

    # ---------------------------------------------------------------- constant-speed check
    w2_vals = [wasserstein_1d(path[0], path[k]) for k in range(len(path))]
    t_vals  = np.linspace(0, 1, len(path))
    expected = np.array(t_vals) * w2_total

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_vals, w2_vals,    "o-", color="steelblue", label="Actual W₂(μ₀, μₜ)", lw=2)
    ax.plot(t_vals, expected, "--",  color="red",       label="Expected: t·W₂(μ₀,μ₁)", lw=1.5)
    ax.set_xlabel("t"); ax.set_ylabel("W₂ distance")
    ax.set_title("Constant-speed geodesic verification")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "geodesic_speed.png", dpi=120)
    plt.close(fig)

    print(f"\nFrames → {frames_dir}  ({n_steps + 1} PNGs)")
    print(f"Mosaic → {figures_dir / 'interpolation_mosaic.png'}")
    print(f"Constant-speed check → {figures_dir / 'geodesic_speed.png'}")


if __name__ == "__main__":
    main()
