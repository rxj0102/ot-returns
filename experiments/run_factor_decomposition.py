"""
Experiment: Rolling transport-map factor decomposition.

Shows what fraction of each day's distributional shift is explained by
the common factor structure vs idiosyncratic noise.

Usage:
    python experiments/run_factor_decomposition.py
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

from data.synthetic import generate_factor_driven_panel
from otreturns.distributions import DistributionPanel
from otreturns.factors import (
    rolling_factor_attribution,
    transport_pca,
    variance_decomposition_ot,
)


def main():
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    figures_dir = Path(cfg["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg["output_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Transport Factor Decomposition Experiment")
    print("=" * 60)

    n_dates, n_stocks, n_factors = 60, 400, 3
    print(f"\nGenerating factor-driven panel  "
          f"(n_dates={n_dates}, n_stocks={n_stocks}, n_factors={n_factors})...")
    panel_df, factor_returns, factor_loadings = generate_factor_driven_panel(
        n_dates=n_dates, n_stocks=n_stocks,
        n_factors=n_factors, seed=cfg["data"]["seed"],
    )
    panel = DistributionPanel.from_panel(
        panel_df, min_stocks=50, winsorize=cfg["data"]["winsorize"]
    )
    dates = panel.dates
    T = len(dates)
    factor_loadings = factor_loadings[:n_stocks]      # trim if needed
    factor_names = [f"factor_{k}" for k in range(n_factors)]

    loadings_df = pd.DataFrame(
        factor_loadings, columns=factor_names
    )

    # ---------------------------------------------------------------- rolling attribution
    print("\nRunning rolling factor attribution (window=1)...")
    df_attr = rolling_factor_attribution(panel, loadings_df, window=1)

    print(f"\nRolling attribution summary (n={len(df_attr)} date pairs):")
    print(f"  Mean factor R²      : {df_attr['factor_r_squared'].mean():.4f}")
    print(f"  Median factor R²    : {df_attr['factor_r_squared'].median():.4f}")
    print(f"  Mean total W₂ shift : {df_attr['total_shift_W2'].mean():.6f}")
    for name in factor_names:
        col = f"{name}_contribution"
        print(f"  Mean α({name}) : {df_attr[col].mean():.6f}")

    # Fraction of days where R² > 0.5
    frac_explained = (df_attr["factor_r_squared"] > 0.5).mean()
    print(f"\n  Days with R²>0.5    : {frac_explained:.1%}")

    # ---------------------------------------------------------------- transport PCA
    print("\nRunning transport PCA (n_components=3)...")
    pca_result = transport_pca(panel, n_components=3, n_support=150)
    evr = pca_result["explained_variance_ratio"]
    print(f"  Explained variance ratio: {' | '.join(f'{v:.3f}' for v in evr)}")
    print(f"  Cumulative: {evr.cumsum()[-1]:.3f}")

    # ---------------------------------------------------------------- variance decomposition
    print("\nRunning variance decomposition...")
    vd = variance_decomposition_ot(panel, loadings_df)
    print(f"  Total variation (mean W₂²) : {vd['total_variation']:.8f}")
    print(f"  Factor explained fraction  : {vd['factor_explained_fraction']:.4f}")
    print(f"  Idiosyncratic fraction     : {vd['idiosyncratic_fraction']:.4f}")

    # ---------------------------------------------------------------- plots
    # 1. Rolling R² over time
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(df_attr["date"].astype(int), df_attr["total_shift_W2"],
                 color="steelblue", lw=1.5)
    axes[0].set_ylabel("W₂ shift"); axes[0].set_title("Total distributional shift (W₂)")
    axes[0].axvline(T // 2, color="red", ls="--", alpha=0.5, label="Vol regime change")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].plot(df_attr["date"].astype(int), df_attr["factor_r_squared"],
                 color="darkorange", lw=1.5, label="Factor R²")
    axes[1].axhline(0.5, color="gray", ls=":", alpha=0.7, label="R²=0.5 threshold")
    axes[1].axvline(T // 2, color="red", ls="--", alpha=0.5)
    axes[1].set_ylabel("R²"); axes[1].set_xlabel("Date index")
    axes[1].set_title("Factor-explained fraction of distributional shift")
    axes[1].set_ylim(0, 1); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "factor_r_squared.png", dpi=120)
    plt.close(fig)

    # 2. Transport PCA scores
    scores = pca_result["scores"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(scores[:, 0], scores[:, 1], c=range(T), cmap="plasma", s=25)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].set_title("Transport PCA scores  (color = time)")
    axes[0].grid(alpha=0.3)

    axes[1].bar(range(1, len(evr) + 1), evr * 100, color="steelblue", alpha=0.8)
    axes[1].set_xlabel("Principal component"); axes[1].set_ylabel("Explained variance (%)")
    axes[1].set_title("Transport PCA — scree plot")
    axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(figures_dir / "transport_pca.png", dpi=120)
    plt.close(fig)

    # 3. PC components as tangent-space functions
    u_grid = np.linspace(0.01, 0.99, 150)
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = ["steelblue", "darkorange", "forestgreen"]
    for i, (comp, col) in enumerate(zip(pca_result["components"], palette)):
        ax.plot(u_grid, comp, color=col, lw=2, label=f"PC{i+1}  ({evr[i]*100:.1f}%)")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Quantile u"); ax.set_ylabel("Displacement (tangent direction)")
    ax.set_title("Transport PCA components in tangent space")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(figures_dir / "pca_components.png", dpi=120)
    plt.close(fig)

    df_attr.to_csv(results_dir / "factor_attribution.csv", index=False)
    print(f"\nFigures → {figures_dir}")
    print(f"Results → {results_dir}")


if __name__ == "__main__":
    main()
