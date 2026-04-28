"""
Experiment: Power analysis — Wasserstein, KS, and energy-distance two-sample tests.

For varying effect sizes (mean shifts and scale changes), estimates the
rejection rate (power) at α=0.05 across 100 Monte Carlo replications.

Usage:
    python experiments/run_two_sample_tests.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import yaml
from pathlib import Path
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic import generate_two_sample_test_data
from otreturns.testing import wasserstein_two_sample_test, energy_distance_test


def _ks_test(s1, s2):
    stat, pval = ks_2samp(s1, s2)
    return {"p_value": pval, "reject_h0": pval < 0.05}


def power_curve(effect_values, effect_type, n=200, n_reps=50,
                n_perms=199, seed=0):
    """Estimate rejection rate for each effect size (n_reps Monte Carlo runs)."""
    rng = np.random.default_rng(seed)
    rows = []
    for effect in effect_values:
        w_rejects = ks_rejects = e_rejects = 0
        for rep in range(n_reps):
            s = int(rng.integers(1_000_000))
            if effect_type == "mean_shift":
                s1 = rng.normal(0, 1, n)
                s2 = rng.normal(effect, 1, n)
            else:  # scale_change
                s1 = rng.normal(0, 1, n)
                s2 = rng.normal(0, effect, n)

            w_rejects  += wasserstein_two_sample_test(
                s1, s2, n_permutations=n_perms, seed=s)["reject_h0"]
            ks_rejects += _ks_test(s1, s2)["reject_h0"]
            e_rejects  += energy_distance_test(s1, s2, n_permutations=n_perms)["reject_h0"]

        rows.append({
            "effect": effect,
            "wasserstein_power": w_rejects / n_reps,
            "ks_power":          ks_rejects / n_reps,
            "energy_power":      e_rejects / n_reps,
        })
    return pd.DataFrame(rows)


def main():
    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    figures_dir = Path(cfg["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(cfg["output_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Two-Sample Test Power Analysis")
    print("=" * 60)

    n_obs   = 150
    n_reps  = 50     # Monte Carlo reps per effect size
    n_perms = 199    # permutations per test call

    # ---------------------------------------------------------------- mean shift
    mean_shifts = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
    print(f"\nMean-shift power curve  (n={n_obs}, reps={n_reps})...")
    df_mean = power_curve(mean_shifts, "mean_shift",
                          n=n_obs, n_reps=n_reps, n_perms=n_perms, seed=1)

    print(f"\n{'Shift':>6}  {'W₂':>8}  {'KS':>8}  {'Energy':>8}")
    print("-" * 36)
    for _, row in df_mean.iterrows():
        print(f"{row['effect']:>6.2f}  "
              f"{row['wasserstein_power']:>8.3f}  "
              f"{row['ks_power']:>8.3f}  "
              f"{row['energy_power']:>8.3f}")

    # ---------------------------------------------------------------- scale change
    scale_vals = [1.0, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0]
    print(f"\nScale-change power curve  (n={n_obs}, reps={n_reps})...")
    df_scale = power_curve(scale_vals, "scale_change",
                           n=n_obs, n_reps=n_reps, n_perms=n_perms, seed=2)

    print(f"\n{'Scale':>6}  {'W₂':>8}  {'KS':>8}  {'Energy':>8}")
    print("-" * 36)
    for _, row in df_scale.iterrows():
        print(f"{row['effect']:>6.2f}  "
              f"{row['wasserstein_power']:>8.3f}  "
              f"{row['ks_power']:>8.3f}  "
              f"{row['energy_power']:>8.3f}")

    # ---------------------------------------------------------------- plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(df_mean["effect"], df_mean["wasserstein_power"], "o-",
            color="steelblue", label="Wasserstein W₂", lw=2)
    ax.plot(df_mean["effect"], df_mean["ks_power"], "s--",
            color="darkorange", label="Kolmogorov–Smirnov", lw=2)
    ax.plot(df_mean["effect"], df_mean["energy_power"], "^:",
            color="forestgreen", label="Energy distance", lw=2)
    ax.axhline(0.05, color="red", ls=":", alpha=0.7, label="α=0.05")
    ax.set_xlabel("Mean shift (σ units)"); ax.set_ylabel("Rejection rate")
    ax.set_title("Power vs mean shift  (N(0,1) vs N(δ,1))")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(df_scale["effect"], df_scale["wasserstein_power"], "o-",
            color="steelblue", label="Wasserstein W₂", lw=2)
    ax.plot(df_scale["effect"], df_scale["ks_power"], "s--",
            color="darkorange", label="Kolmogorov–Smirnov", lw=2)
    ax.plot(df_scale["effect"], df_scale["energy_power"], "^:",
            color="forestgreen", label="Energy distance", lw=2)
    ax.axhline(0.05, color="red", ls=":", alpha=0.7, label="α=0.05")
    ax.set_xlabel("Scale ratio"); ax.set_ylabel("Rejection rate")
    ax.set_title("Power vs scale change  (N(0,1) vs N(0,σ²))")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.suptitle(f"Two-sample test power  (n={n_obs} per group, {n_reps} reps)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(figures_dir / "power_analysis.png", dpi=120)
    plt.close(fig)

    df_mean.to_csv(results_dir / "power_mean_shift.csv", index=False)
    df_scale.to_csv(results_dir / "power_scale_change.csv", index=False)
    print(f"\nFigures → {figures_dir}")
    print(f"Results → {results_dir}")


if __name__ == "__main__":
    main()
