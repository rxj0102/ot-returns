"""
Experiment: Wasserstein-based regime detection on a synthetic panel.

Usage:
    python experiments/run_regime_detection.py
"""

import yaml
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic import generate_regime_switching_panel
from otreturns.distributions import DistributionPanel
# from otreturns.regimes import detect_regimes, compute_distance_matrix  # Prompt 6


def main():
    with open("experiments/config.yaml") as f:
        cfg = yaml.safe_load(f)

    print("Generating regime-switching panel...")
    panel_df, true_labels = generate_regime_switching_panel(
        n_dates=cfg["data"]["n_dates"],
        n_stocks=cfg["data"]["n_stocks"],
        seed=cfg["data"]["seed"],
    )

    panel = DistributionPanel.from_panel(
        panel_df,
        min_stocks=cfg["data"]["min_stocks"],
        winsorize=cfg["data"]["winsorize"],
    )
    print(f"Panel: {panel}")
    print(f"True regime distribution: {np.bincount(true_labels)}")

    # Regime detection — implemented in Prompt 6
    print("Regime detection not yet implemented (Prompt 6).")


if __name__ == "__main__":
    main()
