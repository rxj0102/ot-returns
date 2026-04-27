"""
Experiment: Wasserstein barycenters as canonical regime distributions.

Usage:
    python experiments/run_barycenter_analysis.py
"""

import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic import generate_regime_switching_panel
from otreturns.distributions import DistributionPanel
# from otreturns.barycenters import wasserstein_barycenter_1d  # Prompt 3


def main():
    with open("experiments/config.yaml") as f:
        cfg = yaml.safe_load(f)

    panel_df, true_labels = generate_regime_switching_panel(
        n_dates=cfg["data"]["n_dates"],
        n_stocks=cfg["data"]["n_stocks"],
        seed=cfg["data"]["seed"],
    )
    panel = DistributionPanel.from_panel(panel_df)
    print(f"Panel: {panel}")
    print("Barycenter analysis not yet implemented (Prompt 3).")


if __name__ == "__main__":
    main()
