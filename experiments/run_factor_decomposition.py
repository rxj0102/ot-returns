"""
Experiment: Transport map decomposition into factor components.

Usage:
    python experiments/run_factor_decomposition.py
"""

import sys
from pathlib import Path
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic import generate_factor_driven_panel
from otreturns.distributions import DistributionPanel
# from otreturns.factors import fit_factor_model_ot  # Prompt 7


def main():
    with open("experiments/config.yaml") as f:
        cfg = yaml.safe_load(f)

    panel_df, factor_returns, factor_loadings = generate_factor_driven_panel(
        n_dates=cfg["data"]["n_dates"],
        n_stocks=cfg["data"]["n_stocks"],
        n_factors=cfg["factors"]["n_factors"],
        seed=cfg["data"]["seed"],
    )
    panel = DistributionPanel.from_panel(panel_df)
    print(f"Panel: {panel}")
    print(f"Factor returns shape: {factor_returns.shape}")
    print(f"Factor loadings shape: {factor_loadings.shape}")
    print("Factor decomposition not yet implemented (Prompt 7).")


if __name__ == "__main__":
    main()
