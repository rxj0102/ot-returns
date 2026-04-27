"""
Experiment: Displacement interpolation between regime barycenters.

Usage:
    python experiments/run_interpolation_visualization.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.synthetic import generate_regime_switching_panel
from otreturns.distributions import DistributionPanel
# from otreturns.interpolation import interpolation_path  # Prompt 4


def main():
    panel_df, regime_labels = generate_regime_switching_panel(
        n_dates=504, n_stocks=500, seed=42
    )
    panel = DistributionPanel.from_panel(panel_df)
    print(f"Panel: {panel}")
    print("Displacement interpolation not yet implemented (Prompt 4).")


if __name__ == "__main__":
    main()
