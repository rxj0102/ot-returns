"""
Download sample data for ot-returns experiments.

This script generates synthetic data files to data/ so that experiments
can be run without real market data.

Usage:
    python scripts/download_sample_data.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from data.synthetic import (
    generate_normal_panel,
    generate_regime_switching_panel,
    generate_factor_driven_panel,
)

DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    DATA_DIR.mkdir(exist_ok=True)

    print("Generating normal baseline panel (252 days, 500 stocks)...")
    df_normal = generate_normal_panel(n_dates=252, n_stocks=500, seed=0)
    out = DATA_DIR / "sample_normal.parquet"
    df_normal.to_parquet(out, index=False)
    print(f"  Saved {len(df_normal):,} rows -> {out}")

    print("Generating regime-switching panel (504 days, 500 stocks)...")
    df_regime, labels = generate_regime_switching_panel(
        n_dates=504, n_stocks=500, seed=42
    )
    out = DATA_DIR / "sample_regime.parquet"
    df_regime.to_parquet(out, index=False)
    print(f"  Saved {len(df_regime):,} rows -> {out}")

    labels_path = DATA_DIR / "sample_regime_labels.npy"
    np.save(labels_path, labels)
    print(f"  Saved regime labels -> {labels_path}")

    print("Generating factor-driven panel (252 days, 500 stocks, 3 factors)...")
    df_factor, factor_returns, factor_loadings = generate_factor_driven_panel(
        n_dates=252, n_stocks=500, n_factors=3, seed=99
    )
    out = DATA_DIR / "sample_factor.parquet"
    df_factor.to_parquet(out, index=False)
    print(f"  Saved {len(df_factor):,} rows -> {out}")

    np.save(DATA_DIR / "sample_factor_returns.npy", factor_returns)
    np.save(DATA_DIR / "sample_factor_loadings.npy", factor_loadings)
    print("  Saved factor returns and loadings.")

    print("\nDone. Files written to data/")


if __name__ == "__main__":
    main()
