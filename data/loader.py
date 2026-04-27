"""
Panel data loading utilities.

Expected long-format schema:
    date    : date identifier (parsed to datetime when possible)
    ticker  : stock identifier (string)
    return  : daily simple return (float)

Optionally: permno, market_cap, sector, ...
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def load_panel_csv(
    filepath: str | Path,
    date_col: str = "date",
    parse_dates: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Load long-format panel data from a CSV file.

    Args:
        filepath: path to CSV file
        date_col: name of the date column
        parse_dates: whether to parse the date column as datetime
        **kwargs: additional arguments forwarded to pd.read_csv

    Returns:
        DataFrame with at least [date, ticker/permno, return] columns
    """
    df = pd.read_csv(filepath, **kwargs)
    if parse_dates and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def load_panel_parquet(
    filepath: str | Path,
    **kwargs,
) -> pd.DataFrame:
    """
    Load long-format panel data from a Parquet file.

    Args:
        filepath: path to Parquet file
        **kwargs: additional arguments forwarded to pd.read_parquet

    Returns:
        DataFrame with at least [date, ticker/permno, return] columns
    """
    return pd.read_parquet(filepath, **kwargs)


def validate_panel(
    df: pd.DataFrame,
    date_col: str = "date",
    return_col: str = "return",
    id_col: str = None,
) -> bool:
    """
    Validate that a DataFrame has the expected long-format panel structure.

    Checks:
    - Required columns are present
    - No duplicate (date, id) combinations if id_col is provided
    - Return column contains finite floats

    Returns:
        True if valid, raises ValueError otherwise
    """
    required = [date_col, return_col]
    if id_col is not None:
        required.append(id_col)

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if id_col is not None and df.duplicated([date_col, id_col]).any():
        raise ValueError(f"Duplicate ({date_col}, {id_col}) pairs found")

    if not np.issubdtype(df[return_col].dtype, np.number):
        raise ValueError(f"Column '{return_col}' must be numeric")

    return True


def panel_summary(
    df: pd.DataFrame,
    date_col: str = "date",
    return_col: str = "return",
) -> dict:
    """
    Return a summary of the panel: date range, stock counts, missing data.
    """
    n_dates = df[date_col].nunique()
    n_obs = len(df)
    stocks_per_date = df.groupby(date_col)[return_col].count()

    return {
        "n_dates": n_dates,
        "n_obs": n_obs,
        "date_start": df[date_col].min(),
        "date_end": df[date_col].max(),
        "stocks_per_date_mean": float(stocks_per_date.mean()),
        "stocks_per_date_min": int(stocks_per_date.min()),
        "stocks_per_date_max": int(stocks_per_date.max()),
        "missing_returns": int(df[return_col].isna().sum()),
    }
