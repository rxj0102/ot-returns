# Data

## Expected Format

All functions in this library expect **long-format** panel data:

| Column   | Type      | Description                            |
|----------|-----------|----------------------------------------|
| `date`   | datetime  | Trading date                           |
| `ticker` | str       | Stock identifier (or `permno`)         |
| `return` | float     | Daily simple return (not log-return)   |

## Sourcing Real Data

### CRSP (via WRDS)
The canonical source for US equity returns.
```python
# Example WRDS query (requires institutional access)
import wrds
db = wrds.Connection()
df = db.raw_sql("""
    SELECT date, permno, ret as return
    FROM crsp.dsf
    WHERE date BETWEEN '2000-01-01' AND '2023-12-31'
      AND ret IS NOT NULL
""")
df.to_parquet("data/crsp_daily.parquet")
```

### Yahoo Finance (free, smaller universe)
```python
import yfinance as yf
# Download S&P 500 constituents and pivot to long format
```

### Kenneth French Data Library
Factor return data available at:
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

## Synthetic Data

Use `data.synthetic` for reproducible experiments without real data:

```python
from data.synthetic import generate_regime_switching_panel

panel_df, regime_labels = generate_regime_switching_panel(
    n_dates=504, n_stocks=500, seed=42
)
```

## Loading

```python
from data.loader import load_panel_csv, load_panel_parquet, validate_panel

df = load_panel_csv("my_returns.csv")
validate_panel(df, id_col="ticker")
```
