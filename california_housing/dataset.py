"""
Data loading and cleaning.

Design decisions:
- load_data tries local file first, then a public URL, then sklearn fallback
- clean_data drops missing total_bedrooms rows and censored $500,001 values
"""

import os
import pandas as pd
import numpy as np

from california_housing.config import RAW_DATA_PATH


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_data(local_path: str = None) -> pd.DataFrame:
    """
    Load housing data from local file, public URL, or sklearn fallback.

    Priority:
    1. Local CSV (data/raw/housing.csv)
    2. Download from public URL (includes ocean_proximity)
    3. sklearn's fetch_california_housing (no ocean_proximity)
    """
    path = local_path or str(RAW_DATA_PATH)

    if os.path.exists(path):
        print(f"Loading data from {path}")
        return pd.read_csv(path)

    url = (
        "https://raw.githubusercontent.com/ageron/handson-ml2"
        "/master/datasets/housing/housing.csv"
    )
    try:
        print(f"Downloading data from {url}")
        df = pd.read_csv(url)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Data saved to {path}")
        return df
    except Exception as e:
        print(f"Download failed ({e}). Using sklearn's California Housing dataset.")
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame.copy()
        df.columns = [c.lower() for c in df.columns]
        df.rename(columns={
            "medinc"   : "median_income",
            "houseage" : "housing_median_age",
            "averooms" : "total_rooms",
            "avebedrms": "total_bedrooms",
            "population": "population",
            "aveoccup" : "households",
            "latitude" : "latitude",
            "longitude": "longitude",
            "medhouval": "median_house_value",
        }, inplace=True)
        # Scale target to dollars (sklearn stores in $100k units)
        if df["median_house_value"].max() < 1000:
            df["median_house_value"] = df["median_house_value"] * 100_000
        return df


# ---------------------------------------------------------------------------
# Inspect
# ---------------------------------------------------------------------------

def inspect_data(df: pd.DataFrame) -> dict:
    """Return a data quality summary."""
    missing = df.isnull().sum()
    return {
        "shape"        : df.shape,
        "dtypes"       : df.dtypes,
        "missing_values": missing[missing > 0],
        "missing_pct"  : (missing / len(df) * 100).round(2)[missing > 0],
        "duplicates"   : df.duplicated().sum(),
        "describe"     : df.describe(),
    }


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the housing dataset.

    Steps:
    - Drop rows with missing total_bedrooms (207 rows, <1% of data).
      Imputation is not preferred here because the missingness is not
      random — these are blocks with genuinely unknown bedroom counts —
      and 207 rows is negligible loss.
    - Remove duplicate rows.
    - Cap median_house_value at $500,001 (dataset ceiling — values at this
      threshold are censored and not true market values).
    """
    initial_rows = len(df)

    df = df.dropna(subset=["total_bedrooms"]).copy()
    rows_after_missing = len(df)

    df = df.drop_duplicates().copy()
    rows_after_dedup = len(df)

    df = df[df["median_house_value"] < 500_001].copy()
    rows_after_cap = len(df)

    print(
        f"\nData Cleaning Summary:"
        f"\n  Original rows            : {initial_rows:,}"
        f"\n  After dropping NaN       : {rows_after_missing:,}  "
        f"(-{initial_rows - rows_after_missing} rows)"
        f"\n  After deduplication      : {rows_after_dedup:,}  "
        f"(-{rows_after_missing - rows_after_dedup} rows)"
        f"\n  After removing $500k cap : {rows_after_cap:,}  "
        f"(-{rows_after_dedup - rows_after_cap} rows)"
        f"\n  Final shape              : {df.shape}"
    )

    return df
