"""Load and clean IPL ball-by-ball data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


NUMERIC_COLUMNS = [
    "runs_batter",
    "batter_balls",
    "bowler_wicket",
    "valid_ball",
    "runs_bowler",
    "runs_total",
    "innings",
    "bat_pos",
]


def load_data(path: str | Path = "data/IPL.csv") -> pd.DataFrame:
    """Load IPL CSV from a path relative to the current working directory."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{p}'. Place IPL.csv under data/ (see README)."
        )
    return pd.read_csv(p, low_memory=False)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean ball-by-ball IPL data: valid rows, types, chronological sort.

    Removes invalid header-like rows via numeric coercion on ``match_id``.
    Parses dates from ``dt`` or ``date`` when present.
    """
    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")

    if "match_id" not in df.columns:
        raise ValueError("Expected column 'match_id' in dataset.")

    out = df.copy()
    out["_mid_num"] = pd.to_numeric(out["match_id"], errors="coerce")
    out = out[out["_mid_num"].notna()].copy()
    out["match_id"] = out["_mid_num"].astype(int)
    out = out.drop(columns=["_mid_num"])

    for col in NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].fillna(0)

    dt_col = None
    if "dt" in out.columns:
        dt_col = "dt"
        s = out["dt"].astype(str)
        parsed = pd.to_datetime(s, format="%Y%m%d", errors="coerce")
        fallback = pd.to_datetime(out["dt"], errors="coerce")
        out["_parsed_dt"] = parsed.where(parsed.notna(), fallback)
    elif "date" in out.columns:
        dt_col = "date"
        out["_parsed_dt"] = pd.to_datetime(out["date"], errors="coerce")

    if dt_col is None:
        raise ValueError(
            "Could not find a parseable date column: expected 'dt' or 'date'."
        )

    out["_parsed_dt"] = pd.to_datetime(out["_parsed_dt"], errors="coerce")
    invalid_dates = out["_parsed_dt"].isna().sum()
    if invalid_dates:
        out = out[out["_parsed_dt"].notna()].copy()

    out["dt"] = out["_parsed_dt"]
    out = out.drop(columns=["_parsed_dt"])

    sort_cols = ["dt", "match_id"]
    out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return out
