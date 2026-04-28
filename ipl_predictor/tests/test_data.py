"""Tests for loading and cleaning IPL data."""

from __future__ import annotations

import pandas as pd

from src.data import clean_data, load_data

from tests.helpers import ipl_csv_path


def test_load_and_clean():
    p = ipl_csv_path()
    if not p.exists():
        import pytest

        pytest.skip("data/IPL.csv not present; generate with scripts/gen_synthetic_ipl.py")
    df = clean_data(load_data(p))
    assert "match_id" in df.columns
    assert df["match_id"].dtype in (int, "int64", "int32")
    assert df["match_id"].notna().all()
    assert pd.api.types.is_datetime64_any_dtype(df["dt"])
