"""Tests for match-level features."""

from __future__ import annotations

from src.data import clean_data, load_data
from src.match_features import aggregate_matches

from tests.helpers import ipl_csv_path


def test_one_row_per_match():
    p = ipl_csv_path()
    if not p.exists():
        import pytest

        pytest.skip("data/IPL.csv missing")
    df = clean_data(load_data(p))
    m = aggregate_matches(df)
    assert len(m) == m["match_id"].nunique()
