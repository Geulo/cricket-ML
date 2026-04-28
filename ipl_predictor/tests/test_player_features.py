"""Tests for player features."""

from __future__ import annotations

from src.data import clean_data, load_data
from src.player_features import build_player_features

from tests.helpers import ipl_csv_path


def test_player_features_nonempty():
    p = ipl_csv_path()
    if not p.exists():
        import pytest

        pytest.skip("data/IPL.csv missing")
    df = clean_data(load_data(p))
    bat, bowl = build_player_features(df)
    assert len(bat) > 0 and len(bowl) > 0
