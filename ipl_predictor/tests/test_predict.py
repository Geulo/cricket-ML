"""Tests for unified prediction."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


@pytest.mark.skipif(not (MODELS / "gb.joblib").exists(), reason="train models first")
def test_predict_match_full_keys():
    from src.predict import predict_match_full

    # Names must exist in trained encoders — synthetic uses these franchises.
    out = predict_match_full(
        team_1="Mumbai Indians",
        team_2="Chennai Super Kings",
        venue="Wankhede Stadium",
        toss_winner="Mumbai Indians",
        toss_decision="bat",
        top_n=3,
        models_dir=MODELS,
    )
    required = {
        "winner",
        "strong_team",
        "weak_team",
        "strong_team_prob",
        "weak_team_prob",
        "top_batsmen",
        "top_bowlers",
        "bat_confidence",
        "bat_bet_recommended",
        "bowl_confidence",
        "bowl_bet_recommended",
        "notes",
    }
    assert required <= set(out.keys())
    assert isinstance(out["top_batsmen"], list)
    assert isinstance(out["top_bowlers"], list)
    assert isinstance(out["notes"], list)
