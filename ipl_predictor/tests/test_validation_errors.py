"""Validation error messages when inference inputs are invalid."""

from __future__ import annotations

import pytest

from tests.helpers import ROOT

MODELS = ROOT / "models"


@pytest.mark.skipif(not (MODELS / "gb.joblib").exists(), reason="train models first")
def test_unknown_team_error_message():
    from src.predict import predict_match_full

    with pytest.raises(ValueError, match="Unknown team"):
        predict_match_full(
            "Totally Unknown Franchise",
            "Chennai Super Kings",
            "Wankhede Stadium",
            "Chennai Super Kings",
            "bat",
            models_dir=MODELS,
        )


@pytest.mark.skipif(not (MODELS / "gb.joblib").exists(), reason="train models first")
def test_unknown_venue_error_message():
    from src.predict import predict_match_full

    with pytest.raises(ValueError, match="Unknown venue"):
        predict_match_full(
            "Mumbai Indians",
            "Chennai Super Kings",
            "Fictional Cricket Ground XYZ",
            "Mumbai Indians",
            "bat",
            models_dir=MODELS,
        )


@pytest.mark.skipif(not (MODELS / "gb.joblib").exists(), reason="train models first")
def test_toss_winner_not_participant():
    from src.predict import predict_match_full

    with pytest.raises(ValueError, match="Invalid toss_winner"):
        predict_match_full(
            "Mumbai Indians",
            "Chennai Super Kings",
            "Wankhede Stadium",
            "Royal Challengers Bangalore",
            "bat",
            models_dir=MODELS,
        )


@pytest.mark.skipif(not (MODELS / "gb.joblib").exists(), reason="train models first")
def test_missing_artifact_error_lists_files(tmp_path):
    from src.predict import predict_match_full

    empty = tmp_path / "empty_models"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="gb\\.joblib"):
        predict_match_full(
            "Mumbai Indians",
            "Chennai Super Kings",
            "Wankhede Stadium",
            "Mumbai Indians",
            "bat",
            models_dir=empty,
        )
