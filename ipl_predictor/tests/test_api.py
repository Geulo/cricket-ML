"""FastAPI integration tests (requires trained ``models/`` artifacts)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.helpers import ROOT

MODELS = ROOT / "models"


@pytest.mark.skipif(not (MODELS / "gb.joblib").exists(), reason="train models first")
def test_api_health_teams_venues_predict_mumbai_csk():
    from api.main import app

    client = TestClient(app)

    h = client.get("/health")
    assert h.status_code == 200
    assert h.json().get("status") == "ok"

    tr = client.get("/teams")
    assert tr.status_code == 200
    teams = tr.json().get("teams", [])
    assert "Mumbai Indians" in teams and "Chennai Super Kings" in teams

    vr = client.get("/venues")
    assert vr.status_code == 200
    venues = vr.json().get("venues", [])
    assert "Wankhede Stadium" in venues

    pr = client.post(
        "/predict",
        json={
            "team_1": "Mumbai Indians",
            "team_2": "Chennai Super Kings",
            "venue": "Wankhede Stadium",
            "toss_winner": "Mumbai Indians",
            "toss_decision": "bat",
            "top_n": 3,
        },
    )
    assert pr.status_code == 200, pr.text
    body = pr.json()
    assert "winner" in body
    assert isinstance(body.get("top_batsmen"), list)
    assert isinstance(body.get("top_bowlers"), list)
