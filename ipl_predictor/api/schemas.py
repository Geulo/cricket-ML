"""Pydantic request/response models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class PredictionRequest(BaseModel):
    team_1: str = Field(..., description="First team (not necessarily stronger)")
    team_2: str = Field(..., description="Second team")
    venue: str
    toss_winner: str
    toss_decision: str = Field(..., description='Either "bat" or "field"')
    top_n: int = Field(default=3, ge=1, le=20)

    @model_validator(mode="after")
    def validate_teams_and_toss(self):
        if self.team_1 == self.team_2:
            raise ValueError("team_1 and team_2 must differ.")
        if self.toss_winner not in (self.team_1, self.team_2):
            raise ValueError("toss_winner must be either team_1 or team_2.")
        return self

    @field_validator("toss_decision")
    @classmethod
    def toss_decision_ok(cls, v: str) -> str:
        low = v.lower().strip()
        if low not in ("bat", "field"):
            raise ValueError('toss_decision must be "bat" or "field".')
        return low


class PredictionResponse(BaseModel):
    winner: str
    strong_team: str
    weak_team: str
    strong_team_prob: float
    weak_team_prob: float
    top_batsmen: list[dict[str, Any]]
    top_bowlers: list[dict[str, Any]]
    bat_confidence: float | None = None
    bat_bet_recommended: bool = False
    bowl_confidence: float | None = None
    bowl_bet_recommended: bool = False
    notes: list[str]
