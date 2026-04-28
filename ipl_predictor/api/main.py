"""FastAPI application."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
from fastapi import FastAPI, HTTPException

from api.schemas import PredictionRequest, PredictionResponse
from src.predict import predict_match_full

MODELS = ROOT / "models"

app = FastAPI(title="IPL Prediction Suite", version="1.0.0")


def _load_classes(path: Path) -> list[str]:
    if not path.exists():
        return []
    le = joblib.load(path)
    return list(getattr(le, "classes_", []))


@app.get("/health")
def health():
    return {"status": "ok", "models_dir": str(MODELS)}


@app.get("/teams")
def teams():
    names = _load_classes(MODELS / "le_team.joblib")
    if not names:
        raise HTTPException(
            status_code=503,
            detail="Team encoder not found. Train models with `python train.py` first.",
        )
    return {"teams": sorted(names)}


@app.get("/venues")
def venues():
    names = _load_classes(MODELS / "le_venue.joblib")
    if not names:
        raise HTTPException(
            status_code=503,
            detail="Venue encoder not found. Train models with `python train.py` first.",
        )
    return {"venues": sorted(names)}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    try:
        out = predict_match_full(
            team_1=req.team_1,
            team_2=req.team_2,
            venue=req.venue,
            toss_winner=req.toss_winner,
            toss_decision=req.toss_decision,
            top_n=req.top_n,
            models_dir=MODELS,
        )
        return PredictionResponse(**out)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
