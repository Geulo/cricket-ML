"""Unified inference for match winner and top-N players."""

from __future__ import annotations

import difflib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.betting import normalized_top_gap
from src.match_features import MATCH_FEATURE_COLUMNS, load_win_rates


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Required trained artifacts (``betting_calibration.joblib`` is optional and never loaded here).
REQUIRED_MODEL_ARTIFACTS: tuple[tuple[str, Path], ...] = (
    ("GradientBoosting match model", "gb.joblib"),
    ("team encoder", "le_team.joblib"),
    ("venue encoder", "le_venue.joblib"),
    ("batsman ranker", "bat.joblib"),
    ("bowler ranker", "bowl.joblib"),
    ("latest player feature snapshots", "latest_player_history.joblib"),
    ("team player pools", "team_player_pool.joblib"),
    ("player feature column lists", "player_feature_columns.joblib"),
    ("historical win rates JSON", "win_rates.json"),
)


def _artifacts_dir(models_dir: Path | None) -> Path:
    return models_dir if models_dir is not None else MODELS_DIR


def _check_required_artifacts(md: Path) -> None:
    missing: list[str] = []
    for description, fname in REQUIRED_MODEL_ARTIFACTS:
        p = md / fname
        if not p.exists():
            missing.append(f"  - {fname} ({description})")
    if missing:
        raise FileNotFoundError(
            "Missing model artifact(s). Train first from the `ipl_predictor/` directory:\n"
            "  python train.py\n"
            "after placing the real Kaggle IPL ball-by-ball CSV at `data/IPL.csv` "
            "(see README).\n"
            "The following files were not found under "
            f"{md.resolve()}:\n" + "\n".join(missing)
        )


def batting_bowling_teams_first_innings(
    team_1: str,
    team_2: str,
    toss_winner: str,
    toss_decision: str,
) -> tuple[str, str]:
    """
    First-innings batting vs bowling side from toss.

    ``bat`` → toss winner bats first; ``field`` → toss winner bowls first (opponent bats).
    """
    opp = team_2 if toss_winner == team_1 else team_1
    if str(toss_decision).lower().strip() == "bat":
        return toss_winner, opp
    return opp, toss_winner


def resolve_strong_weak(
    team_1: str,
    team_2: str,
    win_rates: dict[str, float],
) -> tuple[str, str, float, float]:
    """Return strong_team, weak_team by historical win rate (tie-break lexicographic)."""
    r1 = float(win_rates.get(team_1, 0.5))
    r2 = float(win_rates.get(team_2, 0.5))
    if r1 > r2 or (r1 == r2 and str(team_1) <= str(team_2)):
        return team_1, team_2, r1, r2
    return team_2, team_1, r2, r1


def _resolve_venue(venue_s: str, venues_known: set[str]) -> tuple[str, str | None]:
    """Exact match or fuzzy fallback via ``difflib``. Returns (canonical venue, optional note)."""
    if venue_s in venues_known:
        return venue_s, None
    candidates = sorted(venues_known)
    matches = difflib.get_close_matches(venue_s, candidates, n=1, cutoff=0.6)
    if matches:
        matched = matches[0]
        return matched, f"Venue '{venue_s}' not found; using closest match '{matched}'."
    raise ValueError(
        f"Unknown venue '{venue_s}'. The venue string must match training data. "
        "Use `GET /venues` (API) or inspect `le_venue.joblib` after training."
    )


def _optional_betting_confidence(
    models_dir: Path,
    artifact_name: str,
    scores: np.ndarray | None,
) -> tuple[float | None, bool]:
    """
    Raw gap from rank scores; optional isotonic maps gap → display confidence.

    Bet recommendation uses **raw** ``normalized_top_gap`` vs ``threshold`` (same scale as
    training). ``bat_confidence`` / ``bowl_confidence`` use isotonic mapping of that gap when
    ``isotonic`` is present, else the raw gap.

    If the calibration file is missing, returns ``(None, False)``.
    """
    path = models_dir / artifact_name
    if not path.exists():
        return None, False
    if scores is None or scores.size == 0:
        return None, False

    cal = joblib.load(path)
    iso = cal.get("isotonic")
    thr = cal.get("threshold")

    gap = normalized_top_gap(scores)
    if iso is not None:
        conf = float(iso.predict(np.asarray([gap]))[0])
    else:
        conf = float(gap)

    recommended = thr is not None and gap >= thr
    return conf, recommended


def build_single_match_row(
    venue: str,
    strong_team: str,
    weak_team: str,
    toss_winner: str,
    toss_decision: str,
    strong_wr: float,
    weak_wr: float,
    le_team,
    le_venue,
) -> pd.DataFrame:
    """Construct one encoded feature row for the GradientBoosting model."""
    toss_decision_norm = str(toss_decision).lower().strip()
    toss_bat = int(toss_decision_norm == "bat")
    toss_is_strong = int(toss_winner == strong_team)

    venue_enc = le_venue.transform([str(venue)])[0]
    st_enc = le_team.transform([strong_team])[0]
    wk_enc = le_team.transform([weak_team])[0]

    row = {
        "venue_enc": venue_enc,
        "strong_team_enc": st_enc,
        "weak_team_enc": wk_enc,
        "strong_team_winrate": strong_wr,
        "weak_team_winrate": weak_wr,
        "winrate_diff": strong_wr - weak_wr,
        "toss_is_strong_team": toss_is_strong,
        "toss_bat": toss_bat,
    }
    return pd.DataFrame([row])[MATCH_FEATURE_COLUMNS]


def predict_match_full(
    team_1: str,
    team_2: str,
    venue: str,
    toss_winner: str,
    toss_decision: str,
    top_n: int = 3,
    models_dir: str | Path | None = None,
) -> dict:
    """
    Predict match winner probabilities and top-N batsmen/bowlers from saved artifacts.

    ``team_1`` and ``team_2`` are arbitrary labels; internally ``strong_team``/``weak_team``
    are resolved using historical ``win_rates.json``. Does not require ``betting_calibration.joblib``.

    Inference uses only **past** player feature snapshots (latest rows from training history)
    and trained encoders—no live match ball data.

    Top batsmen are drawn only from the **first-innings batting** side; top bowlers only from
    the **first-innings bowling** side (from toss + ``bat``/``field``).
    """
    md = Path(_artifacts_dir(models_dir))
    _check_required_artifacts(md)

    gb = joblib.load(md / "gb.joblib")
    le_team = joblib.load(md / "le_team.joblib")
    le_venue = joblib.load(md / "le_venue.joblib")
    bat_m = joblib.load(md / "bat.joblib")
    bowl_m = joblib.load(md / "bowl.joblib")
    latest_hist = joblib.load(md / "latest_player_history.joblib")
    team_pool = joblib.load(md / "team_player_pool.joblib")
    feat_cols_map = joblib.load(md / "player_feature_columns.joblib")

    win_rates = load_win_rates(md / "win_rates.json")

    teams_known = set(le_team.classes_)
    venues_known = set(le_venue.classes_)

    for label, t in (("team_1", team_1), ("team_2", team_2)):
        if t not in teams_known:
            raise ValueError(
                f"Unknown team '{t}' ({label}). This name must appear in the training data. "
                "Use `GET /teams` (API) or inspect `le_team.joblib` after training for valid labels."
            )

    if toss_winner not in (team_1, team_2):
        raise ValueError(
            f"Invalid toss_winner '{toss_winner}': must be exactly team_1 or team_2 "
            f"({team_1!r} or {team_2!r})."
        )

    venue_s = str(venue).strip()
    venue_for_model, venue_note = _resolve_venue(venue_s, venues_known)

    strong_team, weak_team, swr, wwr = resolve_strong_weak(team_1, team_2, win_rates)

    X = build_single_match_row(
        venue_for_model,
        strong_team,
        weak_team,
        toss_winner,
        toss_decision,
        swr,
        wwr,
        le_team,
        le_venue,
    ).astype(float)

    proba_strong = float(gb.predict_proba(X)[0, 1])
    proba_weak = float(1.0 - proba_strong)

    winner = strong_team if proba_strong >= 0.5 else weak_team

    bat_feats = feat_cols_map["bat"]
    bowl_feats = feat_cols_map["bowl"]
    bat_hist = latest_hist["bat"]
    bowl_hist = latest_hist["bowl"]

    notes: list[str] = [
        "Player predictions are based on recent historical player availability, not confirmed playing XI.",
    ]
    if venue_note:
        notes.append(venue_note)

    batting_side, bowling_side = batting_bowling_teams_first_innings(
        team_1, team_2, toss_winner, toss_decision
    )
    notes.append(
        f"First innings (from toss): batting={batting_side}, bowling={bowling_side}. "
        "Top-N batsman/bowler lists use those sides only."
    )

    bat_candidates = sorted(set(team_pool.get(batting_side, {}).get("batsmen", [])))
    bowl_candidates = sorted(set(team_pool.get(bowling_side, {}).get("bowlers", [])))

    if not bat_candidates:
        notes.append(
            f"Empty batsman pool for first-innings batting team '{batting_side}'. "
            "Batting rankings skipped; train on fuller IPL data or check franchise names."
        )
    if not bowl_candidates:
        notes.append(
            f"Empty bowler pool for first-innings bowling team '{bowling_side}'. "
            "Bowling rankings skipped; train on fuller IPL data or check franchise names."
        )

    scores_bat: np.ndarray | None = None
    scores_bowl: np.ndarray | None = None

    top_batsmen: list[dict[str, object]] = []
    if bat_candidates:
        bat_snapshots: list = []
        for pl in bat_candidates:
            sub = bat_hist[bat_hist["batter"] == pl]
            if sub.empty:
                continue
            bat_snapshots.append(sub.iloc[-1])
        if bat_snapshots:
            bat_df = pd.DataFrame(bat_snapshots).replace([np.inf, -np.inf], np.nan).fillna(0)
            scores_bat = bat_m.predict(bat_df[bat_feats].astype(float))
            order = np.argsort(-scores_bat)
            for idx in order[:top_n]:
                r = bat_df.iloc[int(idx)]
                raw_career = r.get("career", 0.0)
                career_avg = round(float(raw_career), 2) if pd.notna(raw_career) else 0.0
                top_batsmen.append(
                    {
                        "player": str(r["batter"]),
                        "team": str(r["batting_team"]),
                        "rank_score": float(scores_bat[int(idx)]),
                        "career_avg": career_avg,
                    }
                )
        elif bat_candidates:
            notes.append(
                "No feature history in artifacts for pooled batsmen (latest_player_history incomplete). "
                "Batsman rankings skipped."
            )

    top_bowlers: list[dict[str, object]] = []
    if bowl_candidates:
        rows_b = []
        for pl in bowl_candidates:
            sub = bowl_hist[bowl_hist["bowler"] == pl]
            if sub.empty:
                continue
            rows_b.append(sub.iloc[-1])
        if rows_b:
            bowl_df = pd.DataFrame(rows_b).replace([np.inf, -np.inf], np.nan).fillna(0)
            scores_bowl = bowl_m.predict(bowl_df[bowl_feats].astype(float))
            ord_b = np.argsort(-scores_bowl)
            for idx in ord_b[:top_n]:
                r = bowl_df.iloc[int(idx)]
                raw_career = r.get("career", 0.0)
                career_avg = round(float(raw_career), 2) if pd.notna(raw_career) else 0.0
                top_bowlers.append(
                    {
                        "player": str(r["bowler"]),
                        "team": str(r["bowling_team"]),
                        "rank_score": float(scores_bowl[int(idx)]),
                        "career_avg": career_avg,
                    }
                )
        elif bowl_candidates:
            notes.append(
                "No feature history in artifacts for pooled bowlers (latest_player_history incomplete). "
                "Bowler rankings skipped."
            )

    bat_confidence, bat_bet_recommended = _optional_betting_confidence(
        md, "betting_calibration.joblib", scores_bat
    )
    bowl_confidence, bowl_bet_recommended = _optional_betting_confidence(
        md, "bowl_betting_calibration.joblib", scores_bowl
    )

    return {
        "winner": winner,
        "strong_team": strong_team,
        "weak_team": weak_team,
        "strong_team_prob": proba_strong,
        "weak_team_prob": proba_weak,
        "top_batsmen": top_batsmen,
        "top_bowlers": top_bowlers,
        "bat_confidence": bat_confidence,
        "bat_bet_recommended": bat_bet_recommended,
        "bowl_confidence": bowl_confidence,
        "bowl_bet_recommended": bowl_bet_recommended,
        "notes": notes,
    }
