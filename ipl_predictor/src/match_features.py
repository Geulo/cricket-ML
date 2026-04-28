"""Match-level features with strong/weak team framing and causal win rates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def _prior_winrate(state: dict[str, dict[str, float]], team: str) -> float:
    st = state.get(team)
    if not st or st["games"] <= 0:
        return 0.5
    return st["wins"] / st["games"]


def _record_result(state: dict[str, dict[str, float]], team: str, won: bool) -> None:
    if team not in state:
        state[team] = {"wins": 0.0, "games": 0.0}
    state[team]["games"] += 1
    if won:
        state[team]["wins"] += 1


def aggregate_matches(df: pd.DataFrame) -> pd.DataFrame:
    """One row per match with venue, toss, teams, winner, season."""
    req = ["match_id", "venue", "toss_winner", "toss_decision", "batting_team", "bowling_team"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for match aggregation: {missing}")

    if "match_won_by" not in df.columns:
        raise ValueError("Expected column 'match_won_by' for winner.")

    agg_dict = {
        "dt": ("dt", "first"),
        "venue": ("venue", "first"),
        "toss_winner": ("toss_winner", "first"),
        "toss_decision": ("toss_decision", "first"),
        "team1": ("batting_team", "first"),
        "team2": ("bowling_team", "first"),
        "winner": ("match_won_by", "first"),
    }
    if "season" in df.columns:
        agg_dict["season"] = ("season", "first")

    match_df = df.groupby("match_id", sort=False).agg(**agg_dict).reset_index()

    if "season" not in match_df.columns:
        match_df["season"] = match_df["dt"].dt.year.astype(str)

    match_df = match_df.sort_values(["dt", "match_id"], kind="mergesort").reset_index(drop=True)
    return match_df


def build_match_features_table(match_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Build feature rows with expanding *prior-only* win rates (no leakage).

    Returns the enriched dataframe and final cumulative ``win_rates`` dict for inference.
    """
    m = match_df.sort_values(["dt", "match_id"], kind="mergesort").reset_index(drop=True)
    state: dict[str, dict[str, float]] = {}

    rows: list[dict[str, Any]] = []
    final_rates: dict[str, float] = {}

    for _, row in m.iterrows():
        team1 = row["team1"]
        team2 = row["team2"]
        winner = row["winner"]
        toss_winner = row["toss_winner"]
        toss_decision = str(row["toss_decision"]).lower()

        wr1 = _prior_winrate(state, team1)
        wr2 = _prior_winrate(state, team2)

        if wr1 > wr2 or (wr1 == wr2 and str(team1) <= str(team2)):
            strong_team, weak_team = team1, team2
            strong_wr, weak_wr = wr1, wr2
        else:
            strong_team, weak_team = team2, team1
            strong_wr, weak_wr = wr2, wr1

        strong_wins = 0
        if pd.notna(winner) and winner in (team1, team2):
            strong_wins = int(winner == strong_team)
        toss_is_strong = int(pd.notna(toss_winner) and toss_winner == strong_team)
        toss_bat = int(toss_decision == "bat")

        rows.append(
            {
                "match_id": row["match_id"],
                "dt": row["dt"],
                "venue": row["venue"],
                "season": row["season"],
                "team1": team1,
                "team2": team2,
                "toss_winner": toss_winner,
                "toss_decision": row["toss_decision"],
                "winner": winner,
                "strong_team": strong_team,
                "weak_team": weak_team,
                "strong_team_winrate": strong_wr,
                "weak_team_winrate": weak_wr,
                "winrate_diff": strong_wr - weak_wr,
                "toss_is_strong_team": toss_is_strong,
                "toss_bat": toss_bat,
                "strong_team_wins": strong_wins,
            }
        )

        if pd.notna(winner) and winner in (team1, team2):
            _record_result(state, team1, winner == team1)
            _record_result(state, team2, winner == team2)

    out = pd.DataFrame(rows)
    for t, st in state.items():
        final_rates[t] = st["wins"] / st["games"] if st["games"] else 0.5
    return out, final_rates


def fit_encoders(match_feat: pd.DataFrame) -> tuple[LabelEncoder, LabelEncoder]:
    """Fit venue and team encoders on strong/weak teams and venues."""
    le_venue = LabelEncoder()
    le_team = LabelEncoder()
    venues = match_feat["venue"].astype(str).fillna("unknown")
    teams = pd.concat(
        [
            match_feat["team1"],
            match_feat["team2"],
            match_feat["strong_team"],
            match_feat["weak_team"],
            match_feat["toss_winner"],
        ],
        ignore_index=True,
    ).astype(str)
    le_venue.fit(venues)
    le_team.fit(teams)
    return le_team, le_venue


def encode_match_features(match_feat: pd.DataFrame, le_team: LabelEncoder, le_venue: LabelEncoder) -> pd.DataFrame:
    """Add encoded columns."""
    out = match_feat.copy()
    out["venue_enc"] = le_venue.transform(out["venue"].astype(str).fillna("unknown"))
    out["strong_team_enc"] = le_team.transform(out["strong_team"].astype(str))
    out["weak_team_enc"] = le_team.transform(out["weak_team"].astype(str))
    out["toss_is_strong_team"] = out["toss_is_strong_team"].astype(int)
    out["toss_bat"] = out["toss_bat"].astype(int)
    return out


MATCH_FEATURE_COLUMNS = [
    "venue_enc",
    "strong_team_enc",
    "weak_team_enc",
    "strong_team_winrate",
    "weak_team_winrate",
    "winrate_diff",
    "toss_is_strong_team",
    "toss_bat",
]


def save_match_artifacts(
    le_team: LabelEncoder,
    le_venue: LabelEncoder,
    win_rates: dict[str, float],
    feature_columns: list[str],
    directory: str | Path,
) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(le_team, d / "le_team.joblib")
    joblib.dump(le_venue, d / "le_venue.joblib")
    joblib.dump(feature_columns, d / "match_feature_columns.joblib")
    with open(d / "win_rates.json", "w", encoding="utf-8") as f:
        json.dump(win_rates, f, indent=2)


def load_win_rates(path: str | Path) -> dict[str, float]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
