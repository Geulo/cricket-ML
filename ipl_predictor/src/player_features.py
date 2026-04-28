"""
Player-match aggregates and **leakage-safe** rolling features.

Leakage policy (training rows):

- Every rolling mean, EWM, expanding average, H2H average, season-to-date average, and
  opponent-strength series applies ``shift(1)`` **before** rolling/expanding so the feature
  vector for match *t* uses **only matches strictly before *t*** for that player (or team track).
- ``opp_strength`` merges prior rolling opponent bowling economy from ``bowling_team`` ordered by
  time; the current ball/match does not inflate that statistic.

Inference uses **saved latest rows per player** from historical training—there is no future data,
and no incoming fixture has ball-by-ball rows (playing XI unknown).
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


BAT_FEATURE_COLUMNS = [
    "runs_l3",
    "runs_l5",
    "runs_l10",
    "runs_ewm5",
    "trend",
    "career",
    "runs_std5",
    "h2h",
    "season_avg",
    "opp_strength",
]

BOWL_FEATURE_COLUMNS = [
    "wkts_l3",
    "wkts_l5",
    "wkts_l10",
    "eco_ewm5",
    "trend",
    "career",
    "wkts_std5",
]


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for player features: {missing}")


def aggregate_batsmen(df: pd.DataFrame) -> pd.DataFrame:
    """One row per match_id + batter."""
    need = ["match_id", "batter", "batting_team", "bowling_team", "runs_batter", "batter_balls", "dt"]
    _ensure_cols(df, need)

    bat = (
        df.groupby(["match_id", "batter", "batting_team", "bowling_team"], sort=False)
        .agg(
            runs=("runs_batter", "sum"),
            balls=("batter_balls", "max"),
            fours=("runs_batter", lambda x: float((x == 4).sum())),
            sixes=("runs_batter", lambda x: float((x == 6).sum())),
            dt=("dt", "max"),
        )
        .reset_index()
    )
    bat["sr"] = np.where(bat["balls"] > 0, 100 * bat["runs"] / bat["balls"], 0.0)
    bat["boundary_runs"] = bat["fours"] * 4 + bat["sixes"] * 6
    bat["score"] = (
        bat["runs"]
        + 0.5 * bat["boundary_runs"]
        + 0.5 * np.maximum(bat["sr"] - 100, 0)
    )
    return bat


def aggregate_bowlers(df: pd.DataFrame) -> pd.DataFrame:
    """One row per match_id + bowler."""
    need = ["match_id", "bowler", "bowling_team", "batting_team", "bowler_wicket", "valid_ball", "runs_bowler", "dt"]
    _ensure_cols(df, need)

    bowl = (
        df.groupby(["match_id", "bowler", "bowling_team", "batting_team"], sort=False)
        .agg(
            wkts=("bowler_wicket", "max"),
            balls=("valid_ball", "sum"),
            runs=("runs_bowler", "sum"),
            dt=("dt", "max"),
        )
        .reset_index()
    )
    bowl["eco"] = np.where(bowl["balls"] > 0, bowl["runs"] / (bowl["balls"] / 6.0), 0.0)
    bowl["score"] = bowl["wkts"] * 20 + 1.5 * np.maximum(9 - bowl["eco"], 0)
    return bowl


def add_quartile_rank_labels(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    """Within each match, quartile labels 0–3 with qcut fallback."""

    def label_one(scores: pd.Series) -> pd.Series:
        n = len(scores)
        if n <= 1:
            return pd.Series([2] * n, index=scores.index, dtype=int)
        try:
            labs = pd.qcut(scores.astype(float), q=4, labels=False, duplicates="drop")
            labs = pd.Series(labs, index=scores.index).fillna(0)
            mx = labs.max()
            if pd.notna(mx) and mx > 3:
                labs = (labs / mx * 3).round().clip(0, 3).astype(int)
            else:
                labs = labs.astype(int)
            return labs
        except (ValueError, TypeError):
            rk = scores.rank(ascending=False, method="first")
            buckets = pd.cut(rk, bins=min(4, n), labels=False, duplicates="drop")
            buckets = pd.Series(buckets, index=scores.index).fillna(0).astype(float)
            mx = buckets.max()
            if mx > 0:
                buckets = (buckets / mx * 3).round().astype(int)
            return buckets.astype(int)

    out = df.copy()
    parts: list[pd.Series] = []
    for _, grp in df.groupby("match_id", sort=False):
        parts.append(label_one(grp[score_col]))
    out["rank"] = pd.concat(parts).sort_index()
    return out


def engineer_batsmen_features(bat: pd.DataFrame) -> pd.DataFrame:
    """Rolling and contextual batsman features (past-only via shift)."""
    bat = bat.sort_values(["batter", "dt"]).reset_index(drop=True)

    g = bat.groupby("batter", sort=False)["runs"]
    for w in (3, 5, 10):
        bat[f"runs_l{w}"] = g.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

    bat["runs_ewm5"] = g.transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())
    bat["trend"] = bat["runs_l3"] - bat["runs_l10"]
    bat["career"] = g.transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    bat["runs_std5"] = g.transform(lambda x: x.shift(1).rolling(5, min_periods=1).std())

    bat["season"] = bat["dt"].dt.year
    bat["season_avg"] = bat.groupby(["batter", "season"], sort=False)["runs"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    h2h_raw = bat.groupby(["batter", "bowling_team"], sort=False)["runs"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )
    bat["h2h_raw"] = h2h_raw
    bat["h2h"] = bat["h2h_raw"].fillna(bat["career"])
    return bat


def engineer_bowlers_features(bowl: pd.DataFrame) -> pd.DataFrame:
    bowl = bowl.sort_values(["bowler", "dt"]).reset_index(drop=True)
    gw = bowl.groupby("bowler", sort=False)["wkts"]
    for w in (3, 5, 10):
        bowl[f"wkts_l{w}"] = gw.transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())

    bowl["eco_ewm5"] = bowl.groupby("bowler", sort=False)["eco"].transform(
        lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
    )
    bowl["trend"] = bowl["wkts_l3"] - bowl["wkts_l10"]
    bowl["career"] = gw.transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    bowl["wkts_std5"] = gw.transform(lambda x: x.shift(1).rolling(5, min_periods=1).std())
    return bowl


def attach_batsman_opp_strength(bat: pd.DataFrame, bowl: pd.DataFrame) -> pd.DataFrame:
    """Opp strength = rolling mean economy of opponent bowling team (past matches)."""
    team_eco = bowl.groupby(["match_id", "bowling_team"], sort=False)["eco"].mean().reset_index()
    team_eco = team_eco.sort_values(["bowling_team", "match_id"])
    team_eco["opp_strength"] = team_eco.groupby("bowling_team", sort=False)["eco"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    bat = bat.drop(columns=["opp_strength"], errors="ignore")
    bat = bat.merge(team_eco[["match_id", "bowling_team", "opp_strength"]], on=["match_id", "bowling_team"], how="left")
    bat["opp_strength"] = bat["opp_strength"].fillna(0.0)
    return bat


def build_player_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full batsman/bowler feature tables with labels.

    Returns ``bat_feat``, ``bowl_feat`` sorted by time within player for downstream splits.
    """
    bat = aggregate_batsmen(df)
    bowl = aggregate_bowlers(df)
    bat = engineer_batsmen_features(bat)
    bat = attach_batsman_opp_strength(bat, bowl)
    bowl = engineer_bowlers_features(bowl)

    bat = add_quartile_rank_labels(bat)
    bowl = add_quartile_rank_labels(bowl)

    bat = bat.sort_values(["match_id", "dt"]).reset_index(drop=True)
    bowl = bowl.sort_values(["match_id", "dt"]).reset_index(drop=True)
    return bat, bowl


def extract_latest_player_rows(bat: pd.DataFrame, bowl: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Latest feature row per player for inference-time snapshots."""
    bat_last = bat.sort_values(["batter", "dt"]).groupby("batter", sort=False).tail(1)
    bowl_last = bowl.sort_values(["bowler", "dt"]).groupby("bowler", sort=False).tail(1)
    for c in BAT_FEATURE_COLUMNS:
        if c not in bat_last.columns:
            bat_last[c] = 0.0
    for c in BOWL_FEATURE_COLUMNS:
        if c not in bowl_last.columns:
            bowl_last[c] = 0.0
    bat_cols = ["batter", "batting_team"] + BAT_FEATURE_COLUMNS
    bowl_cols = ["bowler", "bowling_team"] + BOWL_FEATURE_COLUMNS
    return {"bat": bat_last[bat_cols].copy(), "bowl": bowl_last[bowl_cols].copy()}


def build_team_player_pools(
    bat: pd.DataFrame,
    bowl: pd.DataFrame,
    recent_matches: int = 40,
) -> dict[str, dict[str, list[str]]]:
    """Players seen for each team in the last ``recent_matches`` distinct matches."""
    mids = bat["match_id"].sort_values().unique()
    last_m = set(mids[max(0, len(mids) - recent_matches) :])

    bat_r = bat[bat["match_id"].isin(last_m)]
    bowl_r = bowl[bowl["match_id"].isin(last_m)]

    pools: dict[str, dict[str, list[str]]] = {}
    for side in bat_r["batting_team"].dropna().unique():
        b_players = bat_r[bat_r["batting_team"] == side]["batter"].dropna().unique().tolist()
        bw_players = bowl_r[bowl_r["bowling_team"] == side]["bowler"].dropna().unique().tolist()
        pools[str(side)] = {"batsmen": sorted(set(b_players)), "bowlers": sorted(set(bw_players))}
    return pools


def save_player_artifacts(
    latest_history: dict[str, pd.DataFrame],
    team_pool: dict[str, dict[str, list[str]]],
    feature_columns: dict[str, list[str]],
    directory: str | Path,
) -> None:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_columns, d / "player_feature_columns.joblib")
    joblib.dump(latest_history, d / "latest_player_history.joblib")
    joblib.dump(team_pool, d / "team_player_pool.joblib")

