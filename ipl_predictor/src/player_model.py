"""Train LambdaRank models for batsmen and bowlers."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.player_features import BAT_FEATURE_COLUMNS, BOWL_FEATURE_COLUMNS


def split_player_holdout(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train before 2024 / test 2024 if present; else last season as test."""
    y = df["dt"].dt.year
    if (y == 2024).any():
        return df[y < 2024].copy(), df[y == 2024].copy()
    ly = int(y.max())
    return df[y < ly].copy(), df[y == ly].copy()


def _eval_ranking(df: pd.DataFrame, player_col: str, score_col: str = "score") -> dict[str, float]:
    best_hits = []
    top3_hits = []

    for _, g in df.groupby("match_id", sort=False):
        if len(g) < 2:
            continue
        pred_best = g.loc[g["pred"].idxmax(), player_col]
        actual_best = g.loc[g[score_col].idxmax(), player_col]
        pred_top3 = set(g.nlargest(3, "pred")[player_col])
        actual_top3 = set(g.nlargest(3, score_col)[player_col])
        best_hits.append(int(pred_best == actual_best))
        top3_hits.append(int(len(pred_top3 & actual_top3) > 0))

    rng_base = []
    rng = np.random.RandomState(42)
    for _, g in df.groupby("match_id", sort=False):
        if len(g) < 2:
            continue
        players = g[player_col].tolist()
        rng_pick = players[int(rng.randint(0, len(players)))]
        actual_best = g.loc[g[score_col].idxmax(), player_col]
        rng_base.append(int(rng_pick == actual_best))

    return {
        "exact_best": float(np.mean(best_hits)) if best_hits else 0.0,
        "top3_hit": float(np.mean(top3_hits)) if top3_hits else 0.0,
        "random_baseline": float(np.mean(rng_base)) if rng_base else 0.0,
    }


def train_ranker(
    train_df: pd.DataFrame,
    feats: list[str],
    label_col: str = "rank",
) -> lgb.LGBMRanker:
    train_df = train_df.sort_values("match_id", kind="mergesort")
    groups = train_df.groupby("match_id", sort=False).size().values
    X = train_df[feats].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = train_df[label_col].astype(int)
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=800,
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y, group=groups)
    return model


def train_player_models(
    bat_feat: pd.DataFrame,
    bowl_feat: pd.DataFrame,
    models_dir: str | Path,
) -> dict[str, dict[str, float]]:
    """Train batsman/bowler rankers, evaluate on held-out year, save artifacts."""

    bat_train, bat_test = split_player_holdout(bat_feat)
    bowl_train, bowl_test = split_player_holdout(bowl_feat)

    bat_model = train_ranker(bat_train, BAT_FEATURE_COLUMNS)
    bowl_model = train_ranker(bowl_train, BOWL_FEATURE_COLUMNS)

    bat_test = bat_test.copy()
    bowl_test = bowl_test.copy()
    bat_test["pred"] = bat_model.predict(bat_test[BAT_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0))
    bowl_test["pred"] = bowl_model.predict(bowl_test[BOWL_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0))

    bat_metrics = _eval_ranking(bat_test, "batter")
    bowl_metrics = _eval_ranking(bowl_test, "bowler")

    print("=== Player rankers (train before 2024 / test 2024 fallback last season) ===")
    print(f"Batsman — rows train/test: {len(bat_train)}/{len(bat_test)}")
    print(f"  exact_best={bat_metrics['exact_best']:.4f} top3={bat_metrics['top3_hit']:.4f} random={bat_metrics['random_baseline']:.4f}")
    print(f"Bowler — rows train/test: {len(bowl_train)}/{len(bowl_test)}")
    print(f"  exact_best={bowl_metrics['exact_best']:.4f} top3={bowl_metrics['top3_hit']:.4f} random={bowl_metrics['random_baseline']:.4f}")

    out = Path(models_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bat_model, out / "bat.joblib")
    joblib.dump(bowl_model, out / "bowl.joblib")

    metrics = {"batsman": bat_metrics, "bowler": bowl_metrics}
    with open(out / "player_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
