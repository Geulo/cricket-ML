"""Train GradientBoosting match winner model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.match_features import MATCH_FEATURE_COLUMNS, save_match_artifacts


def chronological_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last ``test_frac`` rows chronologically as test set."""
    d = df.sort_values(["dt", "match_id"], kind="mergesort").reset_index(drop=True)
    n = len(d)
    split_at = int(np.floor(n * (1 - test_frac)))
    split_at = max(1, min(split_at, n - 1))
    train = d.iloc[:split_at].copy()
    test = d.iloc[split_at:].copy()
    return train, test


def train_match_model(
    encoded_df: pd.DataFrame,
    le_team,
    le_venue,
    win_rates: dict[str, float],
    models_dir: str | Path,
    test_frac: float = 0.2,
) -> dict[str, float]:
    """
    Fit GradientBoosting on a chronological split.

    Saves ``gb.joblib``, encoders, ``match_feature_columns.joblib``, ``win_rates.json``.
    """
    drop_na = encoded_df.dropna(subset=["winner"]).copy()
    drop_na = drop_na[
        drop_na["winner"].isin(drop_na["team1"]) | drop_na["winner"].isin(drop_na["team2"])
    ]

    train_df, test_df = chronological_split(drop_na, test_frac=test_frac)

    X_train = train_df[MATCH_FEATURE_COLUMNS].astype(float).fillna(0)
    y_train = train_df["strong_team_wins"].astype(int)
    X_test = test_df[MATCH_FEATURE_COLUMNS].astype(float).fillna(0)
    y_test = test_df["strong_team_wins"].astype(int)

    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42,
    )
    gb.fit(X_train, y_train)

    pred = gb.predict(X_test)
    acc = accuracy_score(y_test, pred)
    baseline_acc = float((y_test == 1).mean())

    print("=== Match model (GradientBoosting, chronological split) ===")
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Baseline always-strong_team: {baseline_acc:.4f}")
    print(classification_report(y_test, pred, target_names=["weak_wins", "strong_wins"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, pred))

    probs = gb.predict_proba(X_test)[:, 1]
    print(f"Predicted P(strong wins) mean={probs.mean():.4f} std={probs.std():.4f}")

    out_dir = Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(gb, out_dir / "gb.joblib")

    save_match_artifacts(le_team, le_venue, win_rates, list(MATCH_FEATURE_COLUMNS), out_dir)

    dist_path = out_dir / "match_prediction_distribution.png"
    plt.figure(figsize=(6, 4))
    plt.hist(probs, bins=20, edgecolor="black")
    plt.xlabel("P(strong_team wins)")
    plt.ylabel("count")
    plt.title("Test-set predicted probability distribution")
    plt.tight_layout()
    plt.savefig(dist_path)
    plt.close()

    metrics = {
        "accuracy": float(acc),
        "baseline_strong_always": float(baseline_acc),
        "train_rows": float(len(train_df)),
        "test_rows": float(len(test_df)),
    }

    with open(out_dir / "match_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
