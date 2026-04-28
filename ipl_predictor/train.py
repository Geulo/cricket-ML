#!/usr/bin/env python3
"""Train match winner model and player rankers; save all artifacts under models/."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve_data_csv() -> Path:
    """Training CSV path: ``IPL_CSV_PATH`` env override, else ``data/IPL.csv``."""
    env = os.environ.get("IPL_CSV_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return (ROOT / "data" / "IPL.csv").resolve()


def verify_saved_artifacts(models_dir: Path) -> None:
    """Assert expected files exist after a successful training run."""
    expected = [
        "gb.joblib",
        "bat.joblib",
        "bowl.joblib",
        "le_team.joblib",
        "le_venue.joblib",
        "match_feature_columns.joblib",
        "win_rates.json",
        "player_feature_columns.joblib",
        "latest_player_history.joblib",
        "team_player_pool.joblib",
    ]
    missing = [name for name in expected if not (models_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Expected artifacts missing after train: {missing}")
    opt_bat = models_dir / "betting_calibration.joblib"
    opt_bowl = models_dir / "bowl_betting_calibration.joblib"
    bat_st = "present" if opt_bat.exists() else "absent (optional)"
    bowl_st = "present" if opt_bowl.exists() else "absent (optional)"
    print(
        f"\nArtifact check: all required files OK.\n"
        f"  betting_calibration.joblib (bat): {bat_st}\n"
        f"  bowl_betting_calibration.joblib: {bowl_st}\n"
    )


def main() -> None:
    import numpy as np

    import joblib

    from src.data import clean_data, load_data
    from src.match_features import (
        aggregate_matches,
        build_match_features_table,
        encode_match_features,
        fit_encoders,
    )
    from src.match_model import train_match_model
    from src.player_features import (
        BAT_FEATURE_COLUMNS,
        BOWL_FEATURE_COLUMNS,
        build_player_features,
        build_team_player_pools,
        extract_latest_player_rows,
        save_player_artifacts,
    )
    from src.player_model import split_player_holdout, train_player_models

    data_path = _resolve_data_csv()
    models_dir = ROOT / "models"

    if "synthetic_smoke" in data_path.name:
        print(
            "\n*** WARNING: Training on bundled synthetic smoke CSV. "
            "These metrics are NOT valid for reporting real IPL performance. "
            "Replace with Kaggle data at data/IPL.csv. ***\n"
        )

    raw = load_data(data_path)
    df = clean_data(raw)

    match_agg = aggregate_matches(df)
    mf, win_rates = build_match_features_table(match_agg)
    le_team, le_venue = fit_encoders(mf)
    encoded_match = encode_match_features(mf, le_team, le_venue)

    print("\n--- Training match model ---\n")
    match_metrics = train_match_model(encoded_match, le_team, le_venue, win_rates, models_dir=models_dir)

    print("\n--- Building player features ---\n")
    bat_feat, bowl_feat = build_player_features(df)

    latest_hist = extract_latest_player_rows(bat_feat, bowl_feat)
    pools = build_team_player_pools(bat_feat, bowl_feat)
    save_player_artifacts(
        latest_hist,
        pools,
        {"bat": BAT_FEATURE_COLUMNS, "bowl": BOWL_FEATURE_COLUMNS},
        models_dir,
    )

    print("\n--- Training player rankers ---\n")
    player_metrics = train_player_models(bat_feat, bowl_feat, models_dir=models_dir)

    # Optional isotonic calibration from test-split rank gaps (experimental; batsman then bowler).
    try:
        from src.betting import build_calibration_from_validation, normalized_top_gap

        _, bat_test = split_player_holdout(bat_feat)

        gaps_bat: list[float] = []
        hits_bat: list[int] = []

        bat_m = joblib.load(models_dir / "bat.joblib")
        for _, g in bat_test.groupby("match_id", sort=False):
            if len(g) < 2:
                continue
            X = g[BAT_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
            preds = bat_m.predict(X)
            gaps_bat.append(normalized_top_gap(preds))
            top_player = g.loc[g["score"].idxmax(), "batter"]
            pred_best = g.iloc[int(np.argmax(preds))]["batter"]
            hits_bat.append(int(top_player == pred_best))

        gbat = np.array(gaps_bat)
        hbat = np.array(hits_bat)
        iso_bat, thr_bat, warn_bat = build_calibration_from_validation(gbat, hbat)

        joblib.dump(
            {"isotonic": iso_bat, "threshold": thr_bat, "warning": warn_bat},
            models_dir / "betting_calibration.joblib",
        )
        if warn_bat:
            print(f"[betting bat] {warn_bat}")
    except Exception as exc:
        print(f"[betting bat] calibration skipped: {exc}")

    try:
        from src.betting import build_calibration_from_validation, normalized_top_gap

        _, bowl_test = split_player_holdout(bowl_feat)

        gaps_bowl: list[float] = []
        hits_bowl: list[int] = []

        bowl_m = joblib.load(models_dir / "bowl.joblib")
        for _, g in bowl_test.groupby("match_id", sort=False):
            if len(g) < 2:
                continue
            X = g[BOWL_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).fillna(0)
            preds = bowl_m.predict(X)
            gaps_bowl.append(normalized_top_gap(preds))
            top_player = g.loc[g["score"].idxmax(), "bowler"]
            pred_best = g.iloc[int(np.argmax(preds))]["bowler"]
            hits_bowl.append(int(top_player == pred_best))

        gbowl = np.array(gaps_bowl)
        hbowl = np.array(hits_bowl)
        iso_bowl, thr_bowl, warn_bowl = build_calibration_from_validation(gbowl, hbowl)

        joblib.dump(
            {"isotonic": iso_bowl, "threshold": thr_bowl, "warning": warn_bowl},
            models_dir / "bowl_betting_calibration.joblib",
        )
        if warn_bowl:
            print(f"[betting bowl] {warn_bowl}")
    except Exception as exc:
        print(f"[betting bowl] calibration skipped: {exc}")

    verify_saved_artifacts(models_dir)

    print("\n=== Done ===")
    print("Match metrics:", match_metrics)
    print("Player metrics:", player_metrics)


if __name__ == "__main__":
    main()
