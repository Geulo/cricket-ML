#!/usr/bin/env python3
"""
Generate a synthetic IPL-like CSV **only for pytest smoke tests and local pipeline checks**.

This file is **not** a substitute for the real Kaggle IPL dataset. Do **not** report metrics
from models trained on this data as real predictive performance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "tests" / "fixtures" / "synthetic_smoke.csv"


def write_synthetic_csv(out: Path, num_matches: int, seed: int = 42) -> None:
    rng = np.random.RandomState(seed)
    rows = []
    match_id = 100000
    start = pd.Timestamp("2023-03-01")

    teams = [
        "Mumbai Indians",
        "Chennai Super Kings",
        "Royal Challengers Bangalore",
        "Kolkata Knight Riders",
        "Rajasthan Royals",
        "Sunrisers Hyderabad",
    ]
    venues = [
        "Wankhede Stadium",
        "M Chinnaswamy Stadium",
        "Eden Gardens",
        "Arun Jaitley Stadium",
    ]

    for m in range(num_matches):
        row_date = start + pd.Timedelta(days=m * 4)
        dt_int = int(row_date.strftime("%Y%m%d"))
        year = row_date.year
        season = str(year)

        t1, t2 = rng.choice(teams, size=2, replace=False)
        venue = venues[rng.randint(0, len(venues))]
        toss_winner = t1 if rng.random() < 0.5 else t2
        toss_decision = "bat" if rng.random() < 0.5 else "field"
        winner = t1 if rng.random() < 0.52 else t2

        for innings in (1, 2):
            batting = t1 if innings == 1 else t2
            bowling = t2 if innings == 1 else t1
            for ball in range(1, 61):
                striker = f"Player_{batting.split()[0]}_{rng.randint(1, 8)}"
                bowler = f"Bowler_{bowling.split()[0]}_{rng.randint(1, 6)}"
                rb = int(rng.choice([0, 0, 1, 1, 2, 4, 6]))
                extras = 0
                runs_total = rb + extras
                rows.append(
                    {
                        "match_id": match_id,
                        "dt": dt_int,
                        "venue": venue,
                        "toss_winner": toss_winner,
                        "toss_decision": toss_decision,
                        "batting_team": batting,
                        "bowling_team": bowling,
                        "match_won_by": winner,
                        "season": season,
                        "innings": innings,
                        "bat_pos": min(11, 1 + ball // 6),
                        "batter": striker,
                        "bowler": bowler,
                        "runs_batter": rb,
                        "batter_balls": 1,
                        "bowler_wicket": 1 if rng.random() < 0.03 else 0,
                        "valid_ball": 1,
                        "runs_bowler": rb + extras,
                        "runs_total": runs_total,
                    }
                )
        match_id += 1

    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out} ({num_matches} matches). Smoke tests only — not for performance claims.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic IPL CSV for smoke tests only.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output CSV path (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--matches",
        type=int,
        default=120,
        help="Number of synthetic matches (default: 120)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write_synthetic_csv(args.output.expanduser().resolve(), args.matches, args.seed)


if __name__ == "__main__":
    main()
