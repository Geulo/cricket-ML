# IPL Prediction Suite

Unified pipeline that predicts **match winners** (GradientBoosting on strong/weak–team features) and **top-N batsmen and bowlers** (LightGBM LambdaRank on leakage-safe player histories).

**Workflow:** clone → install dependencies → place **real** Kaggle IPL data at `data/IPL.csv` → `python train.py` → `pytest` → `uvicorn api.main:app`. You do **not** need to open the notebooks under `notebooks/` for training or inference.

## Critical: real data vs smoke tests

| Data source | Purpose |
|-------------|---------|
| **[Kaggle IPL ball-by-ball CSV](https://www.kaggle.com/datasets/chaitu20/ipl-dataset2008-2025)** placed at **`data/IPL.csv`** | **Required** for meaningful models and any honest performance discussion. |
| **`tests/fixtures/synthetic_smoke.csv`** (regenerate with `python scripts/gen_synthetic_ipl.py`) | **pytest / CI smoke tests only.** Metrics from training on this file are **not** valid indicators of real IPL performance—do **not** present them as final results. |

## Project layout

- `data/IPL.csv` — **your** ball-by-ball IPL file from Kaggle (you create this path).
- `src/` — feature builders and models (production code).
- `models/` — artifacts produced by `train.py` (ignored from scratch until you train).
- `api/` — FastAPI service.
- `train.py` — single training entrypoint.
- `tests/fixtures/` — synthetic CSV **only** for automated smoke tests.

## Setup

```bash
cd ipl_predictor
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On **macOS**, LightGBM usually needs OpenMP: `brew install libomp` so `import lightgbm` loads its native library.

### Real dataset (required for serious training)

1. Download the IPL dataset from Kaggle (linked above).
2. Save it as **`ipl_predictor/data/IPL.csv`**.
3. Required columns include `match_id`, `dt` or `date`, `venue`, `toss_winner`, `toss_decision`, `batting_team`, `bowling_team`, `match_won_by`, `season`, `batter`, `bowler`, and numeric ball columns such as `runs_batter`, `batter_balls`, `bowler_wicket`, `valid_ball`, `runs_bowler`, `runs_total`, `innings`, `bat_pos`.

Optional override:

```bash
export IPL_CSV_PATH=/path/to/your/IPL.csv
python train.py
```

### Smoke-test CSV (optional, not for performance claims)

Regenerate the bundled pytest fixture:

```bash
python scripts/gen_synthetic_ipl.py
# default: tests/fixtures/synthetic_smoke.csv
```

## Train all models

From **`ipl_predictor/`**:

```bash
python train.py
```

This prints validation-style metrics for development—interpret them **only** when trained on **real** `data/IPL.csv`.

### Saved artifacts (after `train.py`)

| File | Description |
|------|-------------|
| `gb.joblib` | Match winner GradientBoosting model |
| `bat.joblib` / `bowl.joblib` | LambdaRank rankers |
| `le_team.joblib` / `le_venue.joblib` | Encoders |
| `match_feature_columns.joblib` | Match model feature order |
| `player_feature_columns.joblib` | Player feature lists |
| `latest_player_history.joblib` | Latest per-player feature snapshots |
| `team_player_pool.joblib` | Recent players per franchise |
| `win_rates.json` | Historical win rates for strong/weak assignment |
| `betting_calibration.joblib` / `bowl_betting_calibration.joblib` | **Optional** (experimental batsman / bowler gap calibration); inference does **not** depend on them |
| `match_metrics.json` / `player_metrics.json` | Training summaries |

### Match model split

The match model uses a **chronological** train/test split (~80% / ~20%). Random splits would mix future and past—this project avoids that by default.

## Run tests

```bash
cd ipl_predictor
pytest tests -q
```

Pytest sets `IPL_CSV_PATH` to `tests/fixtures/synthetic_smoke.csv` when that file exists.

## Run the API

```bash
cd ipl_predictor
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Example: `POST /predict` — Mumbai Indians vs Chennai Super Kings

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "team_1": "Mumbai Indians",
    "team_2": "Chennai Super Kings",
    "venue": "Wankhede Stadium",
    "toss_winner": "Mumbai Indians",
    "toss_decision": "bat",
    "top_n": 3
  }'
```

Team and venue strings must match labels seen during training—use `GET /teams` and `GET /venues`.

## Limitations

- **No confirmed playing XI** before the official lineup; rankings use **recent historical player availability** inferred from training data.
- **Sports outcomes are noisy**; match accuracy is often only modestly above simple baselines.
- **Betting calibration** (optional gap/isotonic helpers in `betting_calibration.joblib` and `bowl_betting_calibration.joblib`) is **experimental**, sample-dependent, and **not** a promise of ROI.
- **Synthetic smoke data** (`tests/fixtures/synthetic_smoke.csv`) is **invalid** for final performance claims—always train on the **real Kaggle CSV** for coursework reporting.

## Notebooks

Course notebooks under `notebooks/` are reference-only; all runnable logic is in `src/`, `train.py`, and `api/`.
