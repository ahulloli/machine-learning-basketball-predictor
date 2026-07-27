# CourtVision

An end-to-end NBA **player-game prediction system**. It ingests **real** NBA box
scores, engineers leakage-safe features, trains XGBoost models to forecast a
player's points / rebounds / assists for upcoming games, and continuously
monitors its own accuracy against a naive baseline.

```
                          COURTVISION
NBA.com
   │
   ▼
nba_api
   │
   ▼
PostgreSQL
   │
   ├────────────────────┐
   ▼                    ▼
Historical Games     Upcoming Schedule
   │                    │
   ▼                    │
Feature Engineering     │
   │                    │
   ▼                    │
XGBoost Training        │
   │                    │
   ▼                    ▼
Saved Models ─────► Prediction Engine
                       │
                       ▼
                 Tonight's Forecasts
                       │
                       ▼
                   NBA Results
                       │
                       ▼
                 Reconciliation
                       │
                       ▼
                Model Monitoring
```

## Held-out results

Strict chronological **train / validation / test** split. Models train on
2021-22 → 2023-24; XGBoost hyperparameters are selected on the first half of
2024-25 (**validation**); the model is then scored **exactly once** on the
untouched second half of 2024-25 (~13k player-games, **test**). 2020-21 is
ingested only to seed team play-style clusters and is never modeled on.

MAE is the headline metric. "Within" uses a target-specific tolerance
(points ±3, rebounds/assists ±2) since a fixed ±3 is not comparable across stats.

| Target   | Test MAE | Naive last-5 MAE | Improvement | Within |
|----------|---------:|-----------------:|------------:|-------:|
| Points   |    4.690 |            4.891 |      +4.11% | 40.8% (±3) |
| Rebounds |    1.976 |            2.075 |      +4.74% | 61.6% (±2) |
| Assists  |    1.369 |            1.416 |      +3.32% | 78.6% (±2) |

Reproduce: `python scripts/evaluate.py 2024-25`

## What's implemented

- **Real data ingestion** — one `LeagueGameLog` call per season pulls every
  player-game box score (points, rebounds, assists, minutes, shooting, etc.)
  into PostgreSQL. Idempotent upserts, so re-running mid-season adds new games.
- **Leakage-safe feature engineering** — rolling last-5 / last-10 form and
  per-minute efficiency, rest days, opponent defensive rating, and
  **target-specific** player-vs-opponent and player-vs-similar-team history for
  each of points / rebounds / assists.
- **Stable, leakage-free team clustering** — one KMeans (fit once on the context
  season) assigns each team its cluster from the *previous* season's profile, so
  cluster ids mean the same play-style every year and no current-season info leaks.
- **XGBoost models** for points / rebounds / assists, each compared against a
  naive last-5 baseline.
- **Train / validation / test methodology** — split strictly by date;
  hyperparameters are chosen on validation and the test set is scored once.
- **Daily prediction pipeline** — fetches a date's schedule (`ScoreboardV2`),
  reconstructs each player's features from games *before* that date, and outputs
  projections for the slate.
- **Prediction logging + monitoring** — projections are stored, later reconciled
  against real box scores, and summarized (model MAE, baseline MAE, % within 3).

## Setup

Requires a running local PostgreSQL (tested with `postgresql@15`) and Python 3.13.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # edit DATABASE_URL if needed
createdb courtvision          # or: psql -c 'CREATE DATABASE courtvision;'
python scripts/init_db.py
```

## Usage

```bash
# Ingest multiple seasons (2020-21 seeds clusters; 2021-22+ are modeled)
python scripts/ingest_season.py 2020-21
python scripts/ingest_season.py 2021-22
python scripts/ingest_season.py 2022-23
python scripts/ingest_season.py 2023-24
python scripts/ingest_season.py 2024-25

# Build features across all seasons
python scripts/build_features.py

# Credible held-out evaluation (train/validation/test)
python scripts/evaluate.py 2024-25

# Train + persist deployable models (points/rebounds/assists)
python scripts/train_model.py all target_pts
python scripts/train_model.py all target_reb
python scripts/train_model.py all target_ast

# Predict a single matchup or a full slate
python scripts/predict_today.py --player "Stephen Curry" --opp BOS --date 2025-01-15
python scripts/predict_today.py --slate --date 2025-01-15

# Log predictions, reconcile with results, and view the monitoring report
python scripts/monitor.py log-slate --date 2025-01-15 --target target_pts
python scripts/monitor.py reconcile
python scripts/monitor.py report
```

## Schema

- `teams` — team_id, abbreviation, name
- `players` — player_id, name
- `games` — game_id, game_date, season
- `player_game_stats` — one row per player per game (real box score)
- `player_game_features` — engineered features + `target_*` actuals
- `predictions` — logged projections + baseline + actual + errors (monitoring)

### Why "leakage-safe"?
Every rolling / historical feature is computed from a player's or team's games
**before** the game being predicted (`shift(1)` before rolling; expanding means
over prior meetings only), so a game's own result never leaks into its own
inputs. Team clusters for season *S* are fit on season *S-1*.

## Project layout
```
src/courtvision/
  config.py     # env / .env loading (DATABASE_URL)
  db.py         # SQLAlchemy engine + session
  models.py     # ORM schema (stats, features, predictions)
  ingest.py     # LeagueGameLog → player_game_stats
  features.py   # feature engineering + team clustering → player_game_features
  train.py      # XGBoost training + train/validation/test evaluation
  predict.py    # live feature assembly + daily prediction pipeline
  monitor.py    # prediction logging, reconciliation, accuracy reporting
scripts/
  init_db.py  ingest_season.py  build_features.py
  train_model.py  evaluate.py  predict_today.py  monitor.py
```

## Roadmap
- Probabilistic forecasts (prediction intervals + P(over/under thresholds))
- SHAP explanations per prediction
- Real injury / confirmed-lineup feed (replace the minutes-based roster proxy)
- FastAPI + dashboard, automated nightly retraining
