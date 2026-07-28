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
2021-22 → 2023-24; XGBoost hyperparameters are selected on the whole
2024-25 season (**validation**); the final model is then scored **exactly once**
on the untouched 2025-26 season (~26k player-games, **test**). 2020-21 is
ingested only to seed team play-style clusters and is never modeled on.

MAE is the headline metric. "Within" uses a target-specific tolerance
(points ±3, rebounds/assists ±2) since a fixed ±3 is not comparable across stats.

| Target   | Test MAE | Naive last-5 MAE | Improvement | Within |
|----------|---------:|-----------------:|------------:|-------:|
| Points   |    4.610 |            4.802 |      +3.99% | 41.1% (±3) |
| Rebounds |    1.917 |            1.992 |      +3.74% | 62.3% (±2) |
| Assists  |    1.372 |            1.401 |      +2.08% | 78.3% (±2) |

Reproduce: `python scripts/evaluate.py`

### Probabilistic forecasting (calibrated intervals)

Beyond a single number, CourtVision reports an **80% prediction interval** and
empirical **over/under probabilities** via split-conformal calibration. Because
probabilities need calibration data the point model has never seen, 2024-25 is
split in half: hyperparameters are chosen on the **first half**, the point model
is refit on 2021-24 + that first half, and its errors on the **unseen second
half** become the calibration residuals. The intervals are then scored **exactly
once** on 2025-26. Coverage near 80% means the intervals are trustworthy;
narrower average width at the same coverage is more informative.

| Target   | Point MAE | Requested | Actual coverage | Avg. width | Conformal radius |
|----------|----------:|----------:|----------------:|-----------:|-----------------:|
| Points   |     4.612 |       80% |           80.6% |      13.65 |          ±7.267 |
| Rebounds |     1.908 |       80% |           81.4% |       5.73 |          ±3.039 |
| Assists  |     1.353 |       80% |           80.0% |       3.65 |          ±2.061 |

**Known limitation (motivates quantile regression next).** A single global radius
per stat is too wide for low-minute bench players and too narrow for high-minute
starters. Coverage by expected minutes (proxied by `min_last5`) on 2025-26:

| Expected minutes | Points | Rebounds | Assists |
|------------------|-------:|---------:|--------:|
| <15 min          |  92.7% |    88.5% |   92.1% |
| 15–25 min        |  82.9% |    82.1% |   83.9% |
| 25–35 min        |  73.6% |    77.5% |   72.3% |
| 35+ min          |  65.0% |    75.1% |   60.0% |

Over/under probability quality on 2025-26 (Brier score, lower is better):

| Target   | Line | Brier | Line | Brier | Line | Brier |
|----------|-----:|------:|-----:|------:|-----:|------:|
| Points   | 14.5 | 0.132 | 19.5 | 0.088 | 24.5 | 0.052 |
| Rebounds |  4.5 | 0.168 |  6.5 | 0.116 |  8.5 | 0.068 |
| Assists  |  3.5 | 0.128 |  5.5 | 0.073 |  7.5 | 0.039 |

Reproduce: `python scripts/calibrate_uncertainty.py` then
`python scripts/evaluate_uncertainty.py`. Live example:
`python scripts/predict_today.py --player "Stephen Curry" --opp BOS --target target_pts --line 27.5`

### Two-stage minutes ablation (direct vs. minutes × rate)

We tested whether decomposing a projection into **expected minutes × expected
per-minute rate** (two-stage) beats the **direct** XGBoost model that predicts
the stat in one shot. Walk-forward validation (3 rolling-origin folds, 2025-26
never touched):

| Model                | Points | Rebounds | Assists |
|----------------------|-------:|---------:|--------:|
| Last-five baseline   |  4.812 |    2.000 |   1.388 |
| **Current XGBoost**  |  4.675 |    1.919 |   1.339 |
| Minutes × rate       |  4.904 |    2.000 |   1.373 |
| Blended (per-target) |  4.669 |    1.917 |   1.336 |

**Verdict: keep the direct model as default.** The two-stage model is worse in
2 of 3 stats (points +4.9%, assists +2.5%) and ties on rebounds. The optimal
blend weights (w=0.9, 0.9, 0.8 for pts/reb/ast) heavily favor the direct model,
and the blended MAE gain is negligible noise (≤0.003). The learned minutes
model cannot capture exogenous information (injury returns, confirmed rotations,
blowout risk) that a human can — so instead of auto-deploying two-stage, we ship
an opt-in `--expected-minutes` lever (see below).

Reproduce: `python scripts/ablation_minutes.py`

### Opt-in `--expected-minutes` override

When you have exogenous minutes knowledge (confirmed rotation, injury return,
blowout risk), pass `--expected-minutes` to see an alternative **minutes ×
per-minute-rate** projection alongside the default direct projection. The
direct projection and its calibrated interval are always shown unchanged.

```bash
# Train rate models (one-time, after calibrate_uncertainty.py)
python scripts/train_rate_models.py

# Smoke test: Curry vs BOS, 34 expected minutes, 25.5 line
python scripts/predict_today.py --player "Stephen Curry" --opp BOS --home \
  --date 2025-03-15 --target target_pts --line 25.5 --expected-minutes 34
```

Output:
```
==================================================
  Stephen Curry  vs  BOS  (HOME)   2025-03-15
--------------------------------------------------
  Projected pts:             25.84
  80% interval:               18.57–33.10
  [exp] 34 min x rate:  24.05
  Requested line:              25.5
  Probability over:            47.8%
  Probability under:           52.2%
  Last-5 baseline:             27.0
  Games of history:            314
==================================================
```

The `[exp]` line is the two-stage projection; everything else comes from the
untouched direct model.

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
- **Probabilistic forecasting** — split-conformal calibration on an unseen
  season turns each point projection into an 80% prediction interval plus
  empirical over/under probabilities, evaluated by coverage, width and Brier
  score rather than assuming a normal distribution.
- **Two-stage minutes ablation** — walk-forward comparison of direct vs.
  minutes × per-minute-rate vs. blended models. Verdict: direct wins; two-stage
  kept as an opt-in `--expected-minutes` lever for when exogenous minutes
  knowledge is available.
- **Recency & volatility features** — last-3 averages, rolling std (5/10-game),
  exponentially weighted means, and last-game minutes capture short-term form
  and consistency alongside the existing last-5/last-10 rolling means.
- **Train / validation / test methodology** — split by full season
  (train=2021-24, validation=2024-25, test=2025-26); hyperparameters are
  chosen on validation and the untouched test set is scored once.
- **Daily prediction pipeline** — fetches a date's schedule (`ScoreboardV2`),
  reconstructs each player's features from games *before* that date, and outputs
  projections for the slate.
- **Prediction logging + monitoring** — projections are stored, later reconciled
  against real box scores, and summarized (model MAE, baseline MAE, % within
  target-specific tolerance).

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
  calibrate_uncertainty.py  evaluate_uncertainty.py
  ablation_minutes.py  train_rate_models.py
```

## Roadmap
- ~~Probabilistic forecasts (prediction intervals + P(over/under thresholds))~~
- ~~Two-stage minutes model ablation~~ (verdict: direct wins; override shipped)
- Conformalized quantile regression for heteroskedastic intervals (wider for
  high-minute starters, narrower for bench players)
- Real injury / confirmed-lineup feed (replace the minutes-based roster proxy)
- Pace-adjusted defensive rating and role features
- Expanded hyperparameter grid + early stopping with walk-forward model selection
- SHAP explanations per prediction
- FastAPI + dashboard, automated nightly retraining
