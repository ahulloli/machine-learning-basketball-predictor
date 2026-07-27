# CourtVision

NBA player-game prediction pipeline. Pulls **real** historical box scores from
NBA.com (via `nba_api`), stores them in PostgreSQL, engineers leakage-safe
rolling features, and (in later milestones) trains XGBoost to predict player
production for upcoming games.

```
NBA.com → nba_api → PostgreSQL → feature engineering → XGBoost → predictions
```

## Milestone 1 (this repo)
Pull one full NBA season → `player_game_stats` → rolling 5-game features.

## Setup

Requires a running local PostgreSQL (tested with `postgresql@15`) and Python 3.13.

```bash
# 1. create + activate a virtual environment
python3.13 -m venv .venv
source .venv/bin/activate

# 2. install dependencies
pip install -r requirements.txt

# 3. configure the database connection
cp .env.example .env         # edit DATABASE_URL if needed

# 4. create the courtvision database (once)
createdb courtvision         # or: psql -c 'CREATE DATABASE courtvision;'

# 5. create tables
python scripts/init_db.py

# 6. ingest one season (this hits NBA.com; takes ~10-30s)
python scripts/ingest_season.py 2023-24

# 7. build + verify rolling 5-game features
python scripts/build_features.py 2023-24
```

## Schema

- `teams` — team_id, abbreviation, name
- `players` — player_id, name
- `games` — game_id, game_date, season
- `player_game_stats` — one row per player per game (real box score)
- `player_game_features` — rolling `*_last5` features + `target_*` actuals

### Why "leakage-safe"?
Each `*_last5` feature is the mean of a player's **previous** up-to-5 games
(`shift(1)` before rolling), so a game's own result never leaks into its own
input features. The current game's actuals are stored as `target_*`.

## Project layout
```
src/courtvision/
  config.py     # env / .env loading (DATABASE_URL, NBA_SEASON)
  db.py         # SQLAlchemy engine + session
  models.py     # ORM schema
  ingest.py     # LeagueGameLog → player_game_stats
  features.py   # rolling 5-game features → player_game_features
scripts/
  init_db.py
  ingest_season.py
  build_features.py
```
