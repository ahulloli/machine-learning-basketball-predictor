"""Ingest one full NBA season of player box scores into PostgreSQL.

Strategy: a single ``LeagueGameLog`` call with ``player_or_team_abbreviation='P'``
returns EVERY player's box score for EVERY game in a season. That's far more
efficient than iterating ~450 players individually.

The MATCHUP column ("GSW vs. LAL" or "GSW @ LAL") encodes home/away and the
opponent abbreviation, which we parse out.
"""
from __future__ import annotations

import time

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .db import SessionLocal, engine
from .models import Base, Game, Player, PlayerGameStat, Team


def _parse_min(value) -> float | None:
    """MIN may arrive as a number or as 'MM:SS' / 'MM' string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    if ":" in s:
        mm, ss = s.split(":")[:2]
        try:
            return round(int(mm) + int(ss) / 60.0, 2)
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_matchup(matchup: str) -> tuple[str | None, bool]:
    """Return (opponent_abbr, is_home) from a MATCHUP string."""
    if not isinstance(matchup, str):
        return None, False
    if " vs. " in matchup:
        opp = matchup.split(" vs. ")[-1].strip()
        return opp, True
    if " @ " in matchup:
        opp = matchup.split(" @ ")[-1].strip()
        return opp, False
    return None, False


def fetch_season_player_logs(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """Fetch every player-game box score for a season as a DataFrame."""
    print(f"[ingest] Requesting LeagueGameLog for {season} ({season_type})...")
    for attempt in range(1, 4):
        try:
            resp = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation="P",
                timeout=60,
            )
            df = resp.get_data_frames()[0]
            print(f"[ingest] Received {len(df):,} player-game rows.")
            return df
        except Exception as exc:  # network / rate-limit resilience
            print(f"[ingest] attempt {attempt} failed: {exc}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"Failed to fetch LeagueGameLog for {season}")


def _normalize(df: pd.DataFrame, season: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.upper() for c in df.columns]

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"]).dt.date
    df["MIN"] = df["MIN"].map(_parse_min)

    parsed = df["MATCHUP"].map(_parse_matchup)
    df["OPPONENT_ABBR"] = [p[0] for p in parsed]
    df["HOME"] = [p[1] for p in parsed]
    df["SEASON"] = season
    return df


def ingest_season(season: str, season_type: str = "Regular Season") -> int:
    """Fetch a season and upsert into the database. Returns rows processed."""
    Base.metadata.create_all(engine)

    raw = fetch_season_player_logs(season, season_type)
    df = _normalize(raw, season)

    num_cols = [
        "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA",
        "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF",
        "PTS", "PLUS_MINUS",
    ]

    teams: dict[int, dict] = {}
    players: dict[int, dict] = {}
    games: dict[str, dict] = {}
    stat_rows: list[dict] = []

    for r in df.itertuples(index=False):
        row = r._asdict()
        team_id = int(row["TEAM_ID"])
        player_id = int(row["PLAYER_ID"])
        game_id = str(row["GAME_ID"])

        teams.setdefault(team_id, {
            "team_id": team_id,
            "abbreviation": row.get("TEAM_ABBREVIATION"),
            "name": row.get("TEAM_NAME"),
        })
        players.setdefault(player_id, {
            "player_id": player_id,
            "name": row.get("PLAYER_NAME"),
        })
        games.setdefault(game_id, {
            "game_id": game_id,
            "game_date": row["GAME_DATE"],
            "season": season,
        })

        stat = {
            "game_id": game_id,
            "player_id": player_id,
            "team_id": team_id,
            "season": season,
            "game_date": row["GAME_DATE"],
            "opponent_abbr": row["OPPONENT_ABBR"],
            "home": bool(row["HOME"]),
            "wl": row.get("WL"),
            "min": row.get("MIN"),
        }
        for col in num_cols:
            val = row.get(col)
            stat[col.lower()] = None if (val is None or pd.isna(val)) else float(val)
        stat_rows.append(stat)

    with SessionLocal() as session:
        _upsert(session, Team, list(teams.values()), ["team_id"])
        _upsert(session, Player, list(players.values()), ["player_id"])
        _upsert(session, Game, list(games.values()), ["game_id"])
        _upsert(session, PlayerGameStat, stat_rows, ["game_id", "player_id"])
        session.commit()

    print(
        f"[ingest] Upserted {len(teams)} teams, {len(players)} players, "
        f"{len(games)} games, {len(stat_rows):,} player-game stats."
    )
    return len(stat_rows)


def _upsert(session, model, rows: list[dict], conflict_cols: list[str], chunk: int = 1000) -> None:
    """Bulk INSERT ... ON CONFLICT DO NOTHING (idempotent re-ingest)."""
    if not rows:
        return
    table = model.__table__
    for i in range(0, len(rows), chunk):
        batch = rows[i : i + chunk]
        stmt = pg_insert(table).values(batch).on_conflict_do_nothing(
            index_elements=conflict_cols
        )
        session.execute(stmt)
