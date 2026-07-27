"""SQLAlchemy ORM schema for CourtVision.

The core fact table is ``player_game_stats``: one row per player per game,
sourced directly from real NBA box scores via nba_api.
"""
from __future__ import annotations

from datetime import date, datetime

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    team_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    abbreviation: Mapped[str] = mapped_column(String(8), index=True)
    name: Mapped[str | None] = mapped_column(String(64), nullable=True)


class Player(Base):
    __tablename__ = "players"

    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), index=True)


class Game(Base):
    __tablename__ = "games"

    game_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    game_date: Mapped[date] = mapped_column(Date, index=True)
    season: Mapped[str] = mapped_column(String(8), index=True)


class PlayerGameStat(Base):
    __tablename__ = "player_game_stats"
    __table_args__ = (
        UniqueConstraint("game_id", "player_id", name="uq_player_game"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    game_id: Mapped[str] = mapped_column(ForeignKey("games.game_id"), index=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.player_id"), index=True)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.team_id"), index=True)

    season: Mapped[str] = mapped_column(String(8), index=True)
    game_date: Mapped[date] = mapped_column(Date, index=True)
    opponent_abbr: Mapped[str | None] = mapped_column(String(8), nullable=True)
    home: Mapped[bool] = mapped_column(Boolean, default=False)
    wl: Mapped[str | None] = mapped_column(String(1), nullable=True)

    # Box score
    min: Mapped[float | None] = mapped_column(Float, nullable=True)
    fgm: Mapped[float | None] = mapped_column(Float, nullable=True)
    fga: Mapped[float | None] = mapped_column(Float, nullable=True)
    fg_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    fg3m: Mapped[float | None] = mapped_column(Float, nullable=True)
    fg3a: Mapped[float | None] = mapped_column(Float, nullable=True)
    fg3_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    ftm: Mapped[float | None] = mapped_column(Float, nullable=True)
    fta: Mapped[float | None] = mapped_column(Float, nullable=True)
    ft_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    oreb: Mapped[float | None] = mapped_column(Float, nullable=True)
    dreb: Mapped[float | None] = mapped_column(Float, nullable=True)
    reb: Mapped[float | None] = mapped_column(Float, nullable=True)
    ast: Mapped[float | None] = mapped_column(Float, nullable=True)
    stl: Mapped[float | None] = mapped_column(Float, nullable=True)
    blk: Mapped[float | None] = mapped_column(Float, nullable=True)
    tov: Mapped[float | None] = mapped_column(Float, nullable=True)
    pf: Mapped[float | None] = mapped_column(Float, nullable=True)
    pts: Mapped[float | None] = mapped_column(Float, nullable=True)
    plus_minus: Mapped[float | None] = mapped_column(Float, nullable=True)


class PlayerGameFeature(Base):
    """Leakage-safe rolling features + target for supervised learning.

    Every ``*_last5`` column is computed from the player's PREVIOUS games only
    (shifted by one), so it never includes the current game's outcome.
    The target columns (``target_*``) hold the current game's actual result.
    """

    __tablename__ = "player_game_features"
    __table_args__ = (
        UniqueConstraint("game_id", "player_id", name="uq_player_game_feature"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    game_id: Mapped[str] = mapped_column(String(16), index=True)
    player_id: Mapped[int] = mapped_column(Integer, index=True)
    player_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    game_date: Mapped[date] = mapped_column(Date, index=True)
    season: Mapped[str] = mapped_column(String(8), index=True)
    opponent_abbr: Mapped[str | None] = mapped_column(String(8), nullable=True)
    home: Mapped[bool] = mapped_column(Boolean, default=False)
    rest_days: Mapped[float | None] = mapped_column(Float, nullable=True)
    games_played_so_far: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Rolling last-5 features (from previous games only)
    pts_last5: Mapped[float | None] = mapped_column(Float, nullable=True)
    reb_last5: Mapped[float | None] = mapped_column(Float, nullable=True)
    ast_last5: Mapped[float | None] = mapped_column(Float, nullable=True)
    min_last5: Mapped[float | None] = mapped_column(Float, nullable=True)
    fg3m_last5: Mapped[float | None] = mapped_column(Float, nullable=True)
    fga_last5: Mapped[float | None] = mapped_column(Float, nullable=True)
    stl_last5: Mapped[float | None] = mapped_column(Float, nullable=True)
    blk_last5: Mapped[float | None] = mapped_column(Float, nullable=True)
    tov_last5: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Longer-window (last-10) + minutes/efficiency features
    pts_last10: Mapped[float | None] = mapped_column(Float, nullable=True)
    reb_last10: Mapped[float | None] = mapped_column(Float, nullable=True)
    ast_last10: Mapped[float | None] = mapped_column(Float, nullable=True)
    min_last10: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Per-minute efficiency (last 5) for each target stat
    pts_per_min_last5: Mapped[float | None] = mapped_column(Float, nullable=True)
    reb_per_min_last5: Mapped[float | None] = mapped_column(Float, nullable=True)
    ast_per_min_last5: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Opponent defensive strength (points the opponent allows, prior games only)
    opp_def_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    opp_def_rating_last10: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Player-vs-opponent history (prior meetings only), per target stat
    pts_vs_opp: Mapped[float | None] = mapped_column(Float, nullable=True)
    reb_vs_opp: Mapped[float | None] = mapped_column(Float, nullable=True)
    ast_vs_opp: Mapped[float | None] = mapped_column(Float, nullable=True)
    games_vs_opp: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Team-playstyle similarity: opponent's cluster + player's history vs
    # teams that play similarly (prior games only), per target stat
    opp_cluster: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pts_vs_opp_cluster: Mapped[float | None] = mapped_column(Float, nullable=True)
    reb_vs_opp_cluster: Mapped[float | None] = mapped_column(Float, nullable=True)
    ast_vs_opp_cluster: Mapped[float | None] = mapped_column(Float, nullable=True)
    games_vs_opp_cluster: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Supervised targets (current game actuals)
    target_pts: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_reb: Mapped[float | None] = mapped_column(Float, nullable=True)
    target_ast: Mapped[float | None] = mapped_column(Float, nullable=True)


class Prediction(Base):
    """Logged model projections, later reconciled against real results.

    ``predicted`` is stored at prediction time; ``actual`` and ``abs_error`` are
    filled in after the game finishes, powering model-monitoring metrics.
    """

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint(
            "player_id", "game_date", "opponent_abbr", "target",
            name="uq_prediction",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    player_id: Mapped[int] = mapped_column(Integer, index=True)
    player_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    game_date: Mapped[date] = mapped_column(Date, index=True)
    opponent_abbr: Mapped[str] = mapped_column(String(8))
    home: Mapped[bool] = mapped_column(Boolean, default=False)

    target: Mapped[str] = mapped_column(String(16), index=True)  # pts / reb / ast
    predicted: Mapped[float] = mapped_column(Float)
    baseline_last5: Mapped[float | None] = mapped_column(Float, nullable=True)

    actual: Mapped[float | None] = mapped_column(Float, nullable=True)
    abs_error: Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline_abs_error: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
