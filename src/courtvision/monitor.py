"""Prediction logging + accuracy tracking (model monitoring).

Workflow:
  1. ``log_slate`` / ``log_prediction`` store projections BEFORE games happen.
  2. ``reconcile`` fills in the real result + error AFTER games finish, by
     matching each logged prediction to ``player_game_stats``.
  3. ``accuracy_report`` summarizes model vs naive-baseline error over the
     reconciled predictions — the model-monitoring dashboard's data source.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert

from .db import SessionLocal, engine
from .models import Base, Prediction
from .predict import Prediction as Proj
from .predict import predict_player, predict_slate
from .train import WITHIN_TOLERANCE

TARGET_COL = {"target_pts": "pts", "target_reb": "reb", "target_ast": "ast"}


def _ensure_table() -> None:
    Base.metadata.create_all(engine)


def _proj_to_row(p: Proj) -> dict:
    return {
        "player_id": p.player_id,
        "player_name": p.player_name,
        "game_date": pd.to_datetime(p.on_date).date(),
        "opponent_abbr": p.opponent_abbr,
        "home": p.home,
        "target": p.target.replace("target_", ""),
        "predicted": p.projection,
        "baseline_last5": None if pd.isna(p.baseline_last5) else p.baseline_last5,
    }


def log_predictions(projections: list[Proj]) -> int:
    _ensure_table()
    rows = [_proj_to_row(p) for p in projections]
    if not rows:
        return 0
    table = Prediction.__table__
    with SessionLocal() as session:
        for r in rows:
            stmt = pg_insert(table).values(**r).on_conflict_do_update(
                index_elements=["player_id", "game_date", "opponent_abbr", "target"],
                set_={"predicted": r["predicted"], "baseline_last5": r["baseline_last5"]},
            )
            session.execute(stmt)
        session.commit()
    return len(rows)


def log_slate(on_date: date, target: str = "target_pts", top_n: int = 6) -> int:
    return log_predictions(predict_slate(on_date, target=target, top_n=top_n))


def log_prediction(player: str, opp: str, home: bool, on_date: date, target: str = "target_pts") -> int:
    return log_predictions([predict_player(player, opp, home, on_date, target)])


def reconcile() -> int:
    """Fill actual/errors for logged predictions whose games now have results."""
    _ensure_table()
    with engine.connect() as conn:
        preds = pd.read_sql(
            text("SELECT * FROM predictions WHERE actual IS NULL"), conn
        )
        stats = pd.read_sql(
            text(
                "SELECT player_id, game_date, opponent_abbr, pts, reb, ast "
                "FROM player_game_stats"
            ),
            conn,
        )
    if preds.empty:
        return 0

    preds["game_date"] = pd.to_datetime(preds["game_date"]).dt.date
    stats["game_date"] = pd.to_datetime(stats["game_date"]).dt.date

    updated = 0
    with SessionLocal() as session:
        for r in preds.itertuples(index=False):
            row = r._asdict()
            col = row["target"]  # pts/reb/ast
            match = stats[
                (stats["player_id"] == row["player_id"])
                & (stats["game_date"] == row["game_date"])
                & (stats["opponent_abbr"] == row["opponent_abbr"])
            ]
            if match.empty or col not in match.columns:
                continue
            actual = float(match.iloc[0][col])
            abs_err = abs(actual - row["predicted"])
            base_err = (
                abs(actual - row["baseline_last5"])
                if row["baseline_last5"] is not None
                else None
            )
            session.execute(
                text(
                    "UPDATE predictions SET actual=:a, abs_error=:e, "
                    "baseline_abs_error=:b WHERE id=:id"
                ),
                {"a": actual, "e": abs_err, "b": base_err, "id": row["id"]},
            )
            updated += 1
        session.commit()
    return updated


@dataclass
class AccuracyReport:
    target: str
    n: int
    model_mae: float | None
    baseline_mae: float | None
    mae_improvement_pct: float | None
    within_tolerance: float | None   # share within the target-specific tolerance
    tolerance: float                 # the tolerance used (pts ±3, reb/ast ±2)


def accuracy_report(target: str | None = None) -> list[AccuracyReport]:
    with engine.connect() as conn:
        df = pd.read_sql(
            text("SELECT * FROM predictions WHERE actual IS NOT NULL"), conn
        )
    if df.empty:
        return []
    if target:
        df = df[df["target"] == target.replace("target_", "")]

    reports: list[AccuracyReport] = []
    for tgt, grp in df.groupby("target"):
        model_mae = float(grp["abs_error"].mean())
        base = grp["baseline_abs_error"].dropna()
        base_mae = float(base.mean()) if len(base) else None
        imp = (
            round((base_mae - model_mae) / base_mae * 100, 2)
            if base_mae
            else None
        )
        # Same target-specific tolerance used in offline evaluation, so the
        # production monitor and the held-out report stay directly comparable.
        tol = WITHIN_TOLERANCE.get(f"target_{tgt}", 3.0)
        within = float((grp["abs_error"] <= tol).mean())
        reports.append(
            AccuracyReport(
                target=tgt,
                n=len(grp),
                model_mae=round(model_mae, 3),
                baseline_mae=round(base_mae, 3) if base_mae else None,
                mae_improvement_pct=imp,
                within_tolerance=round(within, 3),
                tolerance=tol,
            )
        )
    return reports
