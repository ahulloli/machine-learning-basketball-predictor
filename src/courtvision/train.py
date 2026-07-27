"""Train an XGBoost model to predict player points from rolling features.

Design principles that match the ingestion/feature pipeline:

* **Time-based split** — we train on earlier games and validate on later ones,
  never randomly. Randomly shuffling would let the model "see the future" and
  inflate scores; a chronological split mirrors real deployment.
* **Baseline comparison** — the naive predictor is simply ``pts_last5`` (predict
  the last-5 average). A useful model must beat this baseline.
* **Categorical opponent** — the opponent abbreviation is one-hot encoded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import text
from xgboost import XGBRegressor

from .config import PROJECT_ROOT
from .db import engine

MODEL_DIR = PROJECT_ROOT / "models"

NUMERIC_FEATURES = [
    "pts_last5", "reb_last5", "ast_last5", "min_last5", "fg3m_last5",
    "fga_last5", "stl_last5", "blk_last5", "tov_last5",
    "pts_last10", "reb_last10", "ast_last10", "min_last10",
    "pts_per_min_last5",
    "opp_def_rating", "opp_def_rating_last10",
    "pts_vs_opp", "games_vs_opp",
    "pts_vs_opp_cluster", "games_vs_opp_cluster",
    "rest_days", "games_played_so_far",
]
BOOL_FEATURES = ["home"]
CLUSTER_FEATURE = "opp_cluster"


@dataclass
class TrainResult:
    target: str
    n_train: int
    n_valid: int
    split_date: str
    model_mae: float
    model_rmse: float
    baseline_mae: float
    baseline_rmse: float
    mae_improvement_pct: float
    top_features: dict


@dataclass
class EvalResult:
    target: str
    test_season: str
    n_train: int
    n_valid: int
    n_test: int
    valid_split_date: str
    test_split_date: str
    # Metrics on the untouched TEST set
    test_model_mae: float
    test_model_rmse: float
    test_baseline_mae: float
    test_baseline_rmse: float
    test_mae_improvement_pct: float
    test_within_3: float
    # For reference: validation metrics
    valid_model_mae: float
    valid_baseline_mae: float


def load_features(season: str | None = None) -> pd.DataFrame:
    query = "SELECT * FROM player_game_features"
    params: dict = {}
    if season:
        query += " WHERE season = :season"
        params["season"] = season
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn, params=params)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def prepare_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    # Require a valid rolling history and target; drop the first game per player
    # (no prior-game features) and rows without an actual result.
    df = df.dropna(subset=["pts_last5", target]).copy()

    X = df[NUMERIC_FEATURES + BOOL_FEATURES].copy()
    X["home"] = X["home"].astype(int)
    X["rest_days"] = X["rest_days"].fillna(df["rest_days"].median())
    # Remaining NaNs (early-window features) -> let XGBoost handle as missing.
    X = X.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)

    # Low-cardinality opponent play-style cluster -> one-hot (unlike the noisy
    # 30-team encoding, only ~6 clusters, so this adds signal not noise).
    clu = pd.get_dummies(df[CLUSTER_FEATURE].fillna(-1).astype(int), prefix="cluster")
    X = pd.concat([X, clu.reset_index(drop=True)], axis=1)

    y = df[target].reset_index(drop=True)
    return X, y


def time_split(df: pd.DataFrame, valid_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    df_sorted = df.sort_values("game_date").reset_index(drop=True)
    cutoff_idx = int(len(df_sorted) * (1 - valid_frac))
    split_date = df_sorted.loc[cutoff_idx, "game_date"]
    train_df = df_sorted[df_sorted["game_date"] < split_date]
    valid_df = df_sorted[df_sorted["game_date"] >= split_date]
    return train_df, valid_df, split_date


# Seasons used only to seed team play-style clusters, never trained/evaluated on.
CONTEXT_SEASONS = ("2020-21",)


def three_way_split(
    df: pd.DataFrame,
    test_season: str,
    valid_frac_of_test_season: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Chronological TRAIN / VALIDATION / TEST split.

    * TRAIN      = every modeled season BEFORE ``test_season``.
    * VALIDATION = first ``valid_frac_of_test_season`` of ``test_season``
                   (used for early-stopping / model selection).
    * TEST       = the remainder of ``test_season`` (untouched until the end).

    Context seasons (used only to seed clusters) are excluded entirely.
    """
    df = df[~df["season"].isin(CONTEXT_SEASONS)].copy()

    train_df = df[df["season"] < test_season]
    season_df = df[df["season"] == test_season].sort_values("game_date").reset_index(drop=True)
    if season_df.empty:
        raise ValueError(f"No rows for test_season={test_season}")

    cut = int(len(season_df) * valid_frac_of_test_season)
    split_date = season_df.loc[cut, "game_date"]
    valid_df = season_df[season_df["game_date"] < split_date]
    test_df = season_df[season_df["game_date"] >= split_date]
    return train_df, valid_df, test_df, split_date


def train_target(
    df: pd.DataFrame,
    target: str = "target_pts",
    baseline_col: str = "pts_last5",
    valid_frac: float = 0.2,
) -> tuple[XGBRegressor, list[str], TrainResult]:
    df = df.dropna(subset=["pts_last5", target]).copy()
    train_df, valid_df, split_date = time_split(df, valid_frac)

    X_train, y_train = prepare_xy(train_df, target)
    X_valid, y_valid = prepare_xy(valid_df, target)

    # Align columns (validation may miss some opponent dummy columns).
    X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)
    feature_names = list(X_train.columns)

    model = XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    pred = model.predict(X_valid)
    model_mae = mean_absolute_error(y_valid, pred)
    model_rmse = float(np.sqrt(mean_squared_error(y_valid, pred)))

    # Naive baseline: predict the last-5 rolling average.
    base_pred = valid_df[baseline_col].reindex(valid_df.index)
    base_df = valid_df.dropna(subset=[baseline_col, target])
    baseline_mae = mean_absolute_error(base_df[target], base_df[baseline_col])
    baseline_rmse = float(np.sqrt(mean_squared_error(base_df[target], base_df[baseline_col])))

    importances = model.feature_importances_
    top = sorted(zip(feature_names, importances), key=lambda t: t[1], reverse=True)[:10]
    top_features = {name: round(float(v), 4) for name, v in top}

    result = TrainResult(
        target=target,
        n_train=len(X_train),
        n_valid=len(X_valid),
        split_date=str(split_date.date()),
        model_mae=round(model_mae, 3),
        model_rmse=round(model_rmse, 3),
        baseline_mae=round(baseline_mae, 3),
        baseline_rmse=round(baseline_rmse, 3),
        mae_improvement_pct=round((baseline_mae - model_mae) / baseline_mae * 100, 2),
        top_features=top_features,
    )
    return model, feature_names, result


def _xgb() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
        random_state=42,
    )


def _mae_rmse(y_true, y_pred) -> tuple[float, float]:
    return (
        float(mean_absolute_error(y_true, y_pred)),
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


def evaluate_target(
    df: pd.DataFrame,
    target: str,
    test_season: str,
    baseline_col: str,
) -> EvalResult:
    """Train / validate / test with a strictly chronological split.

    The TEST set (second half of ``test_season``) is never used for fitting or
    model selection — it is scored exactly once, at the end, to give a credible
    estimate of real-world performance.
    """
    df = df.dropna(subset=["pts_last5", target]).copy()
    train_df, valid_df, test_df, valid_split = three_way_split(df, test_season)
    test_split = test_df["game_date"].min()

    X_train, y_train = prepare_xy(train_df, target)
    X_valid, y_valid = prepare_xy(valid_df, target)
    X_test, y_test = prepare_xy(test_df, target)
    feature_names = list(X_train.columns)
    X_valid = X_valid.reindex(columns=feature_names, fill_value=0)
    X_test = X_test.reindex(columns=feature_names, fill_value=0)

    model = _xgb()
    # Validation set drives early stopping / model selection only.
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    def _metrics(frame, X, y):
        m_mae, m_rmse = _mae_rmse(y, model.predict(X))
        base = frame.dropna(subset=[baseline_col, target])
        b_mae, b_rmse = _mae_rmse(base[target], base[baseline_col])
        return m_mae, m_rmse, b_mae, b_rmse

    v_mmae, _, v_bmae, _ = _metrics(valid_df, X_valid, y_valid)
    t_mmae, t_mrmse, t_bmae, t_brmse = _metrics(test_df, X_test, y_test)
    within_3 = float((np.abs(y_test.values - model.predict(X_test)) <= 3).mean())

    return EvalResult(
        target=target,
        test_season=test_season,
        n_train=len(X_train),
        n_valid=len(X_valid),
        n_test=len(X_test),
        valid_split_date=str(valid_split.date()),
        test_split_date=str(test_split.date()),
        test_model_mae=round(t_mmae, 3),
        test_model_rmse=round(t_mrmse, 3),
        test_baseline_mae=round(t_bmae, 3),
        test_baseline_rmse=round(t_brmse, 3),
        test_mae_improvement_pct=round((t_bmae - t_mmae) / t_bmae * 100, 2),
        test_within_3=round(within_3, 3),
        valid_model_mae=round(v_mmae, 3),
        valid_baseline_mae=round(v_bmae, 3),
    )


def save_model(model: XGBRegressor, feature_names: list[str], result: TrainResult) -> Path:
    MODEL_DIR.mkdir(exist_ok=True)
    tag = result.target.replace("target_", "")
    model_path = MODEL_DIR / f"xgb_{tag}.json"
    meta_path = MODEL_DIR / f"xgb_{tag}_meta.json"
    model.save_model(model_path)
    meta_path.write_text(json.dumps(
        {"feature_names": feature_names, "metrics": asdict(result)}, indent=2
    ))
    return model_path
