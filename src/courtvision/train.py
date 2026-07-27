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
    "pts_per_min_last5", "reb_per_min_last5", "ast_per_min_last5",
    "opp_pts_allowed", "opp_pts_allowed_last10",
    "pts_vs_opp", "reb_vs_opp", "ast_vs_opp", "games_vs_opp",
    "pts_vs_opp_cluster", "reb_vs_opp_cluster", "ast_vs_opp_cluster",
    "games_vs_opp_cluster",
    "rest_days", "long_break", "games_played_so_far",
]
BOOL_FEATURES = ["home"]
CLUSTER_FEATURE = "opp_cluster"

# Target-specific "close enough" tolerance. Points are higher-scale/variance,
# so ±3 there is comparable to ±2 for rebounds/assists. MAE stays the headline.
WITHIN_TOLERANCE = {"target_pts": 3.0, "target_reb": 2.0, "target_ast": 2.0}


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
    within_threshold: float          # target-specific tolerance used below
    test_within_threshold: float     # share of test preds within that tolerance
    # For reference: validation metrics + selected hyperparameters
    valid_model_mae: float
    valid_baseline_mae: float
    best_params: dict


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
    # No imputation: NaNs (early-window / first-game rest_days) are passed
    # through and handled natively by XGBoost as "missing". Filling with a
    # split-wide median would leak the split's distribution into each row.
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

# Whole-season holdouts. VALIDATION drives every modeling decision (features +
# hyperparameters); TEST is a genuinely future season, touched exactly once.
VALID_SEASON = "2024-25"
TEST_SEASON = "2025-26"


def three_way_split(
    df: pd.DataFrame,
    valid_season: str = VALID_SEASON,
    test_season: str = TEST_SEASON,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Whole-season chronological TRAIN / VALIDATION / TEST split.

    * TRAIN      = every modeled season BEFORE ``valid_season``.
    * VALIDATION = the entire ``valid_season`` (feature + hyperparameter choices).
    * TEST       = the entire ``test_season`` — a future season held out until
                   the very end and scored only once.

    Context seasons (used only to seed clusters) are excluded entirely.
    """
    df = df[~df["season"].isin(CONTEXT_SEASONS)].copy()

    train_df = df[df["season"] < valid_season]
    valid_df = df[df["season"] == valid_season]
    test_df = df[df["season"] == test_season]
    if valid_df.empty:
        raise ValueError(f"No rows for valid_season={valid_season}")
    if test_df.empty:
        raise ValueError(f"No rows for test_season={test_season}")
    return train_df, valid_df, test_df


def train_target(
    df: pd.DataFrame,
    target: str = "target_pts",
    baseline_col: str = "pts_last5",
    valid_frac: float = 0.2,
) -> tuple[XGBRegressor, list[str], TrainResult]:
    # Context seasons seed clusters only; never train the deployable model on them.
    df = df[~df["season"].isin(CONTEXT_SEASONS)].dropna(subset=["pts_last5", target]).copy()
    train_df, valid_df, split_date = time_split(df, valid_frac)

    X_train, y_train = prepare_xy(train_df, target)
    X_valid, y_valid = prepare_xy(valid_df, target)

    # Align columns (validation may miss some opponent dummy columns).
    X_valid = X_valid.reindex(columns=X_train.columns, fill_value=0)
    feature_names = list(X_train.columns)

    # Select hyperparameters on the validation split (train only).
    best_params, best_mae = None, float("inf")
    for params in PARAM_GRID:
        m = _xgb(params)
        m.fit(X_train, y_train, verbose=False)
        v_mae = mean_absolute_error(y_valid, m.predict(X_valid))
        if v_mae < best_mae:
            best_params, best_mae = params, float(v_mae)

    # Validation MAE of the chosen configuration (reported below), measured
    # before the final refit so it reflects genuine held-out performance.
    model_mae = round(best_mae, 3)
    model_rmse = float(np.sqrt(mean_squared_error(
        y_valid, _xgb(best_params).fit(X_train, y_train).predict(X_valid)
    )))

    # Refit the FINAL deployable model on train+validation with the chosen
    # configuration, so it learns from all available history before deployment.
    X_full = pd.concat([X_train, X_valid.reindex(columns=X_train.columns, fill_value=0)],
                       ignore_index=True)
    y_full = pd.concat([y_train, y_valid], ignore_index=True)
    model = _xgb(best_params)
    model.fit(X_full, y_full, verbose=False)

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


_BASE_PARAMS = dict(
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    n_jobs=-1,
    random_state=42,
)

# Small grid searched on the VALIDATION set only (never the test set).
# n_estimators is held at 600 (paired with the learning-rate sweep to vary
# effective capacity) to keep the search tractable.
PARAM_GRID = [
    {"max_depth": d, "learning_rate": lr, "n_estimators": 600, "min_child_weight": mcw}
    for d in (3, 4, 5, 6)
    for lr in (0.03, 0.05, 0.1)
    for mcw in (3, 5, 10)
]


def _xgb(params: dict | None = None) -> XGBRegressor:
    cfg = dict(_BASE_PARAMS)
    cfg.update(params or {})
    cfg.setdefault("n_estimators", 400)
    cfg.setdefault("max_depth", 5)
    cfg.setdefault("learning_rate", 0.05)
    cfg.setdefault("min_child_weight", 5)
    return XGBRegressor(**cfg)


def _mae_rmse(y_true, y_pred) -> tuple[float, float]:
    return (
        float(mean_absolute_error(y_true, y_pred)),
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


def evaluate_target(
    df: pd.DataFrame,
    target: str,
    baseline_col: str,
    valid_season: str = VALID_SEASON,
    test_season: str = TEST_SEASON,
) -> EvalResult:
    """Whole-season train / validation / test evaluation.

    Model selection (hyperparameters) happens only on ``valid_season``. The
    ``test_season`` is a genuinely future season, scored exactly once at the
    end, so the reported metrics are an honest out-of-sample estimate.
    """
    df = df.dropna(subset=["pts_last5", target]).copy()
    train_df, valid_df, test_df = three_way_split(df, valid_season, test_season)

    X_train, y_train = prepare_xy(train_df, target)
    X_valid, y_valid = prepare_xy(valid_df, target)
    X_test, y_test = prepare_xy(test_df, target)
    feature_names = list(X_train.columns)
    X_valid = X_valid.reindex(columns=feature_names, fill_value=0)
    X_test = X_test.reindex(columns=feature_names, fill_value=0)

    # --- Model selection: search the grid, pick the lowest VALIDATION MAE. ---
    best_params, best_valid_mae, best_model = None, float("inf"), None
    for params in PARAM_GRID:
        m = _xgb(params)
        m.fit(X_train, y_train, verbose=False)
        v_mae = mean_absolute_error(y_valid, m.predict(X_valid))
        if v_mae < best_valid_mae:
            best_params, best_valid_mae, best_model = params, float(v_mae), m

    model = best_model  # selected purely on validation; test still untouched

    def _baseline(frame):
        base = frame.dropna(subset=[baseline_col, target])
        return _mae_rmse(base[target], base[baseline_col])

    v_bmae, _ = _baseline(valid_df)

    # --- Final, one-shot evaluation on the untouched TEST set. ---
    test_pred = model.predict(X_test)
    t_mmae, t_mrmse = _mae_rmse(y_test, test_pred)
    t_bmae, t_brmse = _baseline(test_df)
    tol = WITHIN_TOLERANCE.get(target, 3.0)
    within = float((np.abs(y_test.values - test_pred) <= tol).mean())

    return EvalResult(
        target=target,
        test_season=test_season,
        n_train=len(X_train),
        n_valid=len(X_valid),
        n_test=len(X_test),
        valid_split_date=valid_season,
        test_split_date=test_season,
        test_model_mae=round(t_mmae, 3),
        test_model_rmse=round(t_mrmse, 3),
        test_baseline_mae=round(t_bmae, 3),
        test_baseline_rmse=round(t_brmse, 3),
        test_mae_improvement_pct=round((t_bmae - t_mmae) / t_bmae * 100, 2),
        within_threshold=tol,
        test_within_threshold=round(within, 3),
        valid_model_mae=round(best_valid_mae, 3),
        valid_baseline_mae=round(v_bmae, 3),
        best_params=best_params,
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
