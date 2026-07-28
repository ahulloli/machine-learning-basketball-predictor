"""Fit CourtVision point models and calibrate uncertainty residuals.

Temporal structure (probabilities need calibration data the point model has
never seen):

    2020-21                     -> team-clustering context (never modeled)
    2021-22 .. 2023-24          -> model training
    2024-25 FIRST half          -> hyperparameter validation
    2021-22 .. 2024-25 1st half -> final point-model fitting
    2024-25 SECOND half         -> uncertainty calibration (unseen residuals)
    2025-26                     -> final probability evaluation (evaluate_uncertainty.py)

For each target this saves the deployable point model (``xgb_<tag>.json`` +
``xgb_<tag>_meta.json``) and a target-specific residual calibrator
(``calibration_<tag>.npz``). Points / rebounds / assists get their OWN residuals
because their error distributions differ substantially.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from courtvision.train import (  # noqa: E402
    CONTEXT_SEASONS,
    MODEL_DIR,
    PARAM_GRID,
    _xgb,
    load_features,
    prepare_xy,
)
from courtvision.uncertainty import fit_calibrator, save_calibrator  # noqa: E402

TRAIN_SEASONS = ("2021-22", "2022-23", "2023-24")
CALIB_SEASON = "2024-25"          # split chronologically into val / calibration
TEST_SEASON = "2025-26"           # untouched here; scored in evaluate_uncertainty.py
COVERAGE = 0.80

BASELINE_COL = {
    "target_pts": "pts_last5",
    "target_reb": "reb_last5",
    "target_ast": "ast_last5",
}


def split_calib_season(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically halve the 2024-25 season: (first_half, second_half)."""
    season = df[df["season"] == CALIB_SEASON].sort_values("game_date")
    if season.empty:
        raise RuntimeError(f"No rows for calibration season {CALIB_SEASON}")
    cutoff = season["game_date"].quantile(0.5)
    first_half = season[season["game_date"] < cutoff]
    second_half = season[season["game_date"] >= cutoff]
    return first_half, second_half


def align(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    return X.reindex(columns=feature_names, fill_value=0)


def calibrate_target(df: pd.DataFrame, target: str) -> None:
    df = df[~df["season"].isin(CONTEXT_SEASONS)].dropna(subset=["pts_last5", target]).copy()

    train_core = df[df["season"].isin(TRAIN_SEASONS)]
    val_half, calib_half = split_calib_season(df)

    if train_core.empty:
        raise RuntimeError("No training rows for seasons " + ", ".join(TRAIN_SEASONS))

    # 1. Select hyperparameters on the 2024-25 FIRST half (validation).
    X_train, y_train = prepare_xy(train_core, target)
    X_val, y_val = prepare_xy(val_half, target)
    feature_names = list(X_train.columns)
    X_val = align(X_val, feature_names)

    best_params, best_mae = None, float("inf")
    for params in PARAM_GRID:
        m = _xgb(params)
        m.fit(X_train, y_train, verbose=False)
        mae = float(mean_absolute_error(y_val, m.predict(X_val)))
        if mae < best_mae:
            best_params, best_mae = dict(params), mae

    print(f"[{target}] validation MAE={best_mae:.3f}  params={best_params}")

    # 2. Refit the point model on train + first-half 2024-25.
    point_fit = pd.concat([train_core, val_half], ignore_index=True)
    X_fit, y_fit = prepare_xy(point_fit, target)
    feature_names = list(X_fit.columns)

    model = _xgb(best_params)
    model.fit(X_fit, y_fit, verbose=False)

    # 3. Predict the UNSEEN 2024-25 second half and calibrate on those errors.
    X_calib, y_calib = prepare_xy(calib_half, target)
    X_calib = align(X_calib, feature_names)
    calib_pred = model.predict(X_calib)

    calibrator = fit_calibrator(
        target=target,
        y_true=y_calib.to_numpy(),
        y_pred=calib_pred,
        coverage=COVERAGE,
    )

    # 4. Persist the point model + metadata + target-specific residuals.
    MODEL_DIR.mkdir(exist_ok=True)
    tag = target.replace("target_", "")
    model_path = MODEL_DIR / f"xgb_{tag}.json"
    meta_path = MODEL_DIR / f"xgb_{tag}_meta.json"

    model.save_model(model_path)
    meta_path.write_text(json.dumps({
        "feature_names": feature_names,
        "target": target,
        "best_params": best_params,
        "train_seasons": list(TRAIN_SEASONS),
        "validation": f"{CALIB_SEASON} first half",
        "calibration": f"{CALIB_SEASON} second half",
        "test_season": TEST_SEASON,
        "interval_coverage": COVERAGE,
        "conformal_radius": calibrator.radius,
        "n_calibration": int(len(y_calib)),
    }, indent=2))

    calib_path = save_calibrator(calibrator)

    print(
        f"[{target}] calibrated on {len(y_calib):,} unseen games  "
        f"radius=+/-{calibrator.radius:.3f}\n"
        f"          saved {model_path.name}, {meta_path.name}, {calib_path.name}"
    )


def main() -> None:
    targets = [sys.argv[1]] if len(sys.argv) > 1 else list(BASELINE_COL)
    df = load_features()
    for target in targets:
        calibrate_target(df, target)


if __name__ == "__main__":
    main()
