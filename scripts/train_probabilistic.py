"""Train CourtVision point models and calibrate uncertainty.

Temporal structure:
    2021-22 and 2022-23 -> hyperparameter training
    2023-24             -> hyperparameter validation
    2021-22 to 2023-24  -> final point-model fitting
    2024-25             -> uncertainty calibration
    2025-26             -> untouched final test

Calibration happens on a season the fitted model has never seen, so the
resulting intervals/probabilities are honest out-of-sample estimates. The
2025-26 season is scored exactly once for coverage/width/Brier and must not be
optimized against.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

from courtvision.train import (  # noqa: E402
    MODEL_DIR,
    PARAM_GRID,
    _xgb,
    load_features,
    prepare_xy,
)
from courtvision.uncertainty import (  # noqa: E402
    fit_calibrator,
    probability_over,
    save_calibrator,
)


SEARCH_TRAIN_SEASONS = ("2021-22", "2022-23")
TUNING_SEASON = "2023-24"
CALIBRATION_SEASON = "2024-25"
TEST_SEASON = "2025-26"

# Representative over/under lines per target, used only to report Brier score
# (probability quality) on the untouched test season.
BRIER_LINES = {
    "target_pts": (14.5, 19.5, 24.5),
    "target_reb": (4.5, 6.5, 8.5),
    "target_ast": (3.5, 5.5, 7.5),
}


def align(
    X: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    return X.reindex(columns=feature_names, fill_value=0)


def select_hyperparameters(
    search_train: pd.DataFrame,
    tuning: pd.DataFrame,
    target: str,
) -> dict:
    X_train, y_train = prepare_xy(search_train, target)
    X_tune, y_tune = prepare_xy(tuning, target)

    feature_names = list(X_train.columns)
    X_tune = align(X_tune, feature_names)

    best_params: dict | None = None
    best_mae = float("inf")

    for params in PARAM_GRID:
        model = _xgb(params)
        model.fit(X_train, y_train, verbose=False)

        predictions = model.predict(X_tune)
        mae = float(mean_absolute_error(y_tune, predictions))

        if mae < best_mae:
            best_mae = mae
            best_params = dict(params)

    if best_params is None:
        raise RuntimeError("No hyperparameter configuration was selected")

    print(f"Selected validation MAE: {best_mae:.3f}")
    print(f"Selected parameters: {best_params}")

    return best_params


def train_target(target: str, coverage: float = 0.80) -> None:
    df = load_features()

    search_train = df[df["season"].isin(SEARCH_TRAIN_SEASONS)].copy()
    tuning = df[df["season"] == TUNING_SEASON].copy()
    calibration = df[df["season"] == CALIBRATION_SEASON].copy()
    test = df[df["season"] == TEST_SEASON].copy()

    splits = {
        "search training": search_train,
        "tuning": tuning,
        "calibration": calibration,
        "test": test,
    }

    for name, frame in splits.items():
        if frame.empty:
            raise RuntimeError(
                f"No rows in {name} split. "
                "Make sure every required season was ingested "
                "and features were rebuilt."
            )

    # 1. Choose hyperparameters without looking at calibration or test.
    best_params = select_hyperparameters(
        search_train,
        tuning,
        target,
    )

    # 2. Refit point model using search-training + tuning seasons.
    point_fit = pd.concat(
        [search_train, tuning],
        ignore_index=True,
    )

    X_fit, y_fit = prepare_xy(point_fit, target)
    feature_names = list(X_fit.columns)

    model = _xgb(best_params)
    model.fit(X_fit, y_fit, verbose=False)

    # 3. Generate unseen predictions on 2024-25 for calibration.
    X_cal, y_cal = prepare_xy(calibration, target)
    X_cal = align(X_cal, feature_names)

    cal_predictions = model.predict(X_cal)

    calibrator = fit_calibrator(
        target=target,
        y_true=y_cal.to_numpy(),
        predictions=cal_predictions,
        coverage=coverage,
    )

    # 4. Save point model, metadata and calibration residuals.
    MODEL_DIR.mkdir(exist_ok=True)

    tag = target.replace("target_", "")
    model_path = MODEL_DIR / f"xgb_{tag}_prob.json"
    metadata_path = MODEL_DIR / f"xgb_{tag}_prob_meta.json"

    model.save_model(model_path)

    # 5. Evaluate once on the untouched 2025-26 season.
    X_test, y_test = prepare_xy(test, target)
    X_test = align(X_test, feature_names)

    test_predictions = model.predict(X_test)
    y_test_arr = y_test.to_numpy()

    lower = np.maximum(
        0.0,
        test_predictions - calibrator.radius,
    )
    upper = test_predictions + calibrator.radius

    interval_coverage = float(
        np.mean(
            (y_test_arr >= lower)
            & (y_test_arr <= upper)
        )
    )

    average_width = float(np.mean(upper - lower))
    test_mae = float(
        mean_absolute_error(y_test, test_predictions)
    )

    # Probability quality (Brier score) at representative lines.
    brier: dict[str, float] = {}
    for line in BRIER_LINES.get(target, ()):
        actual_over = (y_test_arr > line).astype(int)
        probs = np.array(
            [
                probability_over(float(p), line, calibrator)
                for p in test_predictions
            ]
        )
        brier[f"over_{line}"] = float(brier_score_loss(actual_over, probs))

    metadata_path.write_text(
        json.dumps(
            {
                "feature_names": feature_names,
                "target": target,
                "best_params": best_params,
                "point_fit_seasons": [
                    *SEARCH_TRAIN_SEASONS,
                    TUNING_SEASON,
                ],
                "calibration_season": CALIBRATION_SEASON,
                "test_season": TEST_SEASON,
                "interval_coverage": coverage,
                "conformal_radius": calibrator.radius,
                "test_mae": round(test_mae, 3),
                "test_interval_coverage": round(interval_coverage, 3),
                "test_average_width": round(average_width, 3),
                "test_brier": {k: round(v, 4) for k, v in brier.items()},
            },
            indent=2,
        )
    )

    calibration_path = save_calibrator(calibrator)

    print("\n" + "=" * 64)
    print(f"Target:                   {target}")
    print(f"Test season:              {TEST_SEASON}")
    print(f"Test rows:                {len(y_test):,}")
    print(f"Point-model MAE:          {test_mae:.3f}")
    print(f"Requested coverage:       {coverage:.1%}")
    print(f"Actual interval coverage: {interval_coverage:.1%}")
    print(f"Average interval width:   {average_width:.3f}")
    print(f"Conformal radius:         +/-{calibrator.radius:.3f}")
    for line, score in brier.items():
        print(f"Brier ({line}):{'':>11}{score:.4f}")
    print(f"Saved model:              {model_path}")
    print(f"Saved calibration:        {calibration_path}")
    print("=" * 64)


def main() -> None:
    targets = (
        [sys.argv[1]]
        if len(sys.argv) > 1
        else ["target_pts", "target_reb", "target_ast"]
    )

    for target in targets:
        train_target(target)


if __name__ == "__main__":
    main()
