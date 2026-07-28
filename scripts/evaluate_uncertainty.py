"""Evaluate CourtVision's calibrated intervals on the untouched 2025-26 season.

Loads the point model + target-specific residual calibrator produced by
scripts/calibrate_uncertainty.py and reports, ONCE, on 2025-26:

* point MAE,
* interval coverage vs. the requested 80% (should be close),
* average interval width (tighter is better at equal coverage),
* coverage broken down by expected-minutes bucket (reveals whether one global
  radius is too narrow for volatile bench players / too wide for starters),
* Brier score at representative over/under lines (probability quality).

Freeze the method before reading these numbers; do not iterate against 2025-26.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_absolute_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from courtvision.train import MODEL_DIR, load_features, prepare_xy  # noqa: E402
from courtvision.uncertainty import (  # noqa: E402
    load_calibrator,
    prediction_interval,
    probability_over,
)

TEST_SEASON = "2025-26"

# Representative over/under lines per target for Brier scoring.
BRIER_LINES = {
    "target_pts": (14.5, 19.5, 24.5),
    "target_reb": (4.5, 6.5, 8.5),
    "target_ast": (3.5, 5.5, 7.5),
}

# Expected-minutes buckets (proxied by min_last5), (label, low, high).
MINUTE_BUCKETS = [
    ("<15 min", 0.0, 15.0),
    ("15-25 min", 15.0, 25.0),
    ("25-35 min", 25.0, 35.0),
    ("35+ min", 35.0, float("inf")),
]

LABEL = {"target_pts": "POINTS", "target_reb": "REBOUNDS", "target_ast": "ASSISTS"}


def _load_model(target: str) -> tuple[XGBRegressor, list[str]]:
    tag = target.replace("target_", "")
    model_path = MODEL_DIR / f"xgb_{tag}.json"
    meta_path = MODEL_DIR / f"xgb_{tag}_meta.json"
    model = XGBRegressor()
    model.load_model(model_path)
    feature_names = json.loads(meta_path.read_text())["feature_names"]
    return model, feature_names


def evaluate_target(df: pd.DataFrame, target: str) -> None:
    test = df[df["season"] == TEST_SEASON].dropna(subset=["pts_last5", target]).copy()
    if test.empty:
        raise RuntimeError(f"No rows for test season {TEST_SEASON}")

    model, feature_names = _load_model(target)
    calibrator = load_calibrator(target)

    X_test, y_test = prepare_xy(test, target)
    X_test = X_test.reindex(columns=feature_names, fill_value=0)
    y = y_test.to_numpy()

    pred = model.predict(X_test)
    mae = float(mean_absolute_error(y, pred))

    lower = np.maximum(0.0, pred - calibrator.radius)
    upper = pred + calibrator.radius
    covered = (y >= lower) & (y <= upper)
    coverage = float(covered.mean())
    avg_width = float(np.mean(upper - lower))

    # Coverage by expected-minutes bucket (min_last5 is the pre-game proxy).
    # prepare_xy drops the same rows for X and y, so realign minutes to X_test.
    minutes = test.dropna(subset=["pts_last5", target])["min_last5"].to_numpy()

    print(f"\n{LABEL.get(target, target)}")
    print(f"Point MAE:                {mae:.3f}")
    print(f"Requested coverage:       {calibrator.coverage:.1%}")
    print(f"Actual coverage:          {coverage:.1%}")
    print(f"Average interval width:   {avg_width:.2f}")
    print(f"Conformal radius:         +/-{calibrator.radius:.3f}")

    print("  coverage by expected minutes:")
    for label, lo, hi in MINUTE_BUCKETS:
        mask = (minutes >= lo) & (minutes < hi)
        n = int(mask.sum())
        if n == 0:
            print(f"    {label:<10} n=0")
            continue
        bucket_cov = float(covered[mask].mean())
        bucket_width = float(np.mean((upper - lower)[mask]))
        print(f"    {label:<10} n={n:<6} coverage={bucket_cov:.1%}  width={bucket_width:.2f}")

    print("  probability quality (Brier, lower is better):")
    for line in BRIER_LINES.get(target, ()):
        actual_over = (y > line).astype(int)
        probs = np.array(
            [probability_over(prediction=float(p), line=line, calibrator=calibrator) for p in pred]
        )
        score = float(brier_score_loss(actual_over, probs))
        print(f"    over {line:<5} Brier={score:.4f}")


def main() -> None:
    targets = [sys.argv[1]] if len(sys.argv) > 1 else list(LABEL)
    df = load_features()
    print("=" * 60)
    print(f"CourtVision uncertainty evaluation — TEST season {TEST_SEASON}")
    print("=" * 60)
    for target in targets:
        evaluate_target(df, target)
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
