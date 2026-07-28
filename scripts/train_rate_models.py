"""Train per-minute production rate models for the manual-minutes override.

The walk-forward ablation (scripts/ablation_minutes.py) showed the *learned*
two-stage model does not beat the direct model on average, so the direct model
remains the default. These rate models exist only to power the OPT-IN
``--expected-minutes`` lever in predict.py: when you supply an exogenous minutes
estimate (confirmed rotation, injury return, blowout risk) the projection becomes

    projection = expected_minutes x expected_per_minute_rate

That is exactly the case the learned minutes model can't capture, so the lever is
worth having even though the fully-learned two-stage lost the ablation.

Each rate model uses the SAME feature construction as the direct model, so
predict.py can score it with its existing live feature vector. Trained on
2021-22 .. 2024-25 (the 2025-26 test season is never touched).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from courtvision.features import build_features  # noqa: E402
from courtvision.train import (  # noqa: E402
    BOOL_FEATURES,
    CLUSTER_FEATURE,
    CONTEXT_SEASONS,
    MODEL_DIR,
    NUMERIC_FEATURES,
)

TARGETS = ["target_pts", "target_reb", "target_ast"]
TEST_SEASON = "2025-26"
MIN_ELIGIBLE = 5.0


def rate_matrix(frame: pd.DataFrame, target: str):
    """Direct-model feature matrix + per-minute rate target on eligible rows."""
    f = frame.dropna(subset=["pts_last5", target, "target_min"]).copy()
    f = f[f["target_min"] >= MIN_ELIGIBLE].reset_index(drop=True)

    X = f[NUMERIC_FEATURES + BOOL_FEATURES].copy()
    X["home"] = X["home"].astype(int)
    X = X.apply(pd.to_numeric, errors="coerce")
    clu = pd.get_dummies(f[CLUSTER_FEATURE].fillna(-1).astype(int), prefix="cluster")
    X = pd.concat([X, clu.reset_index(drop=True)], axis=1)

    y = (f[target].astype(float) / f["target_min"].astype(float))
    return X, y


def main() -> None:
    targets = [sys.argv[1]] if len(sys.argv) > 1 else TARGETS
    print("Building features in memory...")
    df = build_features()
    df = df[~df["season"].isin(CONTEXT_SEASONS)]
    df = df[df["season"] != TEST_SEASON].copy()

    MODEL_DIR.mkdir(exist_ok=True)
    for target in targets:
        X, y = rate_matrix(df, target)
        model = XGBRegressor(
            objective="reg:squarederror",
            max_depth=4, learning_rate=0.03, n_estimators=800,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, n_jobs=-1, random_state=42,
        )
        model.fit(X, y, verbose=False)

        tag = target.replace("target_", "")
        model_path = MODEL_DIR / f"xgb_{tag}_rate.json"
        meta_path = MODEL_DIR / f"xgb_{tag}_rate_meta.json"
        model.save_model(model_path)
        meta_path.write_text(json.dumps({
            "feature_names": list(X.columns),
            "target": f"{target}_per_min",
            "min_eligible": MIN_ELIGIBLE,
            "train_rows": int(len(y)),
            "mean_rate": round(float(y.mean()), 4),
        }, indent=2))
        print(f"[{tag}] rate model trained on {len(y):,} games  "
              f"mean rate={y.mean():.4f}/min  saved {model_path.name}")


if __name__ == "__main__":
    main()
