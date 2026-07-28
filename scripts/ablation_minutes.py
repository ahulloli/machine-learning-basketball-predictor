"""Walk-forward ablation: direct model vs. minutes x per-minute-rate vs. blend.

Decides whether the two-stage architecture (predict expected minutes and
expected per-minute production separately, then multiply) beats the current
direct XGBoost model. Uses rolling-origin validation so the verdict does not
hinge on a single lucky season:

    Fold 1: train 2021-22            -> validate 2022-23
    Fold 2: train 2021-22..2022-23   -> validate 2023-24
    Fold 3: train 2021-22..2023-24   -> validate 2024-25

The 2025-26 TEST season is never touched here. The blend weight is chosen on
these validation folds (not on test), exactly as recommended.

Features are built in memory, so this is a pure decision experiment and does not
modify the database feature table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from courtvision.features import build_features  # noqa: E402
from courtvision.train import (  # noqa: E402
    BOOL_FEATURES,
    CLUSTER_FEATURE,
    NUMERIC_FEATURES,
)

TARGETS = ["target_pts", "target_reb", "target_ast"]
BASELINE = {"target_pts": "pts_last5", "target_reb": "reb_last5", "target_ast": "ast_last5"}

FOLDS = [
    (["2021-22"], "2022-23"),
    (["2021-22", "2022-23"], "2023-24"),
    (["2021-22", "2022-23", "2023-24"], "2024-25"),
]

MIN_FEATURES = [
    "min_last3", "min_last5", "min_last10", "min_ewm5",
    "min_std5", "min_std10", "last_game_minutes",
    "rest_days", "long_break", "games_played_so_far", "home",
]

MIN_ELIGIBLE = 5.0          # only learn per-minute rates from meaningful minutes
WEIGHTS = np.round(np.arange(0.0, 1.01, 0.1), 2)


def _xgb(**overrides) -> XGBRegressor:
    cfg = dict(
        objective="reg:squarederror",
        subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, n_jobs=-1, random_state=42,
    )
    cfg.update(overrides)
    return XGBRegressor(**cfg)


def direct_matrix(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return (subset, X_direct, y_target, y_min) aligned to the direct model's rows."""
    sub = frame.dropna(subset=["pts_last5", target]).reset_index(drop=True)
    X = sub[NUMERIC_FEATURES + BOOL_FEATURES].copy()
    X["home"] = X["home"].astype(int)
    X = X.apply(pd.to_numeric, errors="coerce")
    clu = pd.get_dummies(sub[CLUSTER_FEATURE].fillna(-1).astype(int), prefix="cluster")
    X = pd.concat([X, clu.reset_index(drop=True)], axis=1)
    return sub, X, sub[target].astype(float), sub["target_min"].astype(float)


def minutes_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    f = frame.dropna(subset=["target_min", "min_last5"]).reset_index(drop=True)
    X = f[MIN_FEATURES].copy()
    X["home"] = X["home"].astype(int)
    X = X.apply(pd.to_numeric, errors="coerce")
    return X, f["target_min"].astype(float)


def run_fold(df: pd.DataFrame, train_seasons: list[str], val_season: str) -> dict:
    train = df[df["season"].isin(train_seasons)]
    val = df[df["season"] == val_season]

    # Minutes model is target-agnostic: fit once per fold, reuse for all stats.
    Xm_tr, ym_tr = minutes_matrix(train)
    minutes_model = _xgb(max_depth=4, learning_rate=0.03, n_estimators=800)
    minutes_model.fit(Xm_tr, ym_tr, verbose=False)
    min_cols = list(Xm_tr.columns)

    results: dict[str, dict] = {}
    for target in TARGETS:
        tr_sub, Xd_tr, y_tr, ymin_tr = direct_matrix(train, target)
        va_sub, Xd_va, y_va, _ = direct_matrix(val, target)
        Xd_va = Xd_va.reindex(columns=Xd_tr.columns, fill_value=0)

        # --- Direct model (untouched architecture) ---
        direct = _xgb(max_depth=3, learning_rate=0.03, n_estimators=600, min_child_weight=5)
        direct.fit(Xd_tr, y_tr, verbose=False)
        direct_pred = direct.predict(Xd_va)

        # --- Two-stage: expected minutes x expected per-minute rate ---
        elig = (ymin_tr >= MIN_ELIGIBLE) & ymin_tr.notna() & y_tr.notna()
        y_rate = (y_tr[elig] / ymin_tr[elig]).astype(float)
        rate = _xgb(max_depth=4, learning_rate=0.03, n_estimators=800, min_child_weight=5)
        rate.fit(Xd_tr[elig.values], y_rate, verbose=False)

        Xm_va = va_sub[MIN_FEATURES].copy()
        Xm_va["home"] = Xm_va["home"].astype(int)
        Xm_va = Xm_va.apply(pd.to_numeric, errors="coerce").reindex(columns=min_cols, fill_value=0)

        min_pred = np.clip(minutes_model.predict(Xm_va), 0.0, 48.0)
        rate_pred = np.maximum(0.0, rate.predict(Xd_va))
        two_stage_pred = min_pred * rate_pred

        # --- Naive baseline (last-5 average of the same stat) ---
        base = va_sub[BASELINE[target]].astype(float)
        base_mask = base.notna()

        yv = y_va.to_numpy()
        blended_mae = {
            w: mean_absolute_error(yv, w * direct_pred + (1 - w) * two_stage_pred)
            for w in WEIGHTS
        }

        results[target] = {
            "n_val": len(va_sub),
            "baseline": float(mean_absolute_error(y_va[base_mask], base[base_mask])),
            "direct": float(mean_absolute_error(yv, direct_pred)),
            "two_stage": float(mean_absolute_error(yv, two_stage_pred)),
            "blended": blended_mae,
        }
    return results


def main() -> None:
    print("Building features in memory (all seasons)...")
    df = build_features()
    df["game_date"] = pd.to_datetime(df["game_date"])

    fold_results = []
    for i, (train_seasons, val_season) in enumerate(FOLDS, 1):
        print(f"\nFold {i}: train {train_seasons} -> validate {val_season}")
        res = run_fold(df, train_seasons, val_season)
        for target in TARGETS:
            r = res[target]
            best_w = min(r["blended"], key=r["blended"].get)
            print(
                f"  {target:<11} n={r['n_val']:<6} "
                f"baseline={r['baseline']:.3f}  direct={r['direct']:.3f}  "
                f"two_stage={r['two_stage']:.3f}  "
                f"blend(w={best_w})={r['blended'][best_w]:.3f}"
            )
        fold_results.append(res)

    # Average across folds; pick the blend weight minimizing AVERAGE validation MAE.
    print("\n" + "=" * 68)
    print("WALK-FORWARD AVERAGE (3 folds) — lower MAE is better")
    print("=" * 68)
    header = f"{'Model':<20}{'Points':>10}{'Rebounds':>12}{'Assists':>10}"
    print(header)
    print("-" * len(header))

    def avg(target: str, key: str) -> float:
        return float(np.mean([fr[target][key] for fr in fold_results]))

    for label, key in [("Last-five baseline", "baseline"),
                       ("Current XGBoost", "direct"),
                       ("Minutes x rate", "two_stage")]:
        print(f"{label:<20}"
              + "".join(f"{avg(t, key):>{w}.3f}" for t, w in zip(TARGETS, (10, 12, 10))))

    # Blended: choose w per target by average validation MAE across folds.
    blend_row = {}
    best_ws = {}
    for target in TARGETS:
        avg_by_w = {w: float(np.mean([fr[target]["blended"][w] for fr in fold_results]))
                    for w in WEIGHTS}
        best_w = min(avg_by_w, key=avg_by_w.get)
        best_ws[target] = best_w
        blend_row[target] = avg_by_w[best_w]
    print(f"{'Blended (per-target)':<20}"
          + "".join(f"{blend_row[t]:>{w}.3f}" for t, w in zip(TARGETS, (10, 12, 10))))
    print("\nSelected blend weights (w * direct + (1-w) * two_stage), chosen on validation:")
    for target in TARGETS:
        print(f"  {target}: w={best_ws[target]}")
    print("=" * 68)


if __name__ == "__main__":
    main()
