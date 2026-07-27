"""Credible model evaluation with a strict TRAIN / VALIDATION / TEST split.

The TEST set (second half of the test season) is held out and scored exactly
once, giving an honest estimate of real-world performance. Context seasons
(e.g. 2020-21) are used only to seed team clusters and are never modeled on.

Usage:
    python scripts/evaluate.py                      # test on latest season, all targets
    python scripts/evaluate.py 2024-25              # choose the test season
    python scripts/evaluate.py 2024-25 target_pts   # single target
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from courtvision.train import evaluate_target, load_features  # noqa: E402

BASELINE = {"target_pts": "pts_last5", "target_reb": "reb_last5", "target_ast": "ast_last5"}


def main() -> None:
    test_season = sys.argv[1] if len(sys.argv) > 1 else "2024-25"
    targets = [sys.argv[2]] if len(sys.argv) > 2 else list(BASELINE)

    df = load_features()
    if df.empty:
        raise SystemExit("No features. Run scripts/build_features.py first.")

    results = [evaluate_target(df, t, test_season, BASELINE[t]) for t in targets]

    r0 = results[0]
    print("\n" + "=" * 74)
    print(f"  CourtVision held-out evaluation — TEST season {test_season}")
    print(f"  train={r0.n_train:,}  valid={r0.n_valid:,}  test={r0.n_test:,}")
    print(f"  validation starts {r0.valid_split_date}   TEST starts {r0.test_split_date}")
    print("=" * 74)
    print(f"{'target':8}{'test_MAE':>11}{'base_MAE':>11}{'improve%':>11}"
          f"{'test_RMSE':>12}{'within':>12}")
    print("-" * 74)
    for r in results:
        w = f"{r.test_within_threshold:.3f}(±{r.within_threshold:g})"
        print(f"{r.target.replace('target_',''):8}{r.test_model_mae:>11}"
              f"{r.test_baseline_mae:>11}{r.test_mae_improvement_pct:>11}"
              f"{r.test_model_rmse:>12}{w:>12}")
    print("-" * 74)
    print("Hyperparameters selected on VALIDATION (test never used for selection):")
    for r in results:
        print(f"  {r.target.replace('target_',''):5}  valid_model={r.valid_model_mae}  "
              f"valid_base={r.valid_baseline_mae}  best={r.best_params}")
    print("=" * 74)


if __name__ == "__main__":
    main()
