"""Train XGBoost models on player_game_features and report vs baseline.

Usage:
    python scripts/train_model.py                     # points model, all seasons
    python scripts/train_model.py 2023-24             # single season
    python scripts/train_model.py 2023-24 target_reb  # different target
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from courtvision.train import load_features, train_target, save_model  # noqa: E402


def main() -> None:
    season = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "all" else None
    target = sys.argv[2] if len(sys.argv) > 2 else "target_pts"

    df = load_features(season)
    if df.empty:
        raise SystemExit("No feature rows. Run scripts/build_features.py first.")

    baseline_map = {
        "target_pts": "pts_last5",
        "target_reb": "reb_last5",
        "target_ast": "ast_last5",
    }
    baseline_col = baseline_map.get(target, "pts_last5")

    model, feature_names, result = train_target(df, target=target, baseline_col=baseline_col)
    path = save_model(model, feature_names, result)

    print("=" * 60)
    print(f"Target:            {result.target}")
    print(f"Season filter:     {season or 'ALL'}")
    print(f"Chronological split at {result.split_date}  "
          f"(train={result.n_train:,}, valid={result.n_valid:,})")
    print("-" * 60)
    print(f"{'':18}{'MAE':>10}{'RMSE':>10}")
    print(f"{'Naive last-5':18}{result.baseline_mae:>10}{result.baseline_rmse:>10}")
    print(f"{'XGBoost':18}{result.model_mae:>10}{result.model_rmse:>10}")
    print(f"MAE improvement vs baseline: {result.mae_improvement_pct}%")
    print("-" * 60)
    print("Top features by importance:")
    for name, imp in result.top_features.items():
        print(f"  {name:22} {imp}")
    print("-" * 60)
    print(f"Saved model -> {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
