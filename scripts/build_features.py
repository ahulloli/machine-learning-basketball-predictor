"""Build rolling 5-game features and verify with a known player.

Usage:
    python scripts/build_features.py               # all seasons in DB
    python scripts/build_features.py 2023-24       # single season
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from courtvision.features import build_features, persist_features  # noqa: E402

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def main() -> None:
    season = sys.argv[1] if len(sys.argv) > 1 else None

    features = build_features(season)
    n = persist_features(features)
    print(f"[features] Built and stored {n:,} feature rows.")

    # Verification: show Stephen Curry's chronological rolling features.
    # Exact match — "Curry" alone would also catch Seth Curry.
    curry = features[features["player_name"] == "Stephen Curry"]
    if not curry.empty:
        cols = [
            "game_date", "opponent_abbr", "home", "rest_days",
            "pts_last5", "reb_last5", "ast_last5", "min_last5",
            "target_pts", "target_reb", "target_ast",
        ]
        print("\n[verify] Stephen Curry — first 12 games (rolling features vs actuals):")
        print(curry.sort_values("game_date")[cols].head(12).to_string(index=False))
    else:
        # Fallback: show the most active player.
        top = features["player_name"].value_counts().index[0]
        sample = features[features["player_name"] == top].sort_values("game_date")
        cols = [
            "player_name", "game_date", "opponent_abbr", "home", "rest_days",
            "pts_last5", "target_pts",
        ]
        print(f"\n[verify] Sample player '{top}':")
        print(sample[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
