"""Daily prediction CLI.

Two modes:

  # 1) Single matchup (works fully offline against the DB):
  python scripts/predict_today.py --player "Stephen Curry" --opp BOS --home
  python scripts/predict_today.py --player "Stephen Curry" --opp BOS --date 2024-12-01

  # 2) Full slate for a date (fetches the schedule from NBA.com):
  python scripts/predict_today.py --slate --date 2024-12-01
"""
import argparse
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from courtvision.predict import predict_player, predict_slate  # noqa: E402


def _parse_date(s: str | None) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date() if s else date.today()


def main() -> None:
    ap = argparse.ArgumentParser(description="CourtVision daily predictions")
    ap.add_argument("--player", help="Player name or id")
    ap.add_argument("--opp", help="Opponent team abbreviation, e.g. BOS")
    ap.add_argument("--home", action="store_true", help="Player's team is home")
    ap.add_argument("--date", help="Target date YYYY-MM-DD (default: today)")
    ap.add_argument("--slate", action="store_true", help="Predict the whole schedule")
    ap.add_argument("--target", default="target_pts")
    ap.add_argument("--top", type=int, default=6, help="Players per team in slate mode")
    args = ap.parse_args()

    on_date = _parse_date(args.date)

    if args.slate:
        preds = predict_slate(on_date, target=args.target, top_n=args.top)
        if not preds:
            print(f"No games / predictions for {on_date}.")
            return
        preds.sort(key=lambda p: p.projection, reverse=True)
        print(f"\nCourtVision projections — {on_date}  ({args.target})")
        print("-" * 60)
        print(f"{'Player':22}{'OPP':>5}{'H/A':>5}{'proj':>8}{'last5':>8}")
        for p in preds:
            print(f"{p.player_name[:21]:22}{p.opponent_abbr:>5}"
                  f"{'H' if p.home else 'A':>5}{p.projection:>8}{p.baseline_last5:>8}")
        return

    if not args.player or not args.opp:
        ap.error("Provide --player and --opp (or use --slate).")

    p = predict_player(args.player, args.opp.upper(), args.home, on_date, args.target)
    ha = "HOME" if p.home else "AWAY"
    print("\n" + "=" * 50)
    print(f"  {p.player_name}  vs  {p.opponent_abbr}  ({ha})   {p.on_date}")
    print("-" * 50)
    print(f"  Model projection ({p.target.replace('target_','')}): {p.projection}")
    print(f"  Naive last-5 baseline:              {p.baseline_last5}")
    print(f"  Games of history used:              {p.games_of_history}")
    print("=" * 50)


if __name__ == "__main__":
    main()
