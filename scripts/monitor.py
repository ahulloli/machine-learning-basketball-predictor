"""Prediction logging + accuracy tracking CLI.

  # Log a whole slate's projections (before games):
  python scripts/monitor.py log-slate --date 2024-12-01 --target target_pts

  # Log a single matchup:
  python scripts/monitor.py log --player "Stephen Curry" --opp BOS --date 2024-12-01

  # After games finish, fill in actual results + errors:
  python scripts/monitor.py reconcile

  # Show the model-monitoring accuracy report:
  python scripts/monitor.py report
"""
import argparse
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from courtvision.monitor import (  # noqa: E402
    accuracy_report, log_prediction, log_slate, reconcile,
)


def _date(s: str | None) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date() if s else date.today()


def main() -> None:
    ap = argparse.ArgumentParser(description="CourtVision monitoring")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("log-slate")
    s.add_argument("--date")
    s.add_argument("--target", default="target_pts")
    s.add_argument("--top", type=int, default=6)

    l = sub.add_parser("log")
    l.add_argument("--player", required=True)
    l.add_argument("--opp", required=True)
    l.add_argument("--home", action="store_true")
    l.add_argument("--date")
    l.add_argument("--target", default="target_pts")

    sub.add_parser("reconcile")
    r = sub.add_parser("report")
    r.add_argument("--target")

    args = ap.parse_args()

    if args.cmd == "log-slate":
        n = log_slate(_date(args.date), target=args.target, top_n=args.top)
        print(f"Logged {n} predictions for {_date(args.date)} ({args.target}).")
    elif args.cmd == "log":
        n = log_prediction(args.player, args.opp.upper(), args.home, _date(args.date), args.target)
        print(f"Logged {n} prediction(s).")
    elif args.cmd == "reconcile":
        n = reconcile()
        print(f"Reconciled {n} predictions with actual results.")
    elif args.cmd == "report":
        reports = accuracy_report(args.target if hasattr(args, "target") else None)
        if not reports:
            print("No reconciled predictions yet. Run 'reconcile' after games finish.")
            return
        print("\nCourtVision model-monitoring report")
        print("-" * 64)
        print(f"{'target':8}{'n':>7}{'model_MAE':>12}{'base_MAE':>11}{'improve%':>11}{'within3':>9}")
        for r in reports:
            imp = "n/a" if r.mae_improvement_pct is None else f"{r.mae_improvement_pct}"
            base = "n/a" if r.baseline_mae is None else f"{r.baseline_mae}"
            print(f"{r.target:8}{r.n:>7}{r.model_mae:>12}{base:>11}{imp:>11}{r.within_3:>9}")
        print("-" * 64)


if __name__ == "__main__":
    main()
