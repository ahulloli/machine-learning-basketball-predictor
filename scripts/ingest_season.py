"""Ingest one full NBA season of player box scores.

Usage:
    python scripts/ingest_season.py                 # uses NBA_SEASON from .env
    python scripts/ingest_season.py 2023-24
    python scripts/ingest_season.py 2023-24 "Playoffs"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from courtvision.config import DEFAULT_SEASON  # noqa: E402
from courtvision.ingest import ingest_season  # noqa: E402


def main() -> None:
    season = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SEASON
    season_type = sys.argv[2] if len(sys.argv) > 2 else "Regular Season"
    ingest_season(season, season_type)


if __name__ == "__main__":
    main()
