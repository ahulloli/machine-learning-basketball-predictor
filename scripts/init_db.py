"""Create all tables in the configured PostgreSQL database."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from courtvision.config import DATABASE_URL  # noqa: E402
from courtvision.db import engine  # noqa: E402
from courtvision.models import Base  # noqa: E402


def main() -> None:
    print(f"[init_db] Using {DATABASE_URL}")
    Base.metadata.create_all(engine)
    print("[init_db] Tables created:")
    for t in Base.metadata.sorted_tables:
        print(f"  - {t.name}")


if __name__ == "__main__":
    main()
