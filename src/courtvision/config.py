"""Configuration loaded from environment / .env file."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load .env from project root if present.
load_dotenv(PROJECT_ROOT / ".env")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://rishahulloli@localhost:5432/courtvision",
)

DEFAULT_SEASON = os.getenv("NBA_SEASON", "2023-24")
