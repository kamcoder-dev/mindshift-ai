from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    db_path: Path = Path("mindshift.db")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def sqlite_url_from_path(path: Path) -> str:
    """Return a SQLAlchemy SQLite URL for a local database path."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path}"
