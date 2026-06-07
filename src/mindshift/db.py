from __future__ import annotations

from pathlib import Path
from typing import Iterator

from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, create_engine

from mindshift.config import sqlite_url_from_path


def make_engine(db_path: Path) -> Engine:
    return create_engine(
        sqlite_url_from_path(db_path),
        echo=False,
        connect_args={"check_same_thread": False},
    )


def create_db_and_tables(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)


def session_scope(engine: Engine) -> Iterator[Session]:
    with Session(engine) as session:
        yield session
