"""Connect to the SQLite database file and create tables."""

from pathlib import Path

from sqlalchemy import Engine, event
from sqlmodel import SQLModel, create_engine

from factfit.db import models  # noqa: F401  (registers the tables on SQLModel.metadata)

DEFAULT_URL = "sqlite:///data/factfit.db"


def make_engine(url: str = DEFAULT_URL, echo: bool = False) -> Engine:
    if url.startswith("sqlite:///") and not url.endswith(":memory:"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, echo=echo)

    # SQLite ignores foreign keys unless asked, so a CV linked to an application
    # could otherwise be deleted.
    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _record):
        dbapi_connection.execute("PRAGMA foreign_keys = ON")

    return engine


def init_db(engine: Engine) -> None:
    SQLModel.metadata.create_all(engine)
