"""Connect to the SQLite database file and create tables."""

from pathlib import Path

from sqlalchemy import Engine, event, inspect, text
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
    _add_missing_columns(engine)


def _add_missing_columns(engine: Engine) -> None:
    """Add new nullable columns to tables that already exist.

    `create_all` only creates missing tables, so a column added to a model later would be
    absent from an existing database file. This covers the simple, safe case (a new nullable
    column) until real migrations are set up; anything else fails loudly.
    """
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table in SQLModel.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            present = {c["name"] for c in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                if not column.nullable:
                    raise RuntimeError(
                        f"{table.name}.{column.name} is new and NOT NULL; it needs a migration"
                    )
                col_type = column.type.compile(dialect=engine.dialect)
                conn.execute(
                    text(f'ALTER TABLE {table.name} ADD COLUMN "{column.name}" {col_type}')
                )
