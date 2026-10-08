"""Connect to the SQLite database file and bring its schema up to date."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from sqlalchemy import Connection, Engine, event, inspect, text
from sqlmodel import SQLModel, create_engine

from factfit.db import models  # noqa: F401  (registers the tables on SQLModel.metadata)

DEFAULT_URL = "sqlite:///data/factfit.db"
MIGRATIONS = Path(__file__).parent / "migrations"
BASELINE = "0001"  # the schema as it was when migrations were introduced


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
    """Bring the database to the latest schema by running the migrations in ./migrations.

    A database created before migrations existed (it has tables but no recorded version)
    is first brought up to the baseline schema, then marked as being at the baseline, so
    its data is kept and only later migrations run on it.
    """
    with engine.begin() as conn:
        config = Config()
        config.set_main_option("script_location", str(MIGRATIONS))
        config.attributes["connection"] = conn
        app_tables = set(inspect(conn).get_table_names()) - {"alembic_version"}
        # No recorded version: either a new file, or tables made before migrations. (An empty
        # alembic_version table, left by an aborted alembic command, counts as no version.)
        if app_tables and MigrationContext.configure(conn).get_current_revision() is None:
            # Valid while the models still match the baseline, which is the case for every
            # database made before migrations: they were all created from these models.
            SQLModel.metadata.create_all(conn)
            _add_missing_columns(conn)
            command.stamp(config, BASELINE)
        command.upgrade(config, "head")


def _add_missing_columns(conn: Connection) -> None:
    """Add nullable columns that an old database file lacks (pre-migration upgrade path).

    `create_all` only creates missing tables, so a column added to a model later is absent
    from a database created before it. Anything other than a new nullable column fails.
    """
    inspector = inspect(conn)
    existing_tables = set(inspector.get_table_names())
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
            col_type = column.type.compile(dialect=conn.dialect)
            conn.execute(text(f'ALTER TABLE {table.name} ADD COLUMN "{column.name}" {col_type}'))
