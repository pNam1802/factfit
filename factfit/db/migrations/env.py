"""Alembic environment: compares migrations against the SQLModel tables in factfit.db.models.

`init_db` passes its own connection in `config.attributes["connection"]`; the alembic CLI
(used to write new migrations) connects with the url in alembic.ini instead.
"""

from alembic import context
from sqlalchemy import create_engine
from sqlmodel import SQLModel

from factfit.db import models  # noqa: F401  (registers the tables on SQLModel.metadata)

config = context.config
target_metadata = SQLModel.metadata


def _run(connection) -> None:
    # SQLite cannot ALTER most things in place; batch mode copies the table instead.
    context.configure(connection=connection, target_metadata=target_metadata, render_as_batch=True)
    with context.begin_transaction():
        context.run_migrations()


connection = config.attributes.get("connection")
if connection is not None:
    _run(connection)
else:
    with create_engine(config.get_main_option("sqlalchemy.url")).connect() as conn:
        _run(conn)
