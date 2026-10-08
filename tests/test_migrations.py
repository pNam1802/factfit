"""Migrations must build the same schema as the models, and keep data in old databases."""

import sqlalchemy as sa
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlmodel import Session, SQLModel, select

from factfit.db import Job, init_db, make_engine


def head_of(engine) -> str:
    with engine.connect() as conn:
        return conn.execute(sa.text("SELECT version_num FROM alembic_version")).scalar_one()


def test_migrations_match_the_models(tmp_path):
    # Fails when a model changes without a migration: run
    #   uv run alembic revision --autogenerate -m "..."
    engine = make_engine(f"sqlite:///{tmp_path / 'new.db'}")
    init_db(engine)
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), SQLModel.metadata)
    assert diff == []


def test_init_db_twice_is_a_no_op(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'new.db'}")
    init_db(engine)
    head = head_of(engine)
    init_db(engine)
    assert head_of(engine) == head


def test_a_database_from_before_migrations_keeps_its_data(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'old.db'}")
    SQLModel.metadata.create_all(engine)  # how init_db made databases until week 4
    with Session(engine) as s:
        s.add(Job(title="AI Engineer", raw_text="...", text_hash="h"))
        s.commit()

    init_db(engine)

    assert head_of(engine)
    with Session(engine) as s:
        assert s.exec(select(Job)).one().title == "AI Engineer"


def test_migrated_database_keeps_sent_cvs_immutable(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'new.db'}")
    init_db(engine)
    with engine.connect() as conn:
        triggers = conn.execute(
            sa.text("SELECT name FROM sqlite_master WHERE type = 'trigger'")
        ).scalars()
        assert "cv_versions_immutable" in set(triggers)


def test_an_empty_version_table_counts_as_no_version(tmp_path):
    # Left behind by an alembic command run against an old database before init_db saw it.
    engine = make_engine(f"sqlite:///{tmp_path / 'old.db'}")
    SQLModel.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
    init_db(engine)
    assert head_of(engine)
