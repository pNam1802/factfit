# 0002. Store app data in SQLite through SQLModel

- **Status:** accepted
- **Date:** 2026-10-01

## Context

factfit is a single-user tool that runs on the user's laptop. It stores job descriptions,
match results, every CV version sent, applications with their status history, and one row
per LLM call. Volume is small (hundreds of jobs, thousands of LLM calls a year). The repo must
also be easy for someone else to clone and run, including in a Docker demo with no setup.

Requirements that matter more than scale:

- A CV linked to an application must never change afterwards.
- JSON results (parsed JD, match, tailored CV) evolve as the Pydantic schemas evolve.
- LangGraph needs somewhere to checkpoint paused runs.

Options considered:

| Option | For | Against |
| --- | --- | --- |
| JSON / YAML files | Zero setup, readable | No queries, no constraints, easy to corrupt with concurrent writes |
| **SQLite** | One file, no server, ships with Python; foreign keys, triggers, JSON columns; LangGraph has a SQLite checkpointer | Single writer; no multi-user access; foreign keys are off unless enabled per connection |
| PostgreSQL | Production standard, rich JSON, many users | A server to install and run for one user; heavier Docker demo |

For the ORM: **SQLModel** reuses Pydantic-style classes, matching the rest of the codebase;
plain SQLAlchemy would be more flexible but more verbose.

## Decision

Use SQLite (`data/factfit.db`, gitignored) through SQLModel. Store evolving structures as JSON
columns holding `model_dump(mode="json")` of the Pydantic schema. Enforce the important rules
in the database itself, not only in Python:

- `PRAGMA foreign_keys = ON` on every connection, so a CV used by an application cannot be deleted;
- a `BEFORE UPDATE` trigger on `cv_versions` that aborts every update, so a sent CV is immutable
  whichever code path (ORM, raw SQL, a future script) touches it.

## Consequences

- No database server to install; tests use a temporary file per test.
- Adding a field to a Pydantic schema needs no migration, because it lives inside a JSON column.
- Changing a real column does need a migration (see the follow-up below).
- One writer at a time: fine for one user, wrong for a hosted multi-user version. Moving to
  PostgreSQL later is mostly a connection-string change because access goes through SQLModel,
  but the trigger would need rewriting.

## Follow-up (2026-10-08): Alembic

Added before the first real application is stored, since from then on the database cannot
be thrown away and recreated.

- Migrations live in `factfit/db/migrations/`; `init_db` runs `upgrade head` itself, so the
  app, the CLI and the tests all get the same schema. Write a new one with
  `uv run alembic revision --autogenerate -m "..."`, then check it by hand.
- `0001` is the baseline: the tables as they stood, plus the `cv_versions` immutability
  trigger, which autogenerate cannot see. Databases made before migrations are detected (tables
  but no recorded version), brought up to the baseline and stamped, keeping their rows.
- `tests/test_migrations.py` fails when a model changes without a migration.
- SQLite cannot alter most columns in place, so migrations run in batch mode (copy the table).
