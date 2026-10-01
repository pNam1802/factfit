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
- Changing a real column does need a migration; none is set up yet (add Alembic when the first one
  is needed).
- One writer at a time: fine for one user, wrong for a hosted multi-user version. Moving to
  PostgreSQL later is mostly a connection-string change because access goes through SQLModel,
  but the trigger would need rewriting.
