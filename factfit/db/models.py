"""SQLite tables (PRD §7.4). Each class is one table; each attribute is one column.

JSON columns hold Pydantic models dumped with `model_dump(mode="json")`, e.g. a
`JobDescription` in `Job.parsed_json`, so the database never needs a schema change
when those models gain a field.

Only MVP tables live here; `interview_preps` and `email_alerts` come with F16 / F12.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


def _now() -> datetime:
    return datetime.now(UTC)


class ApplicationStatus(StrEnum):
    saved = "saved"
    applied = "applied"
    screening = "screening"
    interview = "interview"
    offer = "offer"
    rejected = "rejected"
    ghosted = "ghosted"


class Company(SQLModel, table=True):
    __tablename__ = "companies"

    id: int | None = Field(default=None, primary_key=True)
    name: str = Field(index=True, unique=True)
    website: str | None = None
    notes: str = ""
    research_json: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))  # P2, F8
    updated_at: datetime = Field(default_factory=_now)


class Job(SQLModel, table=True):
    __tablename__ = "jobs"

    id: int | None = Field(default=None, primary_key=True)
    company_id: int | None = Field(default=None, foreign_key="companies.id", index=True)
    title: str
    url: str | None = None
    source: str = "manual"  # where the JD came from: manual paste, a job site, ...
    raw_text: str
    parsed_json: dict[str, Any] | None = Field(default=None, sa_column=Column(JSON))
    # Prompt that produced parsed_json, e.g. "parse_jd/v1". A new prompt version re-parses.
    parse_prompt_version: str | None = None
    # Hash of the normalised JD text: same JD pasted twice -> reuse the cached parse.
    text_hash: str = Field(index=True)
    created_at: datetime = Field(default_factory=_now)


class Match(SQLModel, table=True):
    __tablename__ = "matches"

    id: int | None = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="jobs.id", index=True)
    profile_version: str  # hash of profile.yaml at match time
    prompt_version: str | None = None  # e.g. "match/v1"
    score: int
    result_json: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    created_at: datetime = Field(default_factory=_now)


class CVVersion(SQLModel, table=True):
    """A rendered CV. Rows are immutable: a database trigger rejects any UPDATE.

    To change a CV, create a new version. This guarantees the CV linked to an
    application is exactly what was sent. The trigger is created by the baseline migration
    (factfit/db/migrations/versions/0001_baseline_schema.py).
    """

    __tablename__ = "cv_versions"

    id: int | None = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="jobs.id", index=True)
    profile_version: str
    tailored_json: dict[str, Any] = Field(sa_column=Column(JSON, nullable=False))
    tex: str
    pdf_path: str | None = None
    prompt_version: str  # e.g. "rewrite/v3"
    model: str
    created_at: datetime = Field(default_factory=_now)


class Application(SQLModel, table=True):
    __tablename__ = "applications"

    id: int | None = Field(default=None, primary_key=True)
    job_id: int = Field(foreign_key="jobs.id", index=True)
    # Empty when the CV was made by hand (cv_source = "manual").
    cv_version_id: int | None = Field(default=None, foreign_key="cv_versions.id")
    cv_source: str = "generated"  # "generated" | "manual"
    status: ApplicationStatus = ApplicationStatus.saved
    applied_at: datetime | None = None
    channel: str | None = None  # company site, LinkedIn, email, referral, ...
    notes: str = ""


class StatusEvent(SQLModel, table=True):
    """One row per status change, so response times can be measured later."""

    __tablename__ = "status_events"

    id: int | None = Field(default=None, primary_key=True)
    application_id: int = Field(foreign_key="applications.id", index=True)
    from_status: ApplicationStatus | None
    to_status: ApplicationStatus
    at: datetime = Field(default_factory=_now)
    source: str = "manual"  # "manual" | "gmail"


class LLMCall(SQLModel, table=True):
    """One row per LLM request: what it cost and how long it took (observability)."""

    __tablename__ = "llm_calls"

    id: int | None = Field(default=None, primary_key=True)
    run_id: str | None = Field(default=None, index=True)  # groups calls of one tailoring run
    node: str  # "parse_jd", "match", "rewrite", "judge", ...
    model: str
    prompt_version: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float | None = None  # None when the model has no price in config/llm.yaml
    latency_ms: int = 0
    ok: bool = True
    error: str | None = None
    created_at: datetime = Field(default_factory=_now)
