"""Request and response bodies of the HTTP API.

These are shaped for the UI (e.g. each requirement carries its text and the text of its
evidence), while the core schemas in factfit.schemas stay shaped for the agent.
"""

from typing import Literal

from pydantic import BaseModel, Field

from factfit.schemas.base import Strict
from factfit.schemas.job import Category, JobDescription, Seniority
from factfit.schemas.match import Status


class CreateJob(Strict):
    text: str = Field(min_length=1, description="The job description, pasted as text")
    company: str | None = None
    url: str | None = None


class JobOut(BaseModel):
    id: int
    title: str
    company: str | None
    cached: bool  # true when this JD was parsed before with the same prompt version
    jd: JobDescription


class MatchRequest(Strict):
    level: Seniority | None = Field(
        default=None,
        description="Applicant level for postings that hire several; default: lowest accepted",
    )


class Evidence(BaseModel):
    id: str
    where: str
    text: str


class RequirementOut(BaseModel):
    id: str
    text: str
    bucket: Literal["must", "nice"]
    category: Category
    any_of: list[str]
    status: Status
    note: str
    evidence: list[Evidence]


class MatchOut(BaseModel):
    job_id: int
    score: int
    level: Seniority | None
    profile_version: str
    cached: bool
    requirements: list[RequirementOut]
    excluded: list[str]  # requirement ids for another level


class ErrorOut(BaseModel):
    detail: str
    issues: list[str] = []


# --- tailoring runs ----------------------------------------------------------------------------

RunState = Literal["running", "waiting_review", "done", "error", "not_found"]


class TailorRequest(Strict):
    level: Seniority | None = None
    use_judge: bool = True


class RunOut(BaseModel):
    run_id: str
    status: RunState


class ReviewDecisionIn(Strict):
    source_id: str
    action: Literal["accept", "edit", "reject"]
    text: str | None = None


class ReviewRequest(Strict):
    decisions: list[ReviewDecisionIn] = []  # bullets left out are accepted


class DraftOut(BaseModel):
    source_id: str
    entry_id: str
    original: str
    text: str
    attempts: int
    issues: list[str]
    history: list[list]  # [rejected text, [problems]] per failed attempt
    passed: bool
    fallback: bool
    user_edited: bool = False
    user_rejected: bool = False


class RunStatusOut(BaseModel):
    run_id: str
    status: RunState
    drafts: list[DraftOut] = []
    review_errors: dict[str, list[str]] = {}  # source_id -> why an edit was refused
    cv: dict | None = None  # TailoredCV once done
    error: str | None = None
