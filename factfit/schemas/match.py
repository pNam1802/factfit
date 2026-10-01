"""Result of matching a job description against the profile (PRD §7.3, F3).

The LLM decides each requirement's status and evidence; the score is computed
by code from those statuses, never chosen by the LLM.
"""

from typing import Literal

from pydantic import Field, computed_field, model_validator

from factfit.schemas.base import Strict

Status = Literal["met", "partial", "missing"]


class RequirementMatch(Strict):
    req_id: str
    status: Status
    evidence: list[str]  # bullet ids from the profile
    note: str


class MatchResult(Strict):
    score: int = Field(ge=0, le=100)
    requirements: list[RequirementMatch]

    @model_validator(mode="after")
    def _evidence_matches_status(self):
        for r in self.requirements:
            if r.status in ("met", "partial") and not r.evidence:
                raise ValueError(f"{r.req_id} is '{r.status}' but cites no evidence bullet")
            if r.status == "missing" and r.evidence:
                raise ValueError(f"{r.req_id} is 'missing' but cites evidence {r.evidence}")
        return self

    @computed_field
    @property
    def gaps(self) -> list[str]:
        """Requirement ids with no evidence. Derived, so it can never disagree with statuses."""
        return [r.req_id for r in self.requirements if r.status == "missing"]
