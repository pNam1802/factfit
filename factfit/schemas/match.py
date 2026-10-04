"""Result of matching a job description against the profile (PRD §7.3, F3).

The LLM decides each requirement's status and evidence (`MatchJudgement`); code checks the
evidence ids and computes the score (`MatchResult`). The LLM never chooses the score.
"""

from typing import Literal

from pydantic import Field, computed_field, model_validator

from factfit.schemas.base import Strict
from factfit.schemas.job import Seniority

# unverifiable: a CV cannot prove it either way (availability, location, "fluent Vietnamese").
# Such requirements are shown but neither scored nor counted as gaps.
Status = Literal["met", "partial", "missing", "unverifiable"]


class RequirementMatch(Strict):
    req_id: str
    status: Status
    evidence: list[str]  # ids of profile items: bullets, education or certification entries
    note: str  # one short reason


class MatchJudgement(Strict):
    """What the LLM returns. Sent as a Structured Outputs schema, so no defaults."""

    requirements: list[RequirementMatch]


class MatchResult(Strict):
    score: int = Field(ge=0, le=100)
    level: Seniority | None = None  # applicant level used to filter level-specific requirements
    requirements: list[RequirementMatch]
    excluded: list[str] = []  # requirement ids for another level, not judged

    @model_validator(mode="before")
    @classmethod
    def _drop_derived(cls, data):
        # `gaps` is derived; it appears in stored JSON but is not an input.
        if isinstance(data, dict):
            data = {k: v for k, v in data.items() if k != "gaps"}
        return data

    @model_validator(mode="after")
    def _evidence_matches_status(self):
        for r in self.requirements:
            if r.status in ("met", "partial") and not r.evidence:
                raise ValueError(f"{r.req_id} is '{r.status}' but cites no evidence")
            if r.status in ("missing", "unverifiable") and r.evidence:
                raise ValueError(f"{r.req_id} is '{r.status}' but cites evidence {r.evidence}")
        return self

    @computed_field
    @property
    def gaps(self) -> list[str]:
        """Requirement ids with no evidence. Derived, so it can never disagree with statuses."""
        return [r.req_id for r in self.requirements if r.status == "missing"]
