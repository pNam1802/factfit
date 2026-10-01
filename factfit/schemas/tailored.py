"""A CV tailored to one job (PRD §7.3, F4). Rendered to LaTeX after review.

Every line that ends up on the CV carries the ids of the profile items it came from,
so the grounding check can compare it with its sources.
"""

from typing import Literal

from pydantic import Field

from factfit.schemas.base import Strict

Grounding = Literal["pending", "pass", "fail"]  # "pending" = not checked yet
SectionType = Literal["experience", "project", "education", "publication", "award"]


class TailoredSummary(Strict):
    text: str
    source_ids: list[str] = Field(min_length=1)  # summary_variant ids


class TailoredBullet(Strict):
    text: str
    source_bullet_ids: list[str] = Field(min_length=1)
    grounding: Grounding = "pending"
    issues: list[str] = []


class TailoredSection(Strict):
    type: SectionType
    ref: str  # id of the experience / project / ... in the profile
    bullets: list[TailoredBullet]


class TailoredCV(Strict):
    summary: TailoredSummary | None = None
    sections: list[TailoredSection]
    skills: list[str]

    def all_bullets(self) -> list[TailoredBullet]:
        return [b for s in self.sections for b in s.bullets]

    def ready_to_export(self) -> bool:
        """F5: nothing can be exported while any bullet failed or was never checked."""
        return all(b.grounding == "pass" for b in self.all_bullets())
