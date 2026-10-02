"""A job description after parsing (PRD §7.2). Produced by the `parse_jd` node.

This model is sent to OpenAI as the output format, so it follows Structured Outputs
rules: every field is required (use null when unknown) and there are no defaults.
"""

from typing import Literal

from pydantic import model_validator

from factfit.schemas.base import Strict

Category = Literal["skill", "experience", "education", "language", "soft_skill", "other"]
Seniority = Literal["intern", "fresher", "junior", "mid", "senior", "unknown"]


class Requirement(Strict):
    id: str  # "r1", "r2", ... unique across must_have and nice_to_have
    text: str
    category: Category
    # Alternatives where any one is enough: "PyTorch, TensorFlow, or JAX" is ONE requirement
    # with any_of = ["PyTorch", "TensorFlow", "JAX"]. Empty for a plain requirement.
    any_of: list[str]
    # The level this applies to when a posting covers several ("Additional requirements
    # for Senior"), else None meaning every applicant.
    level: Seniority | None


class JobDescription(Strict):
    title: str
    company: str | None
    seniority: Seniority  # the lowest level the posting accepts
    language: Literal["vi", "en"]
    location: str | None
    must_have: list[Requirement]
    nice_to_have: list[Requirement]
    responsibilities: list[str]
    keywords: list[str]

    @model_validator(mode="after")
    def _requirement_ids_unique(self):
        ids = [r.id for r in self.all_requirements()]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"requirement ids must be unique, repeated: {duplicates}")
        return self

    def all_requirements(self) -> list[Requirement]:
        return [*self.must_have, *self.nice_to_have]
