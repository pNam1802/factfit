"""Shape of the Master Profile (`data/profile.yaml`), the single source of truth.

These models only check types and required fields. Rules that look across the
whole file (unique ids, skill evidence, numbers backed by metrics) live in
`factfit.profile.rules`, so every problem can be reported at once.
"""

from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, StringConstraints

from factfit.schemas.base import Strict

# "2026-03", or just "2026" when the month is not known (better than inventing one).
# YAML reads a bare `2026` as a number, so turn numbers into text before checking.
YearMonth = Annotated[
    str,
    BeforeValidator(lambda v: str(v) if isinstance(v, int) else v),
    StringConstraints(pattern=r"^\d{4}(-(0[1-9]|1[0-2]))?$"),
]
Id = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]
NonEmpty = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Metric(Strict):
    name: NonEmpty
    value: float
    unit: str | None = None
    verified: bool = Field(description="True only after you have checked the number yourself.")


class Bullet(Strict):
    id: Id
    text: NonEmpty
    text_vi: str | None = None  # optional Vietnamese version, for the P2 Vietnamese CV
    tags: list[str] = []
    skills: list[str] = []  # technologies used in this bullet
    metrics: list[Metric] = []


class SummaryVariant(Strict):
    id: Id
    tags: list[str] = []
    text: NonEmpty
    text_vi: str | None = None
    metrics: list[Metric] = []


class Basics(Strict):
    name: NonEmpty
    headline: str | None = None  # line under the name, e.g. "AI Engineer Intern"
    email: NonEmpty
    phone: str | None = None
    location: str | None = None
    github: str | None = None
    linkedin: str | None = None
    website: str | None = None


class Experience(Strict):
    id: Id
    org: NonEmpty
    role: NonEmpty
    location: str | None = None
    start: YearMonth
    end: YearMonth | Literal["present"]
    bullets: list[Bullet] = Field(min_length=1)


class Project(Strict):
    id: Id
    name: NonEmpty
    role: str | None = None
    url: str | None = None
    # Shown after the project name on the CV ("| Python, LangGraph"). Empty: use the
    # skills of the project's bullets.
    tech: list[str] = []
    start: YearMonth | None = None
    end: YearMonth | Literal["present"] | None = None
    bullets: list[Bullet] = Field(min_length=1)


class Education(Strict):
    id: Id
    school: NonEmpty
    degree: NonEmpty
    location: str | None = None
    start: YearMonth
    end: YearMonth | Literal["present"]
    gpa: str | None = None  # kept as text: "3.6/4.0", "8.2/10"
    bullets: list[Bullet] = []


class Skill(Strict):
    name: NonEmpty
    level: Literal["beginner", "intermediate", "advanced"] | None = None
    # Group heading on the CV, e.g. "LLM/AI Agents"; groups appear in order of first use.
    category: str | None = None
    evidence: list[Id] = Field(min_length=1, description="Ids of bullets that prove this skill.")


class Publication(Strict):
    id: Id
    title: NonEmpty
    venue: NonEmpty
    year: int
    url: str | None = None
    bullets: list[Bullet] = []


class Award(Strict):
    id: Id
    title: NonEmpty
    issuer: str | None = None
    date: YearMonth | None = None
    bullets: list[Bullet] = []


class Certification(Strict):
    id: Id
    name: NonEmpty
    issuer: str | None = None
    date: YearMonth | None = None
    url: str | None = None
    bullets: list[Bullet] = []


class Profile(Strict):
    basics: Basics
    summary_variants: list[SummaryVariant] = []
    experiences: list[Experience] = []
    projects: list[Project] = []
    education: list[Education] = []
    skills: list[Skill] = []
    publications: list[Publication] = []
    awards: list[Award] = []
    certifications: list[Certification] = []


# Profile fields whose entries have an id and a list of bullets.
SECTIONS = ("experiences", "projects", "education", "publications", "awards", "certifications")
