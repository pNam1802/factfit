"""Shorten a reviewed CV until its PDF fits on one page (PRD F6: a tailored CV is 1 page).

Code only, no LLM: the CV is rendered, and while it runs over a page the least useful item
is removed and it is rendered again. Removal order:

1. the bullet with the lowest selection score, in an entry that keeps at least one bullet
   (on a tie, a bullet without a verified number goes first);
2. once every entry is down to one bullet, the project whose bullet scores lowest.

Experiences are never removed: a gap in the timeline costs more than a long CV. Skills are
recomputed from the bullets that remain, so every listed skill keeps its evidence on the page.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from factfit.agent.tailor import pick_skills
from factfit.grounding.rules import KnowledgeBase
from factfit.render.compile import TailoredRender
from factfit.schemas.job import JobDescription
from factfit.schemas.profile import Profile
from factfit.schemas.tailored import TailoredCV

Render = Callable[[Profile, TailoredCV, Path, list[str]], TailoredRender]


@dataclass
class Fitted:
    cv: TailoredCV
    render: TailoredRender
    removed: list[str] = field(default_factory=list)  # what was cut, for the person to see


def fit_one_page(
    profile: Profile,
    cv: TailoredCV,
    jd: JobDescription,
    kb: KnowledgeBase,
    scores: dict[str, float],  # source bullet id -> selection score
    out_dir: Path,
    render: Render,
) -> Fitted:
    removed: list[str] = []
    while True:
        out = render(profile, cv, out_dir, jd.keywords)
        if out.pages <= 1:
            return Fitted(cv, out, removed)
        step = _trim(cv, profile, scores)
        if step is None:  # nothing left to cut: report the page count as a problem
            return Fitted(cv, out, removed)
        cv, what = step
        used = {i for b in cv.all_bullets() for i in b.source_bullet_ids}
        cv = cv.model_copy(update={"skills": [
            s for s in pick_skills(profile, used, jd, kb) if s in set(cv.skills)
        ]})  # fmt: skip
        removed.append(what)


def _trim(
    cv: TailoredCV, profile: Profile, scores: dict[str, float]
) -> tuple[TailoredCV, str] | None:
    def score(bullet) -> float:
        return max(scores.get(i, 0.0) for i in bullet.source_bullet_ids)

    names = {e.id: e.org for e in profile.experiences} | {p.id: p.name for p in profile.projects}
    measured = {  # bullets backed by a verified number: on a tie, these stay
        b.id
        for e in [*profile.experiences, *profile.projects]
        for b in e.bullets
        if any(m.verified for m in b.metrics)
    }

    def has_number(bullet) -> bool:
        return bool(measured & set(bullet.source_bullet_ids))

    sections = [s.model_copy(deep=True) for s in cv.sections]

    # 1. Lowest-scoring bullet among entries that keep at least one. On a tie, one without
    # a verified number goes first, then the later one.
    candidates = [
        (score(b), has_number(b), si, bi)
        for si, s in enumerate(sections)
        if len(s.bullets) > 1
        for bi, b in enumerate(s.bullets)
    ]
    if candidates:
        _, _, si, bi = min(candidates, key=lambda c: (c[0], c[1], -c[2], -c[3]))
        bullet = sections[si].bullets.pop(bi)
        what = f"bullet in {names.get(sections[si].ref, sections[si].ref)}: “{_short(bullet.text)}”"
        return cv.model_copy(update={"sections": sections}), what

    # 2. Lowest-scoring project.
    projects = [
        (max((score(b) for b in s.bullets), default=0.0), si)
        for si, s in enumerate(sections)
        if s.type == "project"
    ]
    if projects:
        _, si = min(projects, key=lambda p: (p[0], -p[1]))
        section = sections.pop(si)
        return cv.model_copy(update={"sections": sections}), (
            f"project {names.get(section.ref, section.ref)}"
        )
    return None


def _short(text: str, n: int = 70) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"
