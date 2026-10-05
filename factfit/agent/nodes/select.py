"""Node `select`: pick which profile bullets go on the CV for one job (code, no LLM).

Each bullet gets a score: evidence it gives for the job's requirements (from `match`), plus
JD keywords it mentions, plus how recent its entry is. Then:
- every experience stays, with its best few bullets (dropping a job from a CV leaves a gap);
- projects compete: only the best few projects, with their best bullets;
- original order inside each entry is kept, so the CV still reads naturally.
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from factfit.grounding.rules import KnowledgeBase, load_kb
from factfit.schemas.base import Strict
from factfit.schemas.job import JobDescription
from factfit.schemas.match import MatchResult
from factfit.schemas.profile import Bullet, Experience, Profile, Project

DEFAULT_CONFIG = Path("config/tailor.yaml")


class Limits(Strict):
    bullets_per_experience: int = 3
    projects: int = 3
    bullets_per_project: int = 2


class EvidenceWeights(Strict):
    must_met: float
    must_partial: float
    nice_met: float
    nice_partial: float


class Weights(Strict):
    evidence: EvidenceWeights
    keyword: float
    keyword_cap: float
    recency: float
    recency_years: float


class TailorConfig(Strict):
    limits: Limits
    weights: Weights


def load_tailor_config(path: str | Path = DEFAULT_CONFIG) -> TailorConfig:
    return TailorConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


@dataclass
class ScoredBullet:
    bullet: Bullet
    score: float
    reasons: list[str] = field(default_factory=list)  # "evidence for r3 (must, met)", ...


@dataclass
class SelectedEntry:
    section: str  # "experience" | "project"
    entry: Experience | Project
    bullets: list[ScoredBullet]  # in the profile's original order

    @property
    def title(self) -> str:
        if isinstance(self.entry, Experience):
            return f"{self.entry.role} at {self.entry.org}"
        return self.entry.name


def select_bullets(
    profile: Profile,
    jd: JobDescription,
    match: MatchResult,
    config: TailorConfig | None = None,
    kb: KnowledgeBase | None = None,
    today: date | None = None,
) -> list[SelectedEntry]:
    config = config or load_tailor_config()
    kb = kb or load_kb()
    today = today or date.today()
    w = config.weights

    must_ids = {r.id for r in jd.must_have}
    evidence: dict[str, list[tuple[str, float]]] = {}
    for r in match.requirements:
        if r.status not in ("met", "partial"):
            continue
        bucket = "must" if r.req_id in must_ids else "nice"
        weight = getattr(w.evidence, f"{bucket}_{r.status}")
        for bullet_id in r.evidence:
            evidence.setdefault(bullet_id, []).append(
                (f"{r.req_id} ({bucket}, {r.status})", weight)
            )

    keywords = {kb.canonical(k) for k in jd.keywords}

    def score(bullet: Bullet, end: str | None) -> ScoredBullet:
        reasons, total = [], 0.0
        for label, weight in evidence.get(bullet.id, []):
            total += weight
            reasons.append(f"evidence for {label}")
        found = {str(c) for _, c in kb.tech.find(bullet.text)} | {
            kb.canonical(s) for s in bullet.skills
        }
        hits = sorted(found & keywords)
        if hits:
            total += min(w.keyword * len(hits), w.keyword_cap)
            reasons.append(f"JD keywords: {', '.join(hits)}")
        fresh = _recency(end, today, w.recency_years)
        total += w.recency * fresh
        return ScoredBullet(bullet, round(total, 2), reasons)

    def best(entry, k: int) -> list[ScoredBullet]:
        scored = [score(b, entry.end) for b in entry.bullets]
        keep = {id(s) for s in sorted(scored, key=lambda s: -s.score)[:k]}
        return [s for s in scored if id(s) in keep]  # original order

    selected = [
        SelectedEntry("experience", e, best(e, config.limits.bullets_per_experience))
        for e in profile.experiences
    ]
    projects = [
        SelectedEntry("project", p, best(p, config.limits.bullets_per_project))
        for p in profile.projects
    ]
    ranked = sorted(projects, key=lambda s: -max((b.score for b in s.bullets), default=0))
    kept = {id(p.entry) for p in ranked[: config.limits.projects]}
    selected += [p for p in projects if id(p.entry) in kept]  # in the profile's order
    return selected


def _recency(end: str | None, today: date, years: float) -> float:
    """1.0 for an entry still running, fading to 0 after `years` years."""
    if end in (None, "present"):
        return 1.0
    year = int(end[:4])
    month = int(end[5:7]) if len(end) >= 7 else 12
    age = (today.year - year) + (today.month - month) / 12
    return max(0.0, 1 - age / years)
