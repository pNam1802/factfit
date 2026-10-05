"""Tailor a CV for one job: select -> rewrite -> check -> assemble.

`tailor()` runs the steps in order, as plain functions. Step 3e wraps the same functions as
LangGraph nodes with a pause for human review; nothing here depends on the graph.
"""

from collections.abc import Callable
from dataclasses import dataclass

from factfit.agent.nodes.rewrite import Draft, rewrite_and_check
from factfit.agent.nodes.select import SelectedEntry, TailorConfig, select_bullets
from factfit.grounding.rules import KnowledgeBase, load_kb
from factfit.llm import LLMClient
from factfit.schemas.job import JobDescription
from factfit.schemas.match import MatchResult
from factfit.schemas.profile import Profile
from factfit.schemas.tailored import TailoredBullet, TailoredCV, TailoredSection, TailoredSummary


@dataclass
class TailorRun:
    selected: list[SelectedEntry]
    drafts: list[Draft]
    cv: TailoredCV


def tailor(
    profile: Profile,
    jd: JobDescription,
    match: MatchResult,
    *,
    client: LLMClient,
    config: TailorConfig | None = None,
    use_judge: bool = True,
    run_id: str | None = None,
    progress: Callable[[str], None] = lambda m: None,
) -> TailorRun:
    kb = load_kb()
    selected = select_bullets(profile, jd, match, config=config, kb=kb)
    progress(
        f"selected {sum(len(s.bullets) for s in selected)} bullets from {len(selected)} entries"
    )
    drafts = rewrite_and_check(
        selected, jd=jd, match=match, client=client, kb=kb, use_judge=use_judge,
        run_id=run_id, progress=progress,
    )  # fmt: skip
    return TailorRun(selected, drafts, assemble(selected, drafts, profile, jd, kb))


def assemble(
    selected: list[SelectedEntry],
    drafts: list[Draft],
    profile: Profile,
    jd: JobDescription,
    kb: KnowledgeBase,
) -> TailoredCV:
    by_entry: dict[int, list[Draft]] = {}
    for d in drafts:
        by_entry.setdefault(id(d.entry), []).append(d)

    sections = [
        TailoredSection(
            type=s.section,
            ref=s.entry.id,
            bullets=[
                TailoredBullet(
                    text=d.text,
                    source_bullet_ids=[d.source.id],
                    grounding="pass" if d.passed else "fail",
                    issues=[p for _, problems in d.history for p in problems],
                    fallback=d.fallback,
                    attempts=d.attempts,
                )
                for d in by_entry.get(id(s), [])
            ],
        )
        for s in selected
    ]
    used = {d.source.id for d in drafts}
    return TailoredCV(
        summary=pick_summary(profile, jd, kb),
        sections=sections,
        skills=pick_skills(profile, used, jd, kb),
    )


def pick_summary(profile: Profile, jd: JobDescription, kb: KnowledgeBase) -> TailoredSummary | None:
    """The summary variant whose tags overlap the job most; ties keep the profile's order."""
    if not profile.summary_variants:
        return None
    job_words = {k.casefold() for k in jd.keywords} | set(jd.title.casefold().split())
    job_words |= {kb.canonical(k).casefold() for k in jd.keywords}

    def overlap(v) -> int:
        return sum(any(t.casefold() in w or w in t.casefold() for w in job_words) for t in v.tags)

    best = max(profile.summary_variants, key=overlap)
    return TailoredSummary(text=best.text, source_ids=[best.id])


def pick_skills(
    profile: Profile, used_bullets: set[str], jd: JobDescription, kb: KnowledgeBase
) -> list[str]:
    """Skills proven by a bullet on this CV; those the job asks for come first."""
    wanted = {kb.canonical(k) for k in jd.keywords}
    proven = [s.name for s in profile.skills if used_bullets & set(s.evidence)]
    return sorted(proven, key=lambda name: kb.canonical(name) not in wanted)
