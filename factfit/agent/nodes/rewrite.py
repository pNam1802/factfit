"""Nodes `rewrite` and `ground_check`: rewrite selected bullets for a job, then verify them.

Flow for one tailoring run (each step maps to one LangGraph node in step 3e):

1. rewrite_entry   one LLM call per CV entry, one rewritten bullet per source bullet
2. check_bullet    rules first (numbers, technologies, role); the LLM judge only if the
                   rules pass, so a failing rewrite never pays for a judge call. A rewrite
                   equal to its source sentence is not checked at all
3. retry           failed bullets are rewritten once, with the problems listed
4. fallback        still failing: use the source bullet's own text, flagged for review

Every bullet that leaves this module is grounded: either it passed both checks, or it is the
source text itself.
"""

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from factfit.agent.nodes.select import SelectedEntry
from factfit.grounding.judge import judge_bullet
from factfit.grounding.rules import KnowledgeBase, check_bullet, load_kb
from factfit.llm import LLMClient, LLMError
from factfit.prompts import load_prompt
from factfit.schemas.base import Strict
from factfit.schemas.job import JobDescription
from factfit.schemas.match import MatchResult
from factfit.schemas.profile import Bullet

NODE = "rewrite"
MAX_ATTEMPTS = 2  # first rewrite + one retry with feedback


class RewrittenBullet(Strict):
    source_id: str
    text: str


class RewriteOutput(Strict):
    """What the LLM returns. Sent as a Structured Outputs schema, so no defaults."""

    bullets: list[RewrittenBullet]


@dataclass
class Draft:
    entry: SelectedEntry
    source: Bullet
    text: str = ""
    attempts: int = 0
    issues: list[str] = field(default_factory=list)  # problems of the latest attempt
    history: list[tuple[str, list[str]]] = field(default_factory=list)  # (text, problems)
    passed: bool = False
    fallback: bool = False


def allowed_keywords(entry: SelectedEntry, jd: JobDescription, kb: KnowledgeBase) -> list[str]:
    """JD keywords this entry's source bullets already support (by alias-normalised name)."""
    present = set()
    for s in entry.bullets:
        present |= {str(c) for _, c in kb.tech.find(s.bullet.text)}
        present |= {kb.canonical(x) for x in s.bullet.skills}
    return [k for k in jd.keywords if kb.canonical(k) in present]


def rewrite_entry(
    entry: SelectedEntry,
    drafts: list[Draft],
    *,
    jd: JobDescription,
    match: MatchResult,
    client: LLMClient,
    kb: KnowledgeBase,
    run_id: str | None = None,
) -> None:
    """Rewrite the given drafts of one entry in a single call; fills draft.text."""
    prompt = load_prompt(NODE, client.config.prompt_for(NODE))
    ids = {d.source.id for d in drafts}
    texts = {r.id: r.text for r in jd.all_requirements()}
    proven = [
        f"- {texts[r.req_id]} ({r.status})"
        for r in match.requirements
        if r.status in ("met", "partial") and ids & set(r.evidence) and r.req_id in texts
    ]
    previous = [
        f'- {d.source.id}: "{text}" -> problems: {"; ".join(problems)}'
        for d in drafts
        for text, problems in d.history
    ]
    out = client.parse(
        RewriteOutput,
        node=NODE,
        prompt_version=prompt.id,
        system=prompt.system,
        user=prompt.render(
            job_title=jd.title,
            entry=entry.title,
            allowed_keywords=", ".join(allowed_keywords(entry, jd, kb)) or "(none)",
            requirements="\n".join(proven) or "- (none matched)",
            bullets="\n".join(
                f"{d.source.id} | {d.source.text} | {', '.join(d.source.skills)}" for d in drafts
            ),
            previous=("\nPREVIOUS ATTEMPTS:\n" + "\n".join(previous)) if previous else "",
        ),
        run_id=run_id,
    )
    by_id = {b.source_id: b.text.strip() for b in out.bullets if b.source_id in ids}
    for d in drafts:
        d.attempts += 1
        d.text = by_id.get(d.source.id, "")


def same_text(a: str, b: str) -> bool:
    """Equal up to spacing, letter case and a final full stop: such a rewrite adds nothing."""

    def norm(t: str) -> str:
        return " ".join(t.split()).rstrip(".").casefold()

    return norm(a) == norm(b)


def check_draft(
    draft: Draft,
    *,
    client: LLMClient,
    kb: KnowledgeBase,
    use_judge: bool,
    run_id=None,
    all_problems: bool = False,
) -> None:
    """Rules first; the judge only if rules pass, unless all_problems (a person's edit:
    report everything at once instead of one layer per round trip)."""
    if not draft.text:
        draft.issues, draft.passed = ["the model returned no rewrite for this bullet"], False
        return
    if same_text(draft.text, draft.source.text):
        # The source sentence itself is grounded by definition: no rules, no judge call.
        draft.issues, draft.passed = [], True
        return
    issues = [i.message for i in check_bullet(draft.text, [draft.source], kb)]
    if use_judge and (all_problems or not issues):
        try:
            verdict = judge_bullet(draft.text, [draft.source], client, run_id=run_id)
        except LLMError as e:
            # Unchecked is not passed: the bullet is retried, then falls back to its source
            # text, which is always grounded. One network failure must not stop the run.
            issues.append(f"could not run the judge, so this rewrite is not trusted: {e}")
        else:
            issues += [f"{c.kind}: '{c.claim}' ({c.why})" for c in verdict.unsupported_claims]
    draft.issues = issues
    draft.passed = not draft.issues


_NAME_DROP_OPENING = re.compile(
    r"^(used|using|utili[sz]ed|leveraged|applied|employed)\b", re.IGNORECASE
)


def style_problems(drafts: list[Draft], pending: list[Draft], kb: KnowledgeBase) -> None:
    """Flag rewrites that read like keyword stuffing, even when every word is true.

    Seen in real runs: the same JD term taken from bullets' skills lists ("Python") was put
    into every bullet of an entry, and openings became "Used Python to build ...". Within
    one entry, a technology the source sentence does not name may be added to one bullet
    only; and a rewrite may not open with "Used / Leveraged ..." unless its source does.
    Checks only grounded drafts in `pending`; adds problems and marks them failed.
    """
    seen: dict[int, dict[str, str]] = {}  # entry -> technology added -> bullet that added it

    def added(d: Draft) -> set[str]:
        in_text = {str(c) for _, c in kb.tech.find(d.source.text)}
        return {str(c) for _, c in kb.tech.find(d.text)} - in_text

    pending_ids = {id(d) for d in pending}
    for d in drafts:  # bullets that already passed claim their additions first
        if id(d) not in pending_ids and d.passed and not d.fallback:
            for tech in added(d):
                seen.setdefault(id(d.entry), {}).setdefault(tech, d.source.id)
    for d in pending:
        if not d.passed:
            continue
        problems = []
        entry_seen = seen.setdefault(id(d.entry), {})
        new = added(d)
        for tech in sorted(new):
            if tech in entry_seen:
                problems.append(
                    f"style: '{tech}' is already added to {entry_seen[tech]} in this entry; "
                    "add a term from the skills list to one bullet per entry at most"
                )
        if _NAME_DROP_OPENING.match(d.text) and not _NAME_DROP_OPENING.match(d.source.text):
            problems.append(
                "style: the rewrite opens with a tool name ('Used X to ...'); keep the "
                "source's opening verb"
            )
        if problems:
            d.issues, d.passed = d.issues + problems, False
        else:
            for tech in new:
                entry_seen.setdefault(tech, d.source.id)


def rewrite_and_check(
    selected: list[SelectedEntry],
    *,
    jd: JobDescription,
    match: MatchResult,
    client: LLMClient,
    kb: KnowledgeBase | None = None,
    use_judge: bool = True,
    workers: int = 6,
    run_id: str | None = None,
    progress: Callable[[str], None] = lambda m: None,
) -> list[Draft]:
    kb = kb or load_kb()
    drafts = [Draft(entry, s.bullet) for entry in selected for s in entry.bullets]

    def by_entry(items: list[Draft]) -> list[tuple[SelectedEntry, list[Draft]]]:
        groups: dict[int, tuple[SelectedEntry, list[Draft]]] = {}
        for d in items:
            groups.setdefault(id(d.entry), (d.entry, []))[1].append(d)
        return list(groups.values())

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = drafts
        for attempt in range(1, MAX_ATTEMPTS + 1):
            progress(f"rewrite attempt {attempt}: {len(pending)} bullet(s)")
            list(
                pool.map(
                    lambda g: rewrite_entry(
                        g[0], g[1], jd=jd, match=match, client=client, kb=kb, run_id=run_id
                    ),
                    by_entry(pending),
                )
            )
            list(
                pool.map(
                    lambda d: check_draft(
                        d, client=client, kb=kb, use_judge=use_judge, run_id=run_id
                    ),
                    pending,
                )
            )
            style_problems(drafts, pending, kb)
            for d in pending:
                if not d.passed:
                    d.history.append((d.text, d.issues))
            pending = [d for d in pending if not d.passed]
            if not pending:
                break

    for d in pending:  # still failing after the retry: keep the source, flag it
        d.text, d.fallback, d.passed = d.source.text, True, True
    return drafts
