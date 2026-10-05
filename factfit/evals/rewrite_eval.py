"""Compare rewrite prompt versions on real job descriptions.

Measured on each run's FIRST attempt (what the prompt produces before any retry):
- stuffing: bullets that add a technology already added to another bullet of the same entry
- name_drop: bullets that open with "Used / Leveraged ... X to" when the source does not
- first_pass: bullets that pass grounding and style on the first attempt
And on the final result: retries, fallbacks to the original, unchanged bullets, cost.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

from factfit.agent.nodes.rewrite import _NAME_DROP_OPENING, Draft
from factfit.agent.tailor import tailor
from factfit.db.models import LLMCall
from factfit.grounding.rules import KnowledgeBase, load_kb
from factfit.llm import LLMClient
from factfit.llm.config import LLMConfig, NodeSettings
from factfit.schemas.job import JobDescription
from factfit.schemas.match import MatchResult
from factfit.schemas.profile import Profile


@dataclass
class RunStats:
    bullets: int = 0
    stuffing: int = 0
    name_drop: int = 0
    first_pass: int = 0
    retried: int = 0
    fallback: int = 0
    unchanged: int = 0
    cost: float = 0.0
    examples: list[str] = field(default_factory=list)

    def add(self, other: "RunStats") -> None:
        for name in ("bullets", "stuffing", "name_drop", "first_pass", "retried", "fallback",
                     "unchanged", "cost"):  # fmt: skip
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.examples += other.examples


def first_attempt(d: Draft) -> str:
    return d.history[0][0] if d.history else d.text


def measure(drafts: list[Draft], kb: KnowledgeBase) -> RunStats:
    stats = RunStats(bullets=len(drafts))
    added_by_entry: dict[int, set[str]] = {}
    for d in drafts:
        text = first_attempt(d)
        in_source = {str(c) for _, c in kb.tech.find(d.source.text)}
        added = {str(c) for _, c in kb.tech.find(text)} - in_source
        seen = added_by_entry.setdefault(id(d.entry), set())
        if added & seen:
            stats.stuffing += 1
            stats.examples.append(f"[stuffing] {text}")
        seen |= added
        if _NAME_DROP_OPENING.match(text) and not _NAME_DROP_OPENING.match(d.source.text):
            stats.name_drop += 1
            stats.examples.append(f"[name drop] {text}")
        stats.first_pass += not d.history and not d.fallback
        stats.retried += bool(d.history) and not d.fallback
        stats.fallback += d.fallback
        stats.unchanged += d.text.rstrip(".") == d.source.text.rstrip(".")
    return stats


def run_prompt(
    prompt: str,
    cases: list[tuple[str, JobDescription, MatchResult]],
    profile: Profile,
    *,
    backend,
    base_config: LLMConfig,
    progress: Callable[[str], None] = lambda m: None,
) -> RunStats:
    config = base_config.model_copy(deep=True)
    current = config.nodes["rewrite"]
    config.nodes["rewrite"] = NodeSettings(
        model=current.model, prompt=prompt, reasoning_effort=current.reasoning_effort
    )
    kb = load_kb()
    total = RunStats()
    for name, jd, match in cases:
        calls: list[LLMCall] = []
        client = LLMClient(backend, config, log=calls.append)
        run = tailor(profile, jd, match, client=client)
        stats = measure(run.drafts, kb)
        stats.cost = sum(c.cost_usd or 0 for c in calls)
        progress(f"  {prompt} {name}: {stats.stuffing} stuffing, {stats.name_drop} name drops, "
                 f"{stats.fallback} fallbacks, ${stats.cost:.3f}")  # fmt: skip
        total.add(stats)
    return total
