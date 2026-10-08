"""Cost of tailoring runs, from the llm_calls log (PRD §4: < $0.05 per run, parse to checked CV).

Calls are grouped by run_id; a run counts as a tailoring run when it made a rewrite call.
Parse and match are cached per JD, so a run on a JD seen before shows no cost for them:
read the target against runs on new JDs.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from sqlmodel import Session, select

from factfit.db import LLMCall


@dataclass
class RunCost:
    run_id: str
    started: datetime
    calls: int = 0
    cost: float = 0.0
    unpriced: int = 0  # calls to a model with no price in config/llm.yaml
    by_node: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    calls_by_node: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_model: dict[str, float] = field(default_factory=lambda: defaultdict(float))


def run_costs(session: Session, limit: int = 10) -> list[RunCost]:
    """The latest `limit` tailoring runs, newest first."""
    calls = session.exec(
        select(LLMCall).where(LLMCall.run_id.is_not(None)).order_by(LLMCall.created_at)
    ).all()
    runs: dict[str, RunCost] = {}
    for c in calls:
        run = runs.setdefault(c.run_id, RunCost(c.run_id, c.created_at))
        run.calls += 1
        run.calls_by_node[c.node] += 1
        if c.cost_usd is None:
            run.unpriced += 1
            continue
        run.cost += c.cost_usd
        run.by_node[c.node] += c.cost_usd
        run.by_model[c.model] += c.cost_usd
    tailoring = [r for r in runs.values() if r.calls_by_node.get("rewrite")]
    return sorted(tailoring, key=lambda r: r.started, reverse=True)[:limit]
