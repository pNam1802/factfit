"""Score grounding checkers on the test set (PRD §8.3, §10.1), including the ablation.

For each mutation type:
- fabricated types: recall = share of cases the checker flags
- grounded types (paraphrases): false positive rate = share it wrongly flags
Each rate comes with a 95% Wilson interval.

Checkers compared (ablation):
- rules:        code only (factfit.grounding.rules), free and instant
- judge:        the LLM judge only (factfit.grounding.judge), per model / reasoning effort
- rules+judge:  flagged if either flags, the configuration used in the product

Judge verdicts are cached in evals/results/judge-cache.jsonl, so re-scoring costs nothing.
"""

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from factfit.db.models import LLMCall
from factfit.evals.grounding_cases import Case
from factfit.evals.stats import wilson
from factfit.grounding.judge import NODE as JUDGE_NODE
from factfit.grounding.judge import judge_bullet
from factfit.grounding.rules import check_bullet
from factfit.llm import LLMClient
from factfit.llm.config import LLMConfig, NodeSettings
from factfit.prompts import load_prompt
from factfit.schemas.profile import Bullet

Checker = Callable[[Case, list[Bullet]], bool]  # True = flagged as not grounded
CACHE = Path("evals/results/judge-cache.jsonl")
ORDER = ["number_change", "tech_injection", "role_inflation", "scale_inflation",
         "unsupported_outcome", "subtle_claim", "valid_paraphrase", "hard_paraphrase"]  # fmt: skip


def rules_checker(case: Case, sources: list[Bullet]) -> bool:
    return bool(check_bullet(case.text, sources))


@dataclass
class TypeResult:
    mutation_type: str
    label: str
    n: int
    flagged: int
    misses: list[Case]  # fabricated but not flagged, or grounded but flagged

    @property
    def rate(self) -> float:
        return self.flagged / self.n if self.n else 0.0

    @property
    def interval(self) -> tuple[float, float] | None:
        return wilson(self.flagged, self.n)


def evaluate(cases: list[Case], sources: dict[str, Bullet], checker: Checker) -> list[TypeResult]:
    groups: dict[str, list[Case]] = defaultdict(list)
    for case in cases:
        groups[case.mutation_type].append(case)

    results = []
    for mutation_type, group in groups.items():
        flagged, misses = 0, []
        for case in group:
            hit = checker(case, [sources[s] for s in case.source_ids])
            flagged += hit
            wrong = (not hit) if case.label == "fabricated" else hit
            if wrong:
                misses.append(case)
        results.append(TypeResult(mutation_type, group[0].label, len(group), flagged, misses))
    return sorted(results, key=lambda r: ORDER.index(r.mutation_type))


def summary(results: list[TypeResult]) -> dict[str, float | None]:
    fab = [r for r in results if r.label == "fabricated"]
    ok = [r for r in results if r.label == "grounded"]
    caught, total = sum(r.flagged for r in fab), sum(r.n for r in fab)
    fp, n_ok = sum(r.flagged for r in ok), sum(r.n for r in ok)
    return {
        "recall": caught / total if total else None,
        "false_positive_rate": fp / n_ok if n_ok else None,
    }


# --- the LLM judge, with a cache --------------------------------------------------------------


@dataclass(frozen=True)
class JudgeConfig:
    model: str
    effort: str | None  # None = the model's default

    @property
    def label(self) -> str:
        return f"{self.model}/{self.effort or 'default'}"

    @classmethod
    def parse(cls, spec: str) -> "JudgeConfig":
        """ "gpt-5-mini:low" -> JudgeConfig("gpt-5-mini", "low")."""
        model, _, effort = spec.partition(":")
        return cls(model, None if effort in ("", "default") else effort)


@dataclass
class Verdict:
    flagged: bool
    claims: list[dict]
    cost_usd: float | None
    latency_ms: int


def _cache_key(prompt_id: str, cfg: JudgeConfig, case: Case, sources: list[Bullet]) -> str:
    raw = json.dumps([prompt_id, cfg.model, cfg.effort, case.text, [b.text for b in sources]])
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def judge_cases(
    cases: list[Case],
    sources: dict[str, Bullet],
    cfg: JudgeConfig,
    *,
    backend,
    base_config: LLMConfig,
    workers: int = 8,
    cache_path: Path = CACHE,
    progress: Callable[[str], None] = lambda m: None,
) -> dict[str, Verdict]:
    """Run the judge on every case (cached); return verdicts by case id."""
    config = base_config.model_copy(deep=True)
    prompt_version = config.nodes[JUDGE_NODE].prompt
    config.nodes[JUDGE_NODE] = NodeSettings(
        model=cfg.model, prompt=prompt_version, reasoning_effort=cfg.effort
    )
    prompt_id = load_prompt(JUDGE_NODE, prompt_version).id

    cached: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cached[row["key"]] = row

    def one(case: Case) -> tuple[str, Verdict, dict | None]:
        srcs = [sources[s] for s in case.source_ids]
        key = _cache_key(prompt_id, cfg, case, srcs)
        if key in cached:
            r = cached[key]
            return case.id, Verdict(r["flagged"], r["claims"], r["cost_usd"], r["latency_ms"]), None
        calls: list[LLMCall] = []
        client = LLMClient(backend, config, log=calls.append)
        verdict = judge_bullet(case.text, srcs, client)
        costs = [c.cost_usd for c in calls if c.cost_usd is not None]
        v = Verdict(
            flagged=not verdict.grounded,
            claims=[c.model_dump() for c in verdict.unsupported_claims],
            cost_usd=sum(costs) if costs else None,
            latency_ms=sum(c.latency_ms for c in calls),
        )
        return case.id, v, {"key": key, **vars(v)}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        out = list(pool.map(one, cases))

    new_rows = [row for _, _, row in out if row]
    if new_rows:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("a", encoding="utf-8", newline="\n") as f:
            for row in new_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    progress(f"  {cfg.label}: {len(new_rows)} new judge calls, {len(out) - len(new_rows)} cached")
    return {case_id: v for case_id, v, _ in out}


def cost_and_latency(verdicts: dict[str, Verdict]) -> dict[str, float | None]:
    costs = [v.cost_usd for v in verdicts.values() if v.cost_usd is not None]
    return {
        "cost_per_1000": 1000 * sum(costs) / len(costs) if costs else None,
        "latency_p50_s": median(v.latency_ms for v in verdicts.values()) / 1000,
    }


def judge_checker(verdicts: dict[str, Verdict]) -> Checker:
    return lambda case, _sources: verdicts[case.id].flagged


def combined_checker(verdicts: dict[str, Verdict]) -> Checker:
    return lambda case, sources: rules_checker(case, sources) or verdicts[case.id].flagged
