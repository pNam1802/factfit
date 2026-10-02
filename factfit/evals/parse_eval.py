"""Evaluate `parse_jd` against hand labels (PRD §10.1).

For each job description `<name>.txt` with a `<name>.labels.yaml` next to it:

1. run the extractor with a given prompt version and reasoning effort;
2. ask an LLM judge to pair predicted requirements with gold ones by meaning;
3. score with code:
   - precision: share of predicted requirements that match a gold one
   - recall:    share of gold requirements that were found
   - bucket:    of the pairs, share put in the right list (must_have / nice_to_have)
   - any_of:    of gold requirements with alternatives that were found, share the
                extractor also marked as alternatives
   - level:     of the pairs, share with the right level

Metrics are micro-averaged: summed over all job descriptions, then divided.
"""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel

from factfit.agent.nodes.parse_jd import extract_jd
from factfit.db.models import LLMCall
from factfit.llm import LLMClient
from factfit.llm.config import LLMConfig, NodeSettings
from factfit.prompts import load_prompt
from factfit.schemas.base import Strict
from factfit.schemas.job import JobDescription, Requirement, Seniority

JDS_DIR = Path("evals/jds")
JUDGE_NODE = "eval_align"


# --- labels ----------------------------------------------------------------------------------


class GoldRequirement(Strict):
    text: str
    any_of: list[str] = []
    level: Seniority | None = None


class Labels(Strict):
    must_have: list[GoldRequirement]
    nice_to_have: list[GoldRequirement] = []


@dataclass
class Case:
    name: str
    text: str
    labels: Labels


def load_cases(split: str, root: Path = JDS_DIR) -> list[Case]:
    cases = []
    for label_file in sorted((root / split).glob("*.labels.yaml")):
        name = label_file.name.removesuffix(".labels.yaml")
        text_file = label_file.with_name(f"{name}.txt")
        labels = Labels.model_validate(yaml.safe_load(label_file.read_text(encoding="utf-8")))
        cases.append(Case(name, text_file.read_text(encoding="utf-8"), labels))
    return cases


# --- judge -----------------------------------------------------------------------------------


class Pair(Strict):
    gold: str  # "G3"
    pred: str  # "P5"


class Alignment(Strict):
    pairs: list[Pair]


def _gold_items(labels: Labels) -> list[tuple[str, GoldRequirement, str]]:
    items = [(r, "must") for r in labels.must_have] + [(r, "nice") for r in labels.nice_to_have]
    return [(f"G{i + 1}", r, bucket) for i, (r, bucket) in enumerate(items)]


def _pred_items(jd: JobDescription) -> list[tuple[str, Requirement, str]]:
    items = [(r, "must") for r in jd.must_have] + [(r, "nice") for r in jd.nice_to_have]
    return [(f"P{i + 1}", r, bucket) for i, (r, bucket) in enumerate(items)]


def align(judge: LLMClient, labels: Labels, jd: JobDescription) -> dict[str, str]:
    """Return {gold_id: pred_id}, one-to-one."""
    gold, pred = _gold_items(labels), _pred_items(jd)
    if not gold or not pred:
        return {}
    prompt = load_prompt(JUDGE_NODE, judge.config.prompt_for(JUDGE_NODE))
    result = judge.parse(
        Alignment,
        node=JUDGE_NODE,
        prompt_version=prompt.id,
        system=prompt.system,
        user=prompt.render(
            gold="\n".join(f"{gid}: {r.text}" for gid, r, _ in gold),
            predicted="\n".join(f"{pid}: {r.text}" for pid, r, _ in pred),
        ),
    )
    gold_ids, pred_ids = {g for g, _, _ in gold}, {p for p, _, _ in pred}
    pairs: dict[str, str] = {}
    used: set[str] = set()
    for p in result.pairs:  # keep only valid, one-to-one pairs
        if p.gold in gold_ids and p.pred in pred_ids and p.gold not in pairs and p.pred not in used:
            pairs[p.gold] = p.pred
            used.add(p.pred)
    return pairs


# --- scoring ---------------------------------------------------------------------------------


@dataclass
class Counts:
    predicted: int = 0
    gold: int = 0
    paired: int = 0
    bucket_ok: int = 0
    any_of_gold: int = 0  # paired gold requirements that have alternatives
    any_of_ok: int = 0
    level_ok: int = 0

    def add(self, other: "Counts") -> None:
        for name in self.__dataclass_fields__:
            setattr(self, name, getattr(self, name) + getattr(other, name))

    @staticmethod
    def _ratio(a: int, b: int) -> float | None:
        return a / b if b else None

    def metrics(self) -> dict[str, float | None]:
        return {
            "precision": self._ratio(self.paired, self.predicted),
            "recall": self._ratio(self.paired, self.gold),
            "bucket": self._ratio(self.bucket_ok, self.paired),
            "any_of": self._ratio(self.any_of_ok, self.any_of_gold),
            "level": self._ratio(self.level_ok, self.paired),
        }


def score(labels: Labels, jd: JobDescription, pairs: dict[str, str]) -> Counts:
    gold = {gid: (r, b) for gid, r, b in _gold_items(labels)}
    pred = {pid: (r, b) for pid, r, b in _pred_items(jd)}
    counts = Counts(predicted=len(pred), gold=len(gold), paired=len(pairs))
    for gid, pid in pairs.items():
        (g, g_bucket), (p, p_bucket) = gold[gid], pred[pid]
        counts.bucket_ok += g_bucket == p_bucket
        counts.level_ok += g.level == p.level
        if g.any_of:
            counts.any_of_gold += 1
            counts.any_of_ok += bool(p.any_of)
    return counts


# --- running a grid of variants --------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    prompt: str  # "v1"
    effort: str | None  # None = the model's default

    @property
    def label(self) -> str:
        return f"{self.prompt}/{self.effort or 'default'}"


@dataclass
class CaseResult:
    case: str
    counts: Counts
    parse_calls: list[LLMCall]
    error: str | None = None
    jd: JobDescription | None = None


@dataclass
class VariantResult:
    variant: Variant
    cases: list[CaseResult] = field(default_factory=list)

    def totals(self) -> Counts:
        total = Counts()
        for c in self.cases:
            total.add(c.counts)
        return total

    def parse_stats(self) -> dict[str, float | None]:
        ok_runs = [c for c in self.cases if c.error is None]
        calls = [call for c in ok_runs for call in c.parse_calls]
        costs = [call.cost_usd for call in calls if call.cost_usd is not None]
        n = len(ok_runs) or 1
        return {
            "cost_per_jd": sum(costs) / n if costs else None,
            "latency_s_per_jd": sum(call.latency_ms for call in calls) / 1000 / n,
            "failed": len(self.cases) - len(ok_runs),
        }


def variant_config(base: LLMConfig, variant: Variant) -> LLMConfig:
    config = base.model_copy(deep=True)
    current = config.nodes["parse_jd"]
    config.nodes["parse_jd"] = NodeSettings(
        model=current.model, prompt=variant.prompt, reasoning_effort=variant.effort
    )
    return config


def run(
    cases: list[Case],
    variants: list[Variant],
    *,
    backend,
    base_config: LLMConfig,
    workers: int = 6,
    progress: Callable[[str], None] = lambda msg: None,
) -> list[VariantResult]:
    judge_calls: list[LLMCall] = []
    judge = LLMClient(backend, base_config, log=judge_calls.append)

    def one(variant: Variant, case: Case) -> CaseResult:
        calls: list[LLMCall] = []
        client = LLMClient(backend, variant_config(base_config, variant), log=calls.append)
        prompt = load_prompt("parse_jd", variant.prompt)
        try:
            jd = extract_jd(case.text, client=client, prompt=prompt)
            counts = score(case.labels, jd, align(judge, case.labels, jd))
        except Exception as e:  # one failed case must not stop the whole grid
            progress(f"  {variant.label} {case.name}: FAILED {e}")
            return CaseResult(case.name, Counts(), calls, error=str(e))
        progress(f"  {variant.label} {case.name}: done")
        return CaseResult(case.name, counts, calls, jd=jd)

    jobs = [(v, c) for v in variants for c in cases]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outputs = list(pool.map(lambda job: one(*job), jobs))

    results = {v: VariantResult(v) for v in variants}
    for (variant, _), output in zip(jobs, outputs, strict=True):
        results[variant].cases.append(output)
    return list(results.values())


def to_json(results: list[VariantResult]) -> list[dict]:
    """Full results, including each predicted JD, for saving under evals/results/."""

    def dump(model: BaseModel | None):
        return model.model_dump(mode="json") if model else None

    return [
        {
            "variant": r.variant.label,
            "metrics": r.totals().metrics(),
            "parse": r.parse_stats(),
            "cases": [
                {
                    "case": c.case,
                    "error": c.error,
                    "metrics": c.counts.metrics(),
                    "counts": vars(c.counts),
                    "predicted": dump(c.jd),
                }
                for c in r.cases
            ],
        }
        for r in results
    ]
