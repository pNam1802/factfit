"""Score grounding checks on the test set in evals/grounding/cases.jsonl (PRD §8.3, §10.1).

For each mutation type:
- fabricated types: recall = share of cases the checker flags
- valid_paraphrase:  false positive rate = share of faithful rewrites it wrongly flags
Each rate comes with a 95% Wilson interval.

Layers: "rules" (code, factfit.grounding.rules). The LLM judge is added in step 3c so the
three configurations (rules, judge, both) can be compared (ablation).
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass

from factfit.evals.grounding_cases import Case
from factfit.evals.stats import wilson
from factfit.grounding.rules import check_bullet
from factfit.schemas.profile import Bullet

Checker = Callable[[str, list[Bullet]], bool]  # True = flagged as not grounded


def rules_checker(text: str, sources: list[Bullet]) -> bool:
    return bool(check_bullet(text, sources))


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
            hit = checker(case.text, [sources[s] for s in case.source_ids])
            flagged += hit
            wrong = (not hit) if case.label == "fabricated" else hit
            if wrong:
                misses.append(case)
        results.append(TypeResult(mutation_type, group[0].label, len(group), flagged, misses))
    order = ["number_change", "tech_injection", "role_inflation", "scale_inflation",
             "unsupported_outcome", "valid_paraphrase"]  # fmt: skip
    return sorted(results, key=lambda r: order.index(r.mutation_type))


def summary(results: list[TypeResult]) -> dict[str, float | None]:
    fab = [r for r in results if r.label == "fabricated"]
    ok = [r for r in results if r.label == "grounded"]
    caught, total = sum(r.flagged for r in fab), sum(r.n for r in fab)
    fp, n_ok = sum(r.flagged for r in ok), sum(r.n for r in ok)
    return {
        "recall": caught / total if total else None,
        "false_positive_rate": fp / n_ok if n_ok else None,
    }
