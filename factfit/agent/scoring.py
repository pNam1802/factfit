"""Match score: a fixed formula over requirement statuses, weights in config/match.yaml."""

from pathlib import Path

import yaml

from factfit.schemas.base import Strict
from factfit.schemas.job import Category, JobDescription
from factfit.schemas.match import RequirementMatch

DEFAULT_WEIGHTS = Path("config/match.yaml")


class Weights(Strict):
    bucket: dict[str, float]  # must_have / nice_to_have
    category: dict[Category, float]
    credit: dict[str, float]  # met / partial / missing


def load_weights(path: str | Path = DEFAULT_WEIGHTS) -> Weights:
    return Weights.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def compute_score(jd: JobDescription, judged: list[RequirementMatch], weights: Weights) -> int:
    info = {r.id: ("must_have", r.category) for r in jd.must_have}
    info |= {r.id: ("nice_to_have", r.category) for r in jd.nice_to_have}

    total = earned = 0.0
    for r in judged:
        if r.status == "unverifiable" or r.req_id not in info:
            continue
        bucket, category = info[r.req_id]
        weight = weights.bucket[bucket] * weights.category[category]
        total += weight
        earned += weight * weights.credit[r.status]
    return round(100 * earned / total) if total else 0
