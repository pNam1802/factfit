import json

import pytest

from factfit.evals.parse_eval import Labels, align, load_cases, score
from factfit.llm import LLMClient
from factfit.llm.backends import RawOutput
from factfit.llm.config import LLMConfig
from factfit.schemas.job import JobDescription

LABELS = Labels.model_validate(
    {
        "must_have": [
            {"text": "Python"},
            {"text": "One ML framework", "any_of": ["PyTorch", "TensorFlow"]},
            {"text": "4+ years", "level": "senior"},
        ],
        "nice_to_have": [{"text": "Docker"}],
    }
)


def req(id_, text, any_of=(), level=None):
    return {"id": id_, "text": text, "category": "skill", "any_of": list(any_of), "level": level}


PREDICTED = JobDescription.model_validate(
    {
        "title": "AI Engineer",
        "company": None,
        "seniority": "mid",
        "language": "en",
        "location": None,
        "must_have": [
            req("r1", "Strong Python"),
            req("r2", "PyTorch"),  # the alternatives were split: any_of missing
            req("r3", "TensorFlow"),
            req("r4", "4+ years"),  # level missing
        ],
        "nice_to_have": [],  # Docker missed entirely
        "responsibilities": [],
        "keywords": [],
    }
)


def test_score_counts_each_dimension():
    # G1=Python, G2=framework, G3=years, G4=Docker; P1..P4 as above
    counts = score(LABELS, PREDICTED, {"G1": "P1", "G2": "P2", "G3": "P4"})
    m = counts.metrics()
    assert m["precision"] == pytest.approx(3 / 4)  # P3 matched nothing
    assert m["recall"] == pytest.approx(3 / 4)  # Docker not found
    assert m["bucket"] == 1.0
    assert m["any_of"] == 0.0  # framework found, but not as alternatives
    assert m["level"] == pytest.approx(2 / 3)  # "4+ years" lost its senior level


class JudgeBackend:
    def __init__(self, pairs):
        self.pairs = pairs

    def complete(self, *, node, model, system, user, schema, options):
        return RawOutput(text=json.dumps({"pairs": self.pairs}))


def judge(pairs):
    config = LLMConfig.model_validate(
        {"models": {"small": "m"}, "nodes": {"eval_align": {"model": "small"}}}
    )
    return LLMClient(JudgeBackend(pairs), config)


def test_align_drops_unknown_and_repeated_ids():
    pairs = [
        {"gold": "G1", "pred": "P1"},
        {"gold": "G1", "pred": "P2"},  # G1 already paired
        {"gold": "G2", "pred": "P1"},  # P1 already used
        {"gold": "G9", "pred": "P3"},  # no such gold
        {"gold": "G3", "pred": "P4"},
    ]
    assert align(judge(pairs), LABELS, PREDICTED) == {"G1": "P1", "G3": "P4"}


def test_load_cases_pairs_text_and_labels(tmp_path):
    split = tmp_path / "dev"
    split.mkdir()
    (split / "a.txt").write_text("JD text", encoding="utf-8")
    (split / "a.labels.yaml").write_text("must_have:\n  - text: Python\n", encoding="utf-8")
    (split / "unlabelled.txt").write_text("skipped", encoding="utf-8")

    (case,) = load_cases("dev", root=tmp_path)
    assert case.name == "a" and case.text == "JD text"
    assert case.labels.must_have[0].text == "Python"
