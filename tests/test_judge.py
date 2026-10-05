import json

import pytest

from factfit.evals import grounding_cases as gc
from factfit.evals import grounding_eval as ge
from factfit.grounding.judge import judge_bullet
from factfit.llm import LLMClient
from factfit.llm.backends import RawOutput
from factfit.llm.config import LLMConfig
from factfit.schemas.profile import Bullet

CONFIG = LLMConfig.model_validate(
    {"models": {"small": "m"}, "nodes": {"judge": {"model": "small", "prompt": "v1"}}}
)
SOURCE = Bullet(id="s", text="Built a tracker with YOLOv8 running at 25 fps")


class JudgeBackend:
    def __init__(self, claims):
        self.claims = claims
        self.users: list[str] = []

    def complete(self, *, node, model, system, user, schema, options):
        self.users.append(user)
        return RawOutput(text=json.dumps({"unsupported_claims": self.claims}))


def test_judge_sends_sources_and_rewrite_and_reads_claims():
    claim = {"claim": "real-time", "kind": "inference", "why": "inferred from 25 fps"}
    backend = JudgeBackend([claim])
    verdict = judge_bullet("Built a real-time YOLOv8 tracker", [SOURCE], LLMClient(backend, CONFIG))
    assert not verdict.grounded and verdict.unsupported_claims[0].kind == "inference"
    assert "running at 25 fps" in backend.users[0] and "real-time" in backend.users[0]


def test_empty_claim_list_means_grounded():
    verdict = judge_bullet("Built a YOLOv8 tracker", [SOURCE], LLMClient(JudgeBackend([]), CONFIG))
    assert verdict.grounded


def test_judge_config_parsing():
    assert ge.JudgeConfig.parse("gpt-5-mini:low") == ge.JudgeConfig("gpt-5-mini", "low")
    assert ge.JudgeConfig.parse("gpt-5-mini").effort is None
    assert ge.JudgeConfig.parse("gpt-5-mini:default").label == "gpt-5-mini/default"


def test_judge_cases_caches_verdicts(tmp_path):
    sources = {"s": SOURCE}
    case = gc.Case(
        "x1", ["s"], "Built a real-time YOLOv8 tracker", "fabricated", "subtle_claim", "hand"
    )
    claim = {"claim": "real-time", "kind": "inference", "why": "w"}
    backend = JudgeBackend([claim])
    cfg = ge.JudgeConfig("m", "low")
    kwargs = dict(backend=backend, base_config=CONFIG, cache_path=tmp_path / "cache.jsonl")

    first = ge.judge_cases([case], sources, cfg, **kwargs)
    again = ge.judge_cases([case], sources, cfg, **kwargs)
    assert first["x1"].flagged and again["x1"].flagged
    assert len(backend.users) == 1  # the second run came from the cache


def test_combined_checker_flags_if_either_flags():
    sources = [SOURCE]
    rules_hit = gc.Case(
        "a", ["s"], "Built a tracker at 30 fps", "fabricated", "number_change", "code"
    )
    judge_hit = gc.Case("b", ["s"], "Built a robust tracker", "fabricated", "subtle_claim", "hand")
    verdicts = {
        "a": ge.Verdict(False, [], 0.0, 0),
        "b": ge.Verdict(True, [], 0.0, 0),
    }
    check = ge.combined_checker(verdicts)
    assert check(rules_hit, sources) and check(judge_hit, sources)


def test_hand_cases_reject_a_split_text(tmp_path):
    # An unquoted comma inside {...} turns part of the text into an extra key.
    path = tmp_path / "hand.yaml"
    path.write_text(
        "subtle_claim:\n  - {source: s01, adds: x, text: Built over 1,200 slides}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="quote the text"):
        gc.load_hand(path)
