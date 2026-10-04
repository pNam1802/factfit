import json
from pathlib import Path

import pytest
import sqlalchemy as sa
from sqlmodel import Session

from factfit.agent.nodes.match import applicable, check_judgement, match_job, profile_items
from factfit.agent.scoring import Weights, compute_score
from factfit.db import Job, init_db, make_engine
from factfit.llm import LLMClient
from factfit.llm.backends import RawOutput
from factfit.llm.config import LLMConfig
from factfit.profile import load_profile
from factfit.schemas.job import JobDescription
from factfit.schemas.match import MatchJudgement, MatchResult, RequirementMatch

EXAMPLE = Path(__file__).parent.parent / "data" / "profile.example.yaml"

WEIGHTS = Weights.model_validate(
    {
        "bucket": {"must_have": 1.0, "nice_to_have": 0.5},
        "category": {
            "skill": 1.0,
            "experience": 1.0,
            "education": 1.0,
            "language": 0.5,
            "soft_skill": 0.25,
            "other": 0.25,
        },
        "credit": {"met": 1.0, "partial": 0.5, "missing": 0.0},
    }
)


def req(id_, category="skill", level=None):
    return {"id": id_, "text": id_, "category": category, "any_of": [], "level": level}


JD = JobDescription.model_validate(
    {
        "title": "AI Engineer",
        "company": None,
        "seniority": "mid",
        "language": "en",
        "location": None,
        "must_have": [req("r1"), req("r2"), req("r3", "other"), req("r4", "experience", "senior")],
        "nice_to_have": [req("r5")],
        "responsibilities": [],
        "keywords": [],
    }
)


def judged(**statuses):
    return [
        RequirementMatch(
            req_id=rid,
            status=s,
            evidence=["b_acme_01"] if s in ("met", "partial") else [],
            note="",
        )
        for rid, s in statuses.items()
    ]


def test_score_weights_must_over_nice_and_skips_unverifiable():
    # weights: r1 1.0, r2 1.0, r5 0.5; r3 is unverifiable so left out
    result = judged(r1="met", r2="partial", r3="unverifiable", r5="missing")
    assert compute_score(JD, result, WEIGHTS) == round(100 * (1.0 + 0.5) / 2.5)


def test_score_is_zero_when_nothing_can_be_scored():
    assert compute_score(JD, judged(r3="unverifiable"), WEIGHTS) == 0


def test_level_filter_keeps_shared_and_own_level():
    keep, excluded = applicable(JD, "mid")
    assert [r.id for r in keep] == ["r1", "r2", "r3", "r5"]
    assert excluded == ["r4"]
    assert len(applicable(JD, "senior")[0]) == 5


def test_check_drops_unknown_evidence_and_fills_gaps():
    judgement = MatchJudgement(
        requirements=[
            RequirementMatch(req_id="r1", status="met", evidence=["b_fake"], note="shown"),
            RequirementMatch(
                req_id="r2", status="met", evidence=["b_ok", "b_fake", "b_ok"], note="shown"
            ),
            RequirementMatch(req_id="r2", status="missing", evidence=[], note="duplicate"),
            RequirementMatch(req_id="r3", status="unverifiable", evidence=["b_ok"], note="n/a"),
            RequirementMatch(req_id="r99", status="met", evidence=["b_ok"], note="unknown req"),
        ]
    )
    reqs = [r for r in JD.must_have if r.id != "r4"]
    out = {r.req_id: r for r in check_judgement(judgement, reqs, valid_ids={"b_ok"})}

    assert out["r1"].status == "missing" and "not found" in out["r1"].note
    assert out["r2"].status == "met" and out["r2"].evidence == ["b_ok"]
    assert out["r3"].evidence == []
    assert set(out) == {"r1", "r2", "r3"}


def test_profile_items_include_education_and_bullets():
    items = {i.id: i for i in profile_items(load_profile(EXAMPLE))}
    assert "edu_uni" in items and "B.Sc." in items["edu_uni"].text
    assert items["b_acme_01"].where.startswith("AI Engineer Intern at Acme Retail Tech")


def test_match_result_survives_a_round_trip_through_json():
    result = MatchResult(score=50, requirements=judged(r1="missing"))
    stored = result.model_dump(mode="json")
    assert stored["gaps"] == ["r1"]
    assert MatchResult.model_validate(stored) == result


class MatchBackend:
    def __init__(self):
        self.calls = 0

    def complete(self, *, node, model, system, user, schema, options):
        self.calls += 1
        answer = {
            "requirements": [
                {"req_id": "r1", "status": "met", "evidence": ["b_acme_01"], "note": "YOLOv8"},
                {"req_id": "r2", "status": "missing", "evidence": [], "note": "not shown"},
                {"req_id": "r3", "status": "unverifiable", "evidence": [], "note": "schedule"},
                {"req_id": "r5", "status": "missing", "evidence": [], "note": "not shown"},
            ]
        }
        return RawOutput(text=json.dumps(answer))


def test_match_job_caches_judgement_but_recomputes_score(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    config = LLMConfig.model_validate(
        {"models": {"small": "m"}, "nodes": {"match": {"model": "small", "prompt": "v1"}}}
    )
    backend = MatchBackend()
    client = LLMClient(backend, config)
    profile = load_profile(EXAMPLE)

    with Session(engine) as session:
        job = Job(title="t", raw_text="x", text_hash="h", parsed_json=JD.model_dump(mode="json"))
        session.add(job)
        session.commit()

        first = match_job(
            job, profile, profile_version="p1", client=client, session=session,
            level="mid", weights=WEIGHTS,
        )  # fmt: skip
        must_only = WEIGHTS.model_copy(update={"bucket": {"must_have": 1.0, "nice_to_have": 0.0}})
        again = match_job(
            job, profile, profile_version="p1", client=client, session=session,
            level="mid", weights=must_only,
        )  # fmt: skip

    assert not first.cached and again.cached and backend.calls == 1
    assert first.result.score == round(100 * 1 / 2.5)  # r1 met of r1, r2, r5(0.5)
    assert again.result.score == 50  # same judgement, nice_to_have no longer counts
    assert first.result.excluded == ["r4"]


def test_init_db_adds_new_nullable_columns(tmp_path):
    url = f"sqlite:///{tmp_path / 'old.db'}"
    engine = make_engine(url)
    with engine.begin() as conn:  # a jobs table from before parse_prompt_version existed
        conn.execute(
            sa.text(
                "CREATE TABLE jobs (id INTEGER PRIMARY KEY, company_id INTEGER, title VARCHAR,"
                " url VARCHAR, source VARCHAR, raw_text VARCHAR, parsed_json JSON,"
                " text_hash VARCHAR, created_at DATETIME)"
            )
        )
    init_db(engine)
    columns = {c["name"] for c in sa.inspect(engine).get_columns("jobs")}
    assert "parse_prompt_version" in columns


@pytest.mark.parametrize("status", ["missing", "unverifiable"])
def test_no_evidence_allowed_without_a_claim(status):
    with pytest.raises(ValueError, match="cites evidence"):
        MatchResult(
            score=0,
            requirements=[RequirementMatch(req_id="r1", status=status, evidence=["b"], note="")],
        )
