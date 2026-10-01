import pytest
from pydantic import ValidationError

from factfit.schemas.job import JobDescription
from factfit.schemas.match import MatchResult
from factfit.schemas.tailored import TailoredCV

JD = {
    "title": "AI Engineer (LLM)",
    "company": "ABC",
    "seniority": "junior",
    "language": "en",
    "location": "Hanoi",
    "must_have": [{"id": "r1", "text": "Python", "category": "skill"}],
    "nice_to_have": [{"id": "r2", "text": "LangGraph", "category": "skill"}],
    "responsibilities": ["Build LLM agents"],
    "keywords": ["RAG", "agent"],
}


def test_job_description_valid():
    jd = JobDescription.model_validate(JD)
    assert [r.id for r in jd.all_requirements()] == ["r1", "r2"]


def test_job_description_rejects_repeated_requirement_ids():
    bad = {**JD, "nice_to_have": [{"id": "r1", "text": "LangGraph", "category": "skill"}]}
    with pytest.raises(ValidationError, match="repeated"):
        JobDescription.model_validate(bad)


def test_job_description_has_no_defaults_for_structured_outputs():
    # OpenAI Structured Outputs needs every field listed as required.
    schema = JobDescription.model_json_schema()
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False


def match(status, evidence):
    return {
        "score": 50,
        "requirements": [{"req_id": "r1", "status": status, "evidence": evidence, "note": ""}],
    }


def test_match_met_needs_evidence():
    with pytest.raises(ValidationError, match="cites no evidence"):
        MatchResult.model_validate(match("met", []))


def test_match_missing_cannot_cite_evidence():
    with pytest.raises(ValidationError, match="'missing' but cites"):
        MatchResult.model_validate(match("missing", ["b_a_01"]))


def test_match_gaps_derived_from_statuses():
    result = MatchResult.model_validate(match("missing", []))
    assert result.gaps == ["r1"]
    assert result.model_dump()["gaps"] == ["r1"]


def test_match_score_range():
    with pytest.raises(ValidationError):
        MatchResult.model_validate({**match("met", ["b1"]), "score": 120})


def cv(*groundings):
    return TailoredCV.model_validate(
        {
            "sections": [
                {
                    "type": "experience",
                    "ref": "exp_a",
                    "bullets": [
                        {"text": "x", "source_bullet_ids": ["b_a_01"], "grounding": g}
                        for g in groundings
                    ],
                }
            ],
            "skills": ["Python"],
        }
    )


def test_tailored_bullet_needs_a_source():
    with pytest.raises(ValidationError):
        TailoredCV.model_validate(
            {
                "sections": [
                    {
                        "type": "project",
                        "ref": "p",
                        "bullets": [{"text": "x", "source_bullet_ids": []}],
                    }
                ],
                "skills": [],
            }
        )


@pytest.mark.parametrize(
    "groundings, ready",
    [(("pass", "pass"), True), (("pass", "fail"), False), (("pass", "pending"), False)],
)
def test_export_blocked_until_every_bullet_passes(groundings, ready):
    assert cv(*groundings).ready_to_export() is ready
