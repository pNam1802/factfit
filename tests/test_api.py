import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from factfit.api.app import create_app
from factfit.db import make_engine
from factfit.llm import LLMClient, LLMError
from factfit.llm.backends import RawOutput
from factfit.llm.config import LLMConfig

EXAMPLE = str(Path(__file__).parent.parent / "data" / "profile.example.yaml")

CONFIG = LLMConfig.model_validate(
    {
        "models": {"small": "m"},
        "nodes": {
            "parse_jd": {"model": "small", "prompt": "v2"},
            "match": {"model": "small", "prompt": "v1"},
        },
    }
)


def req(id_, text, level=None):
    return {"id": id_, "text": text, "category": "skill", "any_of": [], "level": level}


ANSWERS = {
    "JobDescription": {
        "title": "AI Engineer",
        "company": "ABC",
        "seniority": "mid",
        "language": "en",
        "location": "Hanoi",
        "must_have": [req("r1", "Python"), req("r2", "Docker"), req("r3", "Mentoring", "senior")],
        "nice_to_have": [],
        "responsibilities": [],
        "keywords": ["Python"],
    },
    "MatchJudgement": {
        "requirements": [
            {"req_id": "r1", "status": "met", "evidence": ["b_acme_01"], "note": "Python"},
            {"req_id": "r2", "status": "missing", "evidence": [], "note": "no Docker"},
        ]
    },
}


class FakeBackend:
    def __init__(self):
        self.calls: list[str] = []

    def complete(self, *, node, model, system, user, schema, options):
        self.calls.append(schema.__name__)
        return RawOutput(text=json.dumps(ANSWERS[schema.__name__]))


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def api(tmp_path, backend):
    app = create_app(
        engine=make_engine(f"sqlite:///{tmp_path / 'test.db'}"),
        client_factory=lambda engine: LLMClient(backend, CONFIG),
        profile_path=EXAMPLE,
    )
    return TestClient(app)


def test_paste_parse_then_match(api, backend):
    job = api.post("/jobs", json={"text": "AI Engineer\nPython, Docker"}).json()
    assert job["title"] == "AI Engineer" and job["company"] == "ABC" and not job["cached"]

    match = api.post(f"/jobs/{job['id']}/match", json={}).json()
    by_id = {r["id"]: r for r in match["requirements"]}
    assert match["level"] == "mid" and match["excluded"] == ["r3"]
    assert by_id["r1"]["status"] == "met"
    assert by_id["r1"]["evidence"][0]["text"].startswith("Built a multi-camera")
    assert by_id["r2"]["status"] == "missing" and by_id["r2"]["evidence"] == []
    assert match["score"] == 50
    assert backend.calls == ["JobDescription", "MatchJudgement"]


def test_second_paste_and_match_are_cached(api, backend):
    job = api.post("/jobs", json={"text": "AI Engineer\nPython, Docker"}).json()
    api.post(f"/jobs/{job['id']}/match", json={})
    again = api.post("/jobs", json={"text": "AI Engineer\n  Python,  Docker"}).json()
    match = api.post(f"/jobs/{again['id']}/match", json={}).json()
    assert again["cached"] and match["cached"]
    assert len(backend.calls) == 2


def test_get_job_and_unknown_job(api):
    job = api.post("/jobs", json={"text": "AI Engineer"}).json()
    assert api.get(f"/jobs/{job['id']}").json()["jd"]["must_have"][0]["text"] == "Python"
    assert api.get("/jobs/999").status_code == 404
    assert api.post("/jobs/999/match", json={}).status_code == 404


def test_invalid_profile_explains_how_to_fix(tmp_path, backend):
    bad = tmp_path / "profile.yaml"
    bad.write_text("basics: {name: X, email: x@y.z}\nskills:\n  - {name: Go, evidence: [b_none]}\n")
    api = TestClient(
        create_app(
            engine=make_engine(f"sqlite:///{tmp_path / 'test.db'}"),
            client_factory=lambda engine: LLMClient(backend, CONFIG),
            profile_path=str(bad),
        )
    )
    job = api.post("/jobs", json={"text": "AI Engineer"}).json()
    response = api.post(f"/jobs/{job['id']}/match", json={})
    assert response.status_code == 422
    assert "validate-profile" in response.json()["detail"]
    assert "b_none" in response.json()["issues"][0]


def test_missing_api_key_is_a_clear_503(tmp_path):
    def no_key(engine):
        raise LLMError("OPENAI_API_KEY is not set")

    api = TestClient(
        create_app(
            engine=make_engine(f"sqlite:///{tmp_path / 'test.db'}"),
            client_factory=no_key,
            profile_path=EXAMPLE,
        )
    )
    assert api.get("/health").status_code == 200
    response = api.post("/jobs", json={"text": "AI Engineer"})
    assert response.status_code == 503 and "OPENAI_API_KEY" in response.json()["detail"]


def test_empty_text_is_rejected(api):
    assert api.post("/jobs", json={"text": ""}).status_code == 422


def test_ui_openapi_schema_is_up_to_date(tmp_path):
    # The UI's TypeScript types are generated from ui/openapi.json; if the API changes,
    # regenerate: uv run factfit export-openapi && cd ui && npm run gen:api
    from factfit.api.app import export_openapi

    fresh = tmp_path / "openapi.json"
    export_openapi(fresh)
    committed = Path(__file__).parent.parent / "ui" / "openapi.json"
    assert json.loads(fresh.read_text(encoding="utf-8")) == json.loads(
        committed.read_text(encoding="utf-8")
    ), "ui/openapi.json is stale"
