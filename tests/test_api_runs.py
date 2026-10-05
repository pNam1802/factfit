import shutil
import time

import pytest
import yaml
from fastapi.testclient import TestClient
from sqlmodel import Session
from test_graph import CONFIG, JD, PROFILE, FakeBackend

from factfit.agent.nodes.parse_jd import jd_hash
from factfit.api.app import create_app
from factfit.db import Job, make_engine
from factfit.llm import LLMClient


@pytest.fixture
def api(tmp_path):
    profile = tmp_path / "profile.yaml"
    profile.write_text(yaml.safe_dump(PROFILE.model_dump(mode="json")), encoding="utf-8")
    engine = make_engine(f"sqlite:///{tmp_path / 'app.db'}")
    app = create_app(
        engine=engine,
        client_factory=lambda engine: LLMClient(FakeBackend(), CONFIG),
        profile_path=str(profile),
        checkpoint_path=tmp_path / "checkpoints.db",
        output_dir=tmp_path / "output",
    )
    raw = "CV Engineer\nObject detection"
    with Session(engine) as s:
        job = Job(title="CV Engineer", raw_text=raw, text_hash=jd_hash(raw),
                  parsed_json=JD, parse_prompt_version="parse_jd/v2")  # fmt: skip
        s.add(job)
        s.commit()
        job_id = job.id
    return TestClient(app), job_id


def wait_for(client, run_id, status, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/runs/{run_id}").json()
        if body["status"] in (status, "error"):
            return body
        time.sleep(0.1)
    raise AssertionError(f"run {run_id} never reached {status}")


def test_start_review_and_finish_over_http(api):
    client, job_id = api
    started = client.post(f"/jobs/{job_id}/tailor", json={})
    assert started.status_code == 202
    run_id = started.json()["run_id"]

    waiting = wait_for(client, run_id, "waiting_review")
    assert waiting["status"] == "waiting_review", waiting.get("error")
    assert {d["source_id"] for d in waiting["drafts"]} == {"b1", "b2"}

    events = client.get(f"/runs/{run_id}/events").text
    for node in ("parse", "match", "select", "rewrite", "check", "review", "__end__"):
        assert f'"node": "{node}"' in events

    done = client.post(f"/runs/{run_id}/review", json={"decisions": []}).json()
    assert done["status"] == "done"
    assert done["cv"]["sections"][0]["ref"] == "exp_a"

    # A finished run cannot be reviewed again.
    assert client.post(f"/runs/{run_id}/review", json={"decisions": []}).status_code == 409


def test_failing_edit_comes_back_with_reasons(api):
    client, job_id = api
    run_id = client.post(f"/jobs/{job_id}/tailor", json={}).json()["run_id"]
    wait_for(client, run_id, "waiting_review")
    edit = {"source_id": "b2", "action": "edit", "text": "Contributed to a robust FastAPI service"}
    body = client.post(f"/runs/{run_id}/review", json={"decisions": [edit]}).json()
    assert body["status"] == "waiting_review"
    assert "robust" in body["review_errors"]["b2"][0]


def test_unknown_run_and_unknown_job(api):
    client, _ = api
    assert client.get("/runs/nope").json()["status"] == "not_found"
    assert client.post("/jobs/999/tailor", json={}).status_code == 404


@pytest.mark.skipif(shutil.which("tectonic") is None, reason="tectonic not installed")
def test_render_after_review_gives_a_checked_pdf(api):
    client, job_id = api
    run_id = client.post(f"/jobs/{job_id}/tailor", json={}).json()["run_id"]
    wait_for(client, run_id, "waiting_review")
    # Rendering before the review is finished is refused.
    assert client.post(f"/runs/{run_id}/render", json={}).status_code == 409

    client.post(f"/runs/{run_id}/review", json={"decisions": []})
    out = client.post(f"/runs/{run_id}/render", json={}).json()
    assert out["pages"] == 1 and out["ats_ok"], out["issues"]
    pdf = client.get(out["pdf_url"])
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    assert rb"\resumeItem{" in client.get(out["tex_url"]).content


def test_run_files_reject_odd_run_ids(api):
    client, _ = api
    assert client.get("/runs/..%2F..%2Fdata/cv.pdf").status_code == 404
    assert client.get("/runs/abc/cv.tex").status_code == 404
