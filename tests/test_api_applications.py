from pathlib import Path

import pytest
from test_api_runs import api, wait_for  # noqa: F401  (fixture)

from factfit.render.compile import TailoredRender


@pytest.fixture
def fake_render(monkeypatch):
    """Rendering writes small fake files; `result` sets what the ATS check reports."""
    result = {"issues": []}

    def render(profile, cv, out_dir: Path, keywords):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "cv.tex").write_text("\\resumeItem{x}", encoding="utf-8")
        (out_dir / "cv.pdf").write_bytes(b"%PDF-" + str(len(result["issues"])).encode())
        return TailoredRender(pages=1, issues=list(result["issues"]), warnings=[])

    monkeypatch.setattr("factfit.agent.graph.render_tailored", render)
    monkeypatch.setattr("factfit.api.app.render_tailored", render)
    return result


def finished_run(client, job_id):
    run_id = client.post(f"/jobs/{job_id}/tailor", json={}).json()["run_id"]
    wait_for(client, run_id, "waiting_review")
    client.post(f"/runs/{run_id}/review", json={"decisions": []})
    return run_id


def test_apply_freezes_the_cv_and_lists_the_application(api, fake_render):  # noqa: F811
    client, job_id = api
    run_id = finished_run(client, job_id)
    res = client.post(f"/runs/{run_id}/application", json={"channel": "LinkedIn"})
    assert res.status_code == 201, res.json()
    app = res.json()
    assert app["status"] == "applied" and app["cv_source"] == "generated"
    assert app["applied_at"] and app["channel"] == "LinkedIn"

    sent = client.get(app["cv_pdf_url"]).content
    client.post(f"/runs/{run_id}/render", json={})  # a rebuild after sending...
    assert client.get(app["cv_pdf_url"]).content == sent  # ...does not touch the sent CV
    assert client.get(app["cv_tex_url"]).text == "\\resumeItem{x}"

    listed = client.get("/applications").json()
    assert [a["id"] for a in listed] == [app["id"]]
    assert client.get("/applications", params={"status": "offer"}).json() == []


def test_a_second_application_needs_a_confirmation(api, fake_render):  # noqa: F811
    client, job_id = api
    run_id = finished_run(client, job_id)
    assert client.post(f"/runs/{run_id}/application", json={}).status_code == 201
    again = client.post(f"/runs/{run_id}/application", json={})
    assert again.status_code == 409 and again.json()["code"] == "duplicate"
    assert "CV Engineer" in again.json()["issues"][0]
    confirmed = client.post(f"/runs/{run_id}/application", json={"confirm_duplicate": True})
    assert confirmed.status_code == 201


def test_ats_problems_block_applied_but_not_saved(api, fake_render):  # noqa: F811
    client, job_id = api
    fake_render["issues"] = ["text is not readable"]
    run_id = finished_run(client, job_id)
    blocked = client.post(f"/runs/{run_id}/application", json={})
    assert blocked.status_code == 409 and blocked.json()["code"] == "ats"
    saved = client.post(f"/runs/{run_id}/application", json={"status": "saved"})
    assert saved.status_code == 201 and saved.json()["applied_at"] is None


def test_apply_before_the_review_is_refused(api, fake_render):  # noqa: F811
    client, job_id = api
    run_id = client.post(f"/jobs/{job_id}/tailor", json={}).json()["run_id"]
    wait_for(client, run_id, "waiting_review")
    assert client.post(f"/runs/{run_id}/application", json={}).status_code == 409


def test_manual_application_and_status_changes(api):  # noqa: F811
    client, _ = api
    body = {"company": "XYZ", "title": "ML Intern", "applied_on": "2026-09-30"}
    app = client.post("/applications", json=body).json()
    assert app["cv_source"] == "manual" and app["cv_pdf_url"] is None
    assert app["applied_at"].startswith("2026-09-30")

    moved = client.patch(f"/applications/{app['id']}", json={"status": "interview"}).json()
    assert moved["status"] == "interview"
    noted = client.patch(f"/applications/{app['id']}", json={"notes": "HR call"}).json()
    assert noted["notes"] == "HR call" and noted["status"] == "interview"
    assert client.patch("/applications/999", json={"notes": "x"}).status_code == 404
    assert client.post("/applications", json=body).json()["code"] == "duplicate"


def test_a_refused_manual_application_leaves_no_job_behind(api):  # noqa: F811
    client, _ = api
    body = {"company": "XYZ", "title": "ML Intern"}
    client.post("/applications", json=body)
    jobs_before = len({a["job_id"] for a in client.get("/applications").json()})
    assert client.post("/applications", json=body).status_code == 409
    again = client.post("/applications", json={**body, "confirm_duplicate": True}).json()
    jobs = {a["job_id"] for a in client.get("/applications").json()}
    assert len(jobs) == jobs_before + 1 and again["job_id"] == max(jobs)
