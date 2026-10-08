from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.exc import DBAPIError
from sqlmodel import Session, select

from factfit.applications import (
    change_status,
    create_application,
    freeze_cv,
    list_applications,
    manual_job,
    possible_duplicates,
)
from factfit.db import ApplicationStatus, CVVersion, StatusEvent, init_db, make_engine

NOW = datetime(2026, 10, 8, 9, 0, tzinfo=UTC)
S = ApplicationStatus


@pytest.fixture
def session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'app.db'}")
    init_db(engine)
    with Session(engine) as s:
        yield s


def rendered(tmp_path, tex="\\resumeItem{a}", pdf=b"%PDF-1"):
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    (run / "cv.tex").write_text(tex, encoding="utf-8")
    (run / "cv.pdf").write_bytes(pdf)
    return run


def freeze(session, tmp_path, job_id, run):
    return freeze_cv(
        session, job_id=job_id, profile_version="p1", cv={"sections": [], "skills": []},
        tex_path=run / "cv.tex", pdf_path=run / "cv.pdf", versions_dir=tmp_path / "versions",
        prompt_version="rewrite/v1", model="gpt-5",
    )  # fmt: skip


def test_a_frozen_cv_survives_a_rebuild(session, tmp_path):
    job = manual_job(session, company="ABC", title="AI Engineer")
    run = rendered(tmp_path)
    v1 = freeze(session, tmp_path, job.id, run)
    assert freeze(session, tmp_path, job.id, run).id == v1.id  # same files, same version

    (run / "cv.pdf").write_bytes(b"%PDF-2")  # rebuilt after the application was saved
    v2 = freeze(session, tmp_path, job.id, run)
    assert v2.id != v1.id
    with open(v1.pdf_path, "rb") as f:
        assert f.read() == b"%PDF-1"  # the sent file is untouched

    v1.tex = "changed"
    session.add(v1)
    with pytest.raises(DBAPIError, match="immutable"):
        session.commit()


def test_applied_sets_the_date_and_saved_does_not(session):
    job = manual_job(session, company="ABC", title="AI Engineer")
    sent = create_application(session, job_id=job.id, status=S.applied, now=NOW)
    kept = create_application(session, job_id=job.id, status=S.saved, now=NOW)
    assert sent.cv_source == "manual" and sent.applied_at.replace(tzinfo=UTC) == NOW
    assert kept.applied_at is None


def test_status_changes_keep_their_history(session):
    job = manual_job(session, company="ABC", title="AI Engineer")
    app = create_application(session, job_id=job.id, status=S.saved, now=NOW)
    later = NOW + timedelta(days=3)
    change_status(session, app, S.applied, now=later)
    change_status(session, app, S.applied, now=later)  # same status: nothing recorded
    change_status(session, app, S.interview, now=later + timedelta(days=7))

    events = session.exec(select(StatusEvent).order_by(StatusEvent.at)).all()
    assert [(e.from_status, e.to_status) for e in events] == [
        (None, S.saved),
        (S.saved, S.applied),
        (S.applied, S.interview),
    ]
    assert app.applied_at.replace(tzinfo=UTC) == later  # sent when it left "saved"
    row = list_applications(session)[0]
    assert row.company == "ABC" and row.last_change == later + timedelta(days=7)


def test_duplicates_same_job_any_time_or_same_company_within_90_days(session):
    job = manual_job(session, company="ABC", title="AI Engineer")
    other_role = manual_job(session, company="abc", title="Data Engineer")  # same company
    elsewhere = manual_job(session, company="XYZ", title="AI Engineer")

    assert possible_duplicates(session, job, now=NOW) == []
    create_application(
        session, job_id=other_role.id, status=S.applied, now=NOW - timedelta(days=100)
    )
    assert possible_duplicates(session, job, now=NOW) == []  # outside the window
    create_application(
        session, job_id=other_role.id, status=S.applied, now=NOW - timedelta(days=30)
    )
    assert len(possible_duplicates(session, job, now=NOW)) == 1
    create_application(session, job_id=job.id, status=S.saved, now=NOW - timedelta(days=400))
    assert len(possible_duplicates(session, job, now=NOW)) == 2  # same job: always
    assert possible_duplicates(session, elsewhere, now=NOW) == []


def test_manual_job_reuses_the_same_pasted_text(session):
    a = manual_job(session, company="ABC", title="AI", jd_text="We hire  AI engineers")
    b = manual_job(session, company="ABC", title="AI", jd_text="We hire AI engineers")
    c = manual_job(session, company="ABC", title="AI")
    assert a.id == b.id != c.id
    assert session.exec(select(CVVersion)).all() == []
