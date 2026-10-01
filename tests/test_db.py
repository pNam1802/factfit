import pytest
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from factfit.db import (
    Application,
    ApplicationStatus,
    Company,
    CVVersion,
    Job,
    LLMCall,
    StatusEvent,
    init_db,
    make_engine,
)


@pytest.fixture
def session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as s:
        yield s


@pytest.fixture
def sent_cv(session):
    """A company, a job, a CV version and an application that used it."""
    company = Company(name="ABC")
    session.add(company)
    session.commit()
    job = Job(company_id=company.id, title="AI Engineer", raw_text="...", text_hash="h1")
    session.add(job)
    session.commit()
    version = CVVersion(
        job_id=job.id,
        profile_version="p1",
        tailored_json={"sections": [], "skills": []},
        tex="\\documentclass{article}",
        prompt_version="rewrite/v1",
        model="test-model",
    )
    session.add(version)
    session.commit()
    application = Application(job_id=job.id, cv_version_id=version.id)
    session.add(application)
    session.commit()
    return version, application


def test_tables_round_trip(session, sent_cv):
    version, application = sent_cv
    loaded = session.get(CVVersion, version.id)
    assert loaded.tailored_json == {"sections": [], "skills": []}
    assert application.status == ApplicationStatus.saved


def test_cv_version_cannot_be_updated(session, sent_cv):
    version, _ = sent_cv
    version.tex = "edited after sending"
    session.add(version)
    with pytest.raises(IntegrityError, match="immutable"):
        session.commit()


def test_cv_linked_to_application_cannot_be_deleted(session, sent_cv):
    version, _ = sent_cv
    session.delete(version)
    with pytest.raises(IntegrityError):
        session.commit()


def test_status_event_records_change(session, sent_cv):
    _, application = sent_cv
    session.add(
        StatusEvent(
            application_id=application.id,
            from_status=ApplicationStatus.saved,
            to_status=ApplicationStatus.applied,
        )
    )
    session.commit()
    event = session.exec(select(StatusEvent)).one()
    assert event.to_status == ApplicationStatus.applied
    assert event.at is not None


def test_manual_application_needs_no_cv_version(session, sent_cv):
    _, linked = sent_cv
    manual = Application(job_id=linked.job_id, cv_source="manual")
    session.add(manual)
    session.commit()
    assert manual.cv_version_id is None


def test_llm_call_logged(session):
    session.add(
        LLMCall(
            node="parse_jd", model="m", prompt_version="parse_jd/v1", tokens_in=900, cost_usd=0.001
        )
    )
    session.commit()
    assert session.exec(select(LLMCall)).one().tokens_in == 900
