"""Job applications and the exact CV sent with each one (PRD F7, tracker).

A tailored CV is frozen into a `cv_versions` row when an application is saved: the .tex goes
into the row and the PDF is copied next to the other versions, named by its content hash.
The row cannot be updated (a database trigger), and the file is never overwritten, so the CV
linked to an application stays what was sent even after rebuilds or profile edits.

Every status change is a `status_events` row, so response times can be measured later.
"""

import hashlib
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlmodel import Session, col, select

from factfit.db import Application, ApplicationStatus, Company, CVVersion, Job, StatusEvent

DUPLICATE_WINDOW_DAYS = 90  # many companies ignore a second application inside this window


def now_utc() -> datetime:
    return datetime.now(UTC)


def _aware(t: datetime) -> datetime:
    # SQLite returns naive datetimes; they were stored as UTC.
    return t if t.tzinfo else t.replace(tzinfo=UTC)


# --- frozen CVs ------------------------------------------------------------------------------


def freeze_cv(
    session: Session,
    *,
    job_id: int,
    profile_version: str,
    cv: dict,
    tex_path: Path,
    pdf_path: Path,
    versions_dir: Path,
    prompt_version: str,
    model: str,
) -> CVVersion:
    """Store the rendered CV as an immutable version; the same files give the same version."""
    tex = tex_path.read_text(encoding="utf-8")
    pdf = pdf_path.read_bytes()
    digest = hashlib.sha256(tex.encode() + pdf).hexdigest()[:16]
    dest = versions_dir / f"{digest}.pdf"

    existing = session.exec(
        select(CVVersion).where(CVVersion.job_id == job_id, CVVersion.pdf_path == str(dest))
    ).first()
    if existing:
        return existing

    versions_dir.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        shutil.copyfile(pdf_path, dest)
    version = CVVersion(
        job_id=job_id,
        profile_version=profile_version,
        tailored_json=cv,
        tex=tex,
        pdf_path=str(dest),
        prompt_version=prompt_version,
        model=model,
    )
    session.add(version)
    session.commit()
    session.refresh(version)
    return version


# --- applications ----------------------------------------------------------------------------


def possible_duplicates(
    session: Session,
    *,
    company_id: int | None,
    job_id: int | None = None,
    now: datetime | None = None,
) -> list[tuple[Application, Job]]:
    """Applications for this same job (any time), or sent to this company in the last 90 days.

    Takes ids rather than a Job, so it can run before a manual application's job is created.
    """
    now = now or now_utc()
    since = now - timedelta(days=DUPLICATE_WINDOW_DAYS)
    rows = session.exec(select(Application, Job).join(Job)).all()
    out = []
    for application, other in rows:
        same_job = job_id is not None and other.id == job_id
        same_company = company_id is not None and other.company_id == company_id
        recent = application.applied_at is not None and _aware(application.applied_at) >= since
        if same_job or (same_company and recent):
            out.append((application, other))
    return out


def describe(session: Session, application: Application, job: Job) -> str:
    company = session.get(Company, job.company_id) if job.company_id else None
    when = (
        f"applied {_aware(application.applied_at):%Y-%m-%d}"
        if application.applied_at
        else "saved, not sent"
    )
    name = company.name if company else "unknown company"
    return f"{name}: {job.title} ({when}, {application.status})"


def create_application(
    session: Session,
    *,
    job_id: int,
    status: ApplicationStatus,
    cv_version_id: int | None = None,
    channel: str | None = None,
    notes: str = "",
    applied_at: datetime | None = None,
    now: datetime | None = None,
) -> Application:
    """Create an application and its first status event. A CV-less one is a manual application."""
    now = now or now_utc()
    if status != ApplicationStatus.saved and applied_at is None:
        applied_at = now
    application = Application(
        job_id=job_id,
        cv_version_id=cv_version_id,
        cv_source="generated" if cv_version_id else "manual",
        status=status,
        applied_at=applied_at,
        channel=channel,
        notes=notes,
    )
    session.add(application)
    session.flush()
    session.add(
        StatusEvent(application_id=application.id, from_status=None, to_status=status, at=now)
    )
    session.commit()
    session.refresh(application)
    return application


def change_status(
    session: Session,
    application: Application,
    to: ApplicationStatus,
    *,
    now: datetime | None = None,
) -> Application:
    """Move an application to another status, keeping the history. Any move is allowed, so a
    mistake can be undone; the event log still shows it."""
    if application.status == to:
        return application
    now = now or now_utc()
    session.add(
        StatusEvent(
            application_id=application.id, from_status=application.status, to_status=to, at=now
        )
    )
    if to != ApplicationStatus.saved and application.applied_at is None:
        application.applied_at = now  # first moved out of "saved": that is when it was sent
    application.status = to
    session.add(application)
    session.commit()
    session.refresh(application)
    return application


@dataclass
class ApplicationRow:
    application: Application
    job: Job
    company: str | None
    last_change: datetime


def list_applications(
    session: Session, status: ApplicationStatus | None = None
) -> list[ApplicationRow]:
    """Applications with their job and company, the most recently changed first."""
    query = select(Application, Job).join(Job)
    if status:
        query = query.where(Application.status == status)
    rows = []
    for application, job in session.exec(query).all():
        company = session.get(Company, job.company_id) if job.company_id else None
        last = session.exec(
            select(StatusEvent.at)
            .where(StatusEvent.application_id == application.id)
            .order_by(col(StatusEvent.at).desc())
        ).first()
        rows.append(
            ApplicationRow(application, job, company.name if company else None, _aware(last))
        )
    return sorted(rows, key=lambda r: r.last_change, reverse=True)


def manual_job(
    session: Session,
    *,
    company: str,
    title: str,
    url: str | None = None,
    jd_text: str | None = None,
) -> Job:
    """A job for an application made without factfit (the manual baseline, PRD §10).

    Not parsed: it costs nothing and is only used by the tracker. Pasting the same JD text
    again reuses the job, as parse_jd does.
    """
    from factfit.agent.nodes.parse_jd import get_or_create_company, jd_hash, normalize_jd

    text = normalize_jd(jd_text or "")
    text_hash = jd_hash(text) if text else f"manual:{uuid.uuid4().hex}"
    job = job_with_text(session, jd_text)
    if job is None:
        job = Job(title=title, raw_text=jd_text or "", text_hash=text_hash, source="manual")
    job.url = url or job.url
    job.company_id = get_or_create_company(session, company).id
    session.add(job)
    session.commit()
    session.refresh(job)
    return job


def job_with_text(session: Session, jd_text: str | None) -> Job | None:
    """The job already stored for this JD text, if any."""
    from factfit.agent.nodes.parse_jd import jd_hash, normalize_jd

    text = normalize_jd(jd_text or "")
    if not text:
        return None
    return session.exec(select(Job).where(Job.text_hash == jd_hash(text))).first()
