"""HTTP API used by the UI. A thin layer over the same functions the CLI calls.

Run with `uv run factfit dev` (API + UI), or the API alone with
`uv run uvicorn factfit.api.app:create_app --factory --reload`.
Interactive docs: http://localhost:8000/docs
"""

import json
import os
import re
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from langgraph.types import Command
from sqlalchemy import Engine
from sqlmodel import Session

from factfit import applications as tracker
from factfit.agent.graph import (
    DEFAULT_CHECKPOINTS,
    build_graph,
    initial_state,
    open_checkpointer,
    run_config,
    run_status,
)
from factfit.agent.nodes.match import match_job, profile_items
from factfit.agent.nodes.parse_jd import parse_jd
from factfit.api.schemas import (
    ApplicationOut,
    ApplicationUpdate,
    ApplyFromRun,
    CreateJob,
    ErrorOut,
    Evidence,
    JobOut,
    ManualApplication,
    MatchOut,
    MatchRequest,
    RenderOut,
    RenderRequest,
    RequirementOut,
    ReviewRequest,
    RunOut,
    RunStatusOut,
    TailorRequest,
)
from factfit.db import (
    Application,
    ApplicationStatus,
    Company,
    CVVersion,
    Job,
    init_db,
    make_engine,
)
from factfit.llm import LLMClient, LLMError, make_client
from factfit.llm.config import load_llm_config
from factfit.profile import ProfileError, load_profile, profile_version
from factfit.render.compile import RenderError, render_tailored
from factfit.schemas.job import JobDescription
from factfit.schemas.profile import Profile
from factfit.schemas.tailored import TailoredCV

DEFAULT_PROFILE = "data/profile.yaml"


class NeedsConfirmation(Exception):
    """409: the request is fine but needs a yes first; send it again with the confirm flag."""

    def __init__(self, code: str, detail: str, issues: list[str]):
        self.code, self.detail, self.issues = code, detail, issues


class ProfileInvalid(Exception):
    def __init__(self, path: str, error: ProfileError):
        self.path, self.error = path, error


def create_app(
    engine: Engine | None = None,
    client_factory: Callable[..., LLMClient] = make_client,
    profile_path: str | None = None,
    checkpoint_path: str | Path | None = None,
    output_dir: str | Path = "output",
) -> FastAPI:
    load_dotenv()
    engine = engine or make_engine()
    init_db(engine)
    profile_file = profile_path or os.environ.get("FACTFIT_PROFILE", DEFAULT_PROFILE)
    checkpoint_file = checkpoint_path or DEFAULT_CHECKPOINTS
    output_root = Path(output_dir)

    app = FastAPI(title="factfit", version="0.1.0", description="Tailor CVs without inventing.")
    clients: dict[str, LLMClient] = {}

    # --- dependencies ------------------------------------------------------------------------

    def get_session() -> Iterator[Session]:
        with Session(engine) as session:
            yield session

    def get_client() -> LLMClient:
        # Created on first use, so the API starts (and shows its docs) even without a key.
        if "client" not in clients:
            try:
                clients["client"] = client_factory(engine=engine)
            except LLMError as e:
                raise HTTPException(status_code=503, detail=str(e)) from e
        return clients["client"]

    SessionDep = Annotated[Session, Depends(get_session)]
    ClientDep = Annotated[LLMClient, Depends(get_client)]

    # --- errors ------------------------------------------------------------------------------

    @app.exception_handler(LLMError)
    def _llm_error(_: Request, e: LLMError) -> JSONResponse:
        return JSONResponse(status_code=502, content=ErrorOut(detail=str(e)).model_dump())

    @app.exception_handler(NeedsConfirmation)
    def _needs_confirmation(_: Request, e: NeedsConfirmation) -> JSONResponse:
        body = ErrorOut(detail=e.detail, issues=e.issues, code=e.code)
        return JSONResponse(status_code=409, content=body.model_dump())

    @app.exception_handler(ProfileInvalid)
    def _profile_error(_: Request, e: ProfileInvalid) -> JSONResponse:
        body = ErrorOut(
            detail=f"{e.path} is not valid yet. Run: uv run factfit validate-profile {e.path}",
            issues=[i.format(e.path) for i in e.error.issues],
        )
        return JSONResponse(status_code=422, content=body.model_dump())

    # --- routes ------------------------------------------------------------------------------

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "profile": profile_file}

    @app.post("/jobs", response_model=JobOut, responses={502: {"model": ErrorOut}})
    def create_job(
        body: CreateJob,
        session: SessionDep,
        client: ClientDep,
    ) -> JobOut:
        """Paste a job description: it is parsed into requirements (cached per text)."""
        try:
            result = parse_jd(
                body.text, client=client, session=session, company=body.company, url=body.url
            )
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        return _job_out(session, result.job, result.jd, cached=result.cached)

    @app.get("/jobs/{job_id}", response_model=JobOut, responses={404: {"model": ErrorOut}})
    def get_job(job_id: int, session: SessionDep) -> JobOut:
        job = _get_parsed_job(session, job_id)
        return _job_out(session, job, JobDescription.model_validate(job.parsed_json), cached=True)

    @app.post(
        "/jobs/{job_id}/match",
        response_model=MatchOut,
        responses={404: {"model": ErrorOut}, 422: {"model": ErrorOut}, 502: {"model": ErrorOut}},
    )
    def match(
        job_id: int,
        session: SessionDep,
        client: ClientDep,
        body: MatchRequest | None = None,
    ) -> MatchOut:
        """Check the profile against a parsed job: status and evidence per requirement."""
        job = _get_parsed_job(session, job_id)
        try:
            profile = load_profile(profile_file)
        except ProfileError as e:
            raise ProfileInvalid(profile_file, e) from e
        except FileNotFoundError as e:
            raise HTTPException(status_code=422, detail=f"{profile_file}: file not found") from e

        jd = JobDescription.model_validate(job.parsed_json)
        level = (body.level if body else None) or (
            jd.seniority if jd.seniority != "unknown" else None
        )
        outcome = match_job(
            job,
            profile,
            profile_version=profile_version(profile_file),
            client=client,
            session=session,
            level=level,
        )

        items = {i.id: i for i in profile_items(profile)}
        reqs = {r.id: (r, "must") for r in jd.must_have}
        reqs |= {r.id: (r, "nice") for r in jd.nice_to_have}
        result = outcome.result
        return MatchOut(
            job_id=job.id,
            score=result.score,
            level=result.level,
            profile_version=outcome.match.profile_version,
            cached=outcome.cached,
            excluded=result.excluded,
            requirements=[
                RequirementOut(
                    id=m.req_id,
                    text=reqs[m.req_id][0].text,
                    bucket=reqs[m.req_id][1],
                    category=reqs[m.req_id][0].category,
                    any_of=reqs[m.req_id][0].any_of,
                    status=m.status,
                    note=m.note,
                    evidence=[
                        Evidence(id=e, where=items[e].where, text=items[e].text)
                        for e in m.evidence
                        if e in items
                    ],
                )
                for m in result.requirements
            ],
        )

    # --- tailoring runs (the LangGraph agent) ------------------------------------------------
    #
    # A run takes 30-90 s, so it runs in a background thread: POST returns a run_id at once,
    # the UI follows progress on /runs/{id}/events (SSE) and reads drafts from /runs/{id}.
    # Progress events live in memory; the run itself is checkpointed to SQLite, so after a
    # restart /runs/{id} still shows a run waiting for review and it can be resumed.

    graphs: dict[str, object] = {}
    events: dict[str, list[dict]] = {}
    failures: dict[str, str] = {}
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tailor")

    def get_graph(client: ClientDep):
        if "graph" not in graphs:
            graphs["graph"] = build_graph(
                client=client,
                engine=engine,
                checkpointer=open_checkpointer(checkpoint_file),
                output_dir=output_root,
            )
        return graphs["graph"]

    GraphDep = Annotated[object, Depends(get_graph)]

    def stream_run(graph, payload, run_id: str) -> None:
        log = events.setdefault(run_id, [])
        try:
            for chunk in graph.stream(payload, run_config(run_id), stream_mode="updates"):
                for node in chunk:
                    log.append({"node": "review" if node == "__interrupt__" else node})
        except Exception as e:  # reported through /runs/{id}; never crash the server
            failures[run_id] = f"{type(e).__name__}: {e}"
        log.append({"node": "__end__"})

    def status_of(graph, run_id: str) -> RunStatusOut:
        if run_id in failures:
            return RunStatusOut(run_id=run_id, status="error", error=failures[run_id])
        status = run_status(graph, run_id)
        state = status["status"]
        if state == "not_found":
            # Started but not checkpointed yet, or unknown.
            return RunStatusOut(run_id=run_id, status="running" if run_id in events else state)
        if state == "waiting_review":
            review = status["review"]
            return RunStatusOut(
                run_id=run_id, status=state, drafts=review["drafts"],
                review_errors=review.get("review_errors", {}),
            )  # fmt: skip
        if state == "done":
            rendered = status.get("render") or {}
            return RunStatusOut(
                run_id=run_id,
                status=state,
                drafts=status["drafts"],
                cv=status["cv"],
                render=render_out(run_id, rendered) if "pages" in rendered else None,
                render_error=rendered.get("error"),
            )
        return RunStatusOut(run_id=run_id, status="running")

    @app.post(
        "/jobs/{job_id}/tailor",
        response_model=RunOut,
        status_code=202,
        responses={404: {"model": ErrorOut}, 422: {"model": ErrorOut}},
    )
    def start_tailor(
        job_id: int, session: SessionDep, graph: GraphDep, body: TailorRequest | None = None
    ) -> RunOut:
        """Start tailoring a CV for a parsed job; it pauses when drafts are ready for review."""
        _get_parsed_job(session, job_id)
        body = body or TailorRequest()
        try:
            state = initial_state(job_id, profile_file, level=body.level, use_judge=body.use_judge)
        except ProfileError as e:
            raise ProfileInvalid(profile_file, e) from e
        run_id = uuid.uuid4().hex
        events[run_id] = []
        pool.submit(stream_run, graph, state, run_id)
        return RunOut(run_id=run_id, status="running")

    @app.get("/runs/{run_id}", response_model=RunStatusOut)
    def get_run(run_id: str, graph: GraphDep) -> RunStatusOut:
        """Where a run is, with its drafts once they are ready for review."""
        return status_of(graph, run_id)

    @app.post(
        "/runs/{run_id}/review", response_model=RunStatusOut, responses={409: {"model": ErrorOut}}
    )
    def review(run_id: str, body: ReviewRequest, graph: GraphDep) -> RunStatusOut:
        """Accept, edit or reject each draft. Edits are checked; a failing edit comes back."""
        if status_of(graph, run_id).status != "waiting_review":
            raise HTTPException(status_code=409, detail=f"run {run_id} is not waiting for review")
        decisions = [d.model_dump() for d in body.decisions]
        graph.invoke(Command(resume=decisions), run_config(run_id))
        return status_of(graph, run_id)

    def run_dir(run_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):  # never build paths from arbitrary input
            raise HTTPException(status_code=404, detail="unknown run")
        return output_root / "runs" / run_id

    @app.post(
        "/runs/{run_id}/render",
        response_model=RenderOut,
        responses={409: {"model": ErrorOut}, 422: {"model": ErrorOut}, 503: {"model": ErrorOut}},
    )
    def render_run(run_id: str, graph: GraphDep, body: RenderRequest | None = None) -> RenderOut:
        """Compile the reviewed CV to PDF and run the ATS check on it."""
        out_dir = run_dir(run_id)
        values = graph.get_state(run_config(run_id)).values or {}
        if not values.get("cv"):
            raise HTTPException(status_code=409, detail=f"run {run_id} has no reviewed CV yet")
        # The profile copied when the run started: the PDF matches what was reviewed.
        profile = Profile.model_validate(values["profile"])
        cv = TailoredCV.model_validate(values["cv"])
        keywords = (body.keywords if body and body.keywords is not None else None) or values.get(
            "jd", {}
        ).get("keywords", [])
        try:
            out = render_tailored(profile, cv, out_dir, keywords)
        except RenderError as e:
            status = 503 if "not found on PATH" in str(e) else 422
            raise HTTPException(status_code=status, detail=str(e)) from e
        rendered = {"pages": out.pages, "issues": out.issues, "warnings": out.warnings}
        # Record the rebuild in the run, so GET /runs/{id} shows the PDF that is on disk.
        graph.update_state(run_config(run_id), {"render": rendered}, as_node="render")
        return render_out(run_id, rendered)

    def render_out(run_id: str, rendered: dict) -> RenderOut:
        return RenderOut(
            run_id=run_id,
            pages=rendered["pages"],
            ats_ok=not rendered["issues"],
            issues=rendered["issues"],
            warnings=rendered["warnings"],
            pdf_url=f"/runs/{run_id}/cv.pdf",
            tex_url=f"/runs/{run_id}/cv.tex",
        )

    @app.get("/runs/{run_id}/cv.pdf", response_class=FileResponse)
    def run_pdf(run_id: str) -> FileResponse:
        path = run_dir(run_id) / "cv.pdf"
        if not path.exists():
            raise HTTPException(status_code=404, detail="render the CV first")
        return FileResponse(
            path, media_type="application/pdf", filename="cv.pdf", content_disposition_type="inline"
        )  # shown in the page; the UI link downloads it

    @app.get("/runs/{run_id}/cv.tex", response_class=FileResponse)
    def run_tex(run_id: str) -> FileResponse:
        path = run_dir(run_id) / "cv.tex"
        if not path.exists():
            raise HTTPException(status_code=404, detail="render the CV first")
        return FileResponse(path, media_type="application/x-tex", filename="cv.tex")

    @app.get("/runs/{run_id}/events")
    def run_events(run_id: str) -> StreamingResponse:
        """Server-sent events: one per finished step, then the run's status."""

        def stream() -> Iterator[str]:
            sent = 0
            while True:
                log = events.get(run_id, [])
                while sent < len(log):
                    yield f"data: {json.dumps(log[sent])}\n\n"
                    sent += 1
                if run_id not in events or (log and log[-1]["node"] == "__end__"):
                    return
                time.sleep(0.3)

        return StreamingResponse(stream(), media_type="text/event-stream")

    # --- applications ------------------------------------------------------------------------

    versions_dir = output_root / "cv_versions"

    def check_duplicates(session: Session, job: Job, confirmed: bool) -> None:
        found = tracker.possible_duplicates(session, job)
        if found and not confirmed:
            raise NeedsConfirmation(
                "duplicate",
                "You already have an application for this job, or applied to this company in "
                f"the last {tracker.DUPLICATE_WINDOW_DAYS} days.",
                [tracker.describe(session, a, j) for a, j in found],
            )

    def application_out(row: tracker.ApplicationRow) -> ApplicationOut:
        a = row.application
        cv = f"/cv-versions/{a.cv_version_id}" if a.cv_version_id else None
        return ApplicationOut(
            id=a.id, job_id=a.job_id, company=row.company, title=row.job.title,
            url=row.job.url, status=a.status, cv_source=a.cv_source,
            cv_version_id=a.cv_version_id, cv_pdf_url=cv and f"{cv}/cv.pdf",
            cv_tex_url=cv and f"{cv}/cv.tex", channel=a.channel, notes=a.notes,
            applied_at=a.applied_at, last_change=row.last_change,
        )  # fmt: skip

    def one_application(session: Session, application_id: int) -> ApplicationOut:
        rows = tracker.list_applications(session)
        return application_out(next(r for r in rows if r.application.id == application_id))

    @app.post(
        "/runs/{run_id}/application",
        response_model=ApplicationOut,
        status_code=201,
        responses={409: {"model": ErrorOut}},
    )
    def apply_from_run(
        run_id: str, body: ApplyFromRun, session: SessionDep, graph: GraphDep
    ) -> ApplicationOut:
        """Save a finished run as an application. Its CV is frozen: later rebuilds or profile
        edits do not change what is linked to this application."""
        out_dir = run_dir(run_id)
        values = graph.get_state(run_config(run_id)).values or {}
        rendered = values.get("render") or {}
        if not values.get("cv") or "pages" not in rendered:
            raise HTTPException(status_code=409, detail="finish the review and build the PDF first")
        job = session.get(Job, values["job_id"])
        check_duplicates(session, job, body.confirm_duplicate)
        if body.status == "applied" and rendered["issues"] and not body.confirm_ats:
            raise NeedsConfirmation(
                "ats",
                "The ATS check found problems: an ATS may misread this CV.",
                rendered["issues"],
            )
        config = load_llm_config()
        version = tracker.freeze_cv(
            session,
            job_id=job.id,
            profile_version=values["profile_version"],
            cv=values["cv"],
            tex_path=out_dir / "cv.tex",
            pdf_path=out_dir / "cv.pdf",
            versions_dir=versions_dir,
            prompt_version="rewrite/" + config.prompt_for("rewrite"),
            model=config.model_for("rewrite"),
        )
        application = tracker.create_application(
            session, job_id=job.id, status=ApplicationStatus(body.status),
            cv_version_id=version.id, channel=body.channel, notes=body.notes,
        )  # fmt: skip
        return one_application(session, application.id)

    @app.post(
        "/applications",
        response_model=ApplicationOut,
        status_code=201,
        responses={409: {"model": ErrorOut}},
    )
    def add_manual_application(body: ManualApplication, session: SessionDep) -> ApplicationOut:
        """Record an application made without factfit (the manual baseline). No LLM call."""
        job = tracker.manual_job(
            session, company=body.company, title=body.title, url=body.url, jd_text=body.jd_text
        )
        check_duplicates(session, job, body.confirm_duplicate)
        applied_at = (
            datetime.combine(body.applied_on, datetime.min.time(), tzinfo=UTC)
            if body.applied_on
            else None
        )
        application = tracker.create_application(
            session, job_id=job.id, status=body.status, channel=body.channel,
            notes=body.notes, applied_at=applied_at,
        )  # fmt: skip
        return one_application(session, application.id)

    @app.get("/applications", response_model=list[ApplicationOut])
    def get_applications(
        session: SessionDep, status: ApplicationStatus | None = None
    ) -> list[ApplicationOut]:
        """All applications, the most recently changed first."""
        return [application_out(r) for r in tracker.list_applications(session, status)]

    @app.patch(
        "/applications/{application_id}",
        response_model=ApplicationOut,
        responses={404: {"model": ErrorOut}},
    )
    def update_application(
        application_id: int, body: ApplicationUpdate, session: SessionDep
    ) -> ApplicationOut:
        """Change the status (recorded as an event), the channel or the notes."""
        application = session.get(Application, application_id)
        if application is None:
            raise HTTPException(status_code=404, detail=f"application {application_id} not found")
        if body.channel is not None:
            application.channel = body.channel
        if body.notes is not None:
            application.notes = body.notes
        session.add(application)
        session.commit()
        if body.status is not None:
            tracker.change_status(session, application, body.status)
        return one_application(session, application_id)

    def frozen_cv(session: Session, version_id: int) -> CVVersion:
        version = session.get(CVVersion, version_id)
        if version is None or not version.pdf_path:
            raise HTTPException(status_code=404, detail=f"CV version {version_id} not found")
        return version

    @app.get("/cv-versions/{version_id}/cv.pdf", response_class=FileResponse)
    def version_pdf(version_id: int, session: SessionDep) -> FileResponse:
        """The PDF exactly as it was when the application was saved."""
        path = Path(frozen_cv(session, version_id).pdf_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"{path.name} is missing from disk")
        return FileResponse(path, media_type="application/pdf", filename=f"cv-v{version_id}.pdf")

    @app.get("/cv-versions/{version_id}/cv.tex", response_class=Response)
    def version_tex(version_id: int, session: SessionDep) -> Response:
        return Response(
            frozen_cv(session, version_id).tex,
            media_type="application/x-tex",
            headers={"Content-Disposition": f'attachment; filename="cv-v{version_id}.tex"'},
        )

    return app


def _get_parsed_job(session: Session, job_id: int) -> Job:
    job = session.get(Job, job_id)
    if job is None or not job.parsed_json:
        raise HTTPException(status_code=404, detail=f"job {job_id} not found")
    return job


def _job_out(session: Session, job: Job, jd: JobDescription, cached: bool) -> JobOut:
    company = session.get(Company, job.company_id) if job.company_id else None
    return JobOut(
        id=job.id,
        title=job.title,
        company=company.name if company else jd.company,
        cached=cached,
        jd=jd,
    )


def export_openapi(path: str | Path) -> None:
    """Write the OpenAPI schema; the UI generates its TypeScript types from it."""
    import json

    spec = create_app(engine=make_engine("sqlite://")).openapi()
    text = json.dumps(spec, indent=2, ensure_ascii=False) + "\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")  # LF on every OS
