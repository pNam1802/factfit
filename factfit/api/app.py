"""HTTP API used by the UI. A thin layer over the same functions the CLI calls.

Run with `uv run factfit dev` (API + UI), or the API alone with
`uv run uvicorn factfit.api.app:create_app --factory --reload`.
Interactive docs: http://localhost:8000/docs
"""

import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy import Engine
from sqlmodel import Session

from factfit.agent.nodes.match import match_job, profile_items
from factfit.agent.nodes.parse_jd import parse_jd
from factfit.api.schemas import (
    CreateJob,
    ErrorOut,
    Evidence,
    JobOut,
    MatchOut,
    MatchRequest,
    RequirementOut,
)
from factfit.db import Company, Job, init_db, make_engine
from factfit.llm import LLMClient, LLMError, make_client
from factfit.profile import ProfileError, load_profile, profile_version
from factfit.schemas.job import JobDescription

DEFAULT_PROFILE = "data/profile.yaml"


class ProfileInvalid(Exception):
    def __init__(self, path: str, error: ProfileError):
        self.path, self.error = path, error


def create_app(
    engine: Engine | None = None,
    client_factory: Callable[..., LLMClient] = make_client,
    profile_path: str | None = None,
) -> FastAPI:
    load_dotenv()
    engine = engine or make_engine()
    init_db(engine)
    profile_file = profile_path or os.environ.get("FACTFIT_PROFILE", DEFAULT_PROFILE)

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
