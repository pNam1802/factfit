"""Node `parse_jd`: job description text -> `JobDescription` (PRD F2).

The result is stored on a `jobs` row and cached by the hash of the normalised text plus
the prompt version: pasting the same JD again costs nothing, while switching to a new
prompt version parses it again.
"""

import hashlib
import re
from dataclasses import dataclass

from pydantic import ValidationError
from sqlalchemy import func
from sqlmodel import Session, select

from factfit.db.models import Company, Job
from factfit.llm import LLMClient
from factfit.prompts import Prompt, load_prompt
from factfit.schemas.job import JobDescription

NODE = "parse_jd"


@dataclass
class ParseResult:
    job: Job
    jd: JobDescription
    cached: bool


def normalize_jd(text: str) -> str:
    """Same JD pasted from different places -> same text, so the hash matches."""
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def jd_hash(text: str) -> str:
    return hashlib.sha256(normalize_jd(text).encode()).hexdigest()


def extract_jd(
    text: str,
    *,
    client: LLMClient,
    prompt: Prompt,
    company: str | None = None,
    run_id: str | None = None,
) -> JobDescription:
    """The LLM call alone, without database or cache. Also used by the evaluation."""
    return client.parse(
        JobDescription,
        node=NODE,
        prompt_version=prompt.id,
        system=prompt.system,
        user=prompt.render(company=company or "", jd_text=normalize_jd(text)),
        run_id=run_id,
    )


def _from_cache(parsed_json: dict) -> JobDescription | None:
    # Results saved under an older schema no longer validate: parse again instead of failing.
    try:
        return JobDescription.model_validate(parsed_json)
    except ValidationError:
        return None


def parse_jd(
    raw_text: str,
    *,
    client: LLMClient,
    session: Session,
    company: str | None = None,
    url: str | None = None,
    source: str = "manual",
    run_id: str | None = None,
) -> ParseResult:
    text = normalize_jd(raw_text)
    if not text:
        raise ValueError("the job description is empty")
    text_hash = jd_hash(text)
    prompt = load_prompt(NODE, client.config.prompt_for(NODE))

    job = session.exec(select(Job).where(Job.text_hash == text_hash)).first()
    if job and job.parsed_json and job.parse_prompt_version == prompt.id:
        cached = _from_cache(job.parsed_json)
        if cached is not None:
            return ParseResult(job=job, jd=cached, cached=True)

    jd = extract_jd(text, client=client, prompt=prompt, company=company, run_id=run_id)

    if job is None:
        job = Job(raw_text=raw_text, text_hash=text_hash, title=jd.title, source=source)
    job.title = jd.title
    job.url = url or job.url
    job.parsed_json = jd.model_dump(mode="json")
    job.parse_prompt_version = prompt.id
    company_name = company or jd.company
    if company_name:
        job.company_id = get_or_create_company(session, company_name).id

    session.add(job)
    session.commit()
    session.refresh(job)
    return ParseResult(job=job, jd=jd, cached=False)


def get_or_create_company(session: Session, name: str) -> Company:
    name = name.strip()
    found = session.exec(select(Company).where(func.lower(Company.name) == name.lower())).first()
    if found:
        return found
    company = Company(name=name)
    session.add(company)
    session.commit()
    session.refresh(company)
    return company
