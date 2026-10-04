"""Node `match`: job requirements x verified profile -> `MatchResult` (PRD F3).

The LLM judges each requirement and cites profile ids. Code then:
- keeps only requirements for the applicant's level,
- drops cited ids that do not exist in the profile (an unbacked "met" becomes "missing"),
- fills in any requirement the model skipped, as "missing",
- computes the score from config/match.yaml.

Results are cached per job, profile version, prompt version and level.
"""

from dataclasses import dataclass

from sqlmodel import Session, select

from factfit.agent.scoring import Weights, compute_score, load_weights
from factfit.db.models import Job, Match
from factfit.llm import LLMClient
from factfit.prompts import load_prompt
from factfit.schemas.job import JobDescription, Requirement, Seniority
from factfit.schemas.match import MatchJudgement, MatchResult, RequirementMatch
from factfit.schemas.profile import Profile

NODE = "match"
MAX_EVIDENCE = 3


@dataclass
class ProfileItem:
    id: str
    where: str
    text: str
    skills: list[str]


@dataclass
class MatchOutcome:
    match: Match
    result: MatchResult
    cached: bool


def profile_items(profile: Profile) -> list[ProfileItem]:
    """Everything that can serve as evidence, each with an id the LLM can cite."""
    items: list[ProfileItem] = []

    def span(start, end) -> str:
        return f"{start or '?'} to {end or '?'}"

    for e in profile.experiences:
        where = f"{e.role} at {e.org}, {span(e.start, e.end)}"
        items += [ProfileItem(b.id, where, b.text, b.skills) for b in e.bullets]
    for p in profile.projects:
        where = f"project {p.name}, {span(p.start, p.end)}"
        items += [ProfileItem(b.id, where, b.text, b.skills) for b in p.bullets]
    for ed in profile.education:
        text = f"{ed.degree}, {ed.school}, {span(ed.start, ed.end)}"
        items.append(
            ProfileItem(ed.id, "education", text + (f", GPA {ed.gpa}" if ed.gpa else ""), [])
        )
        items += [ProfileItem(b.id, f"education {ed.school}", b.text, b.skills) for b in ed.bullets]
    for c in profile.certifications:
        text = ", ".join(x for x in (c.name, c.issuer, c.date) if x)
        items.append(ProfileItem(c.id, "certification", text, []))
    for section in ("publications", "awards"):
        for entry in getattr(profile, section):
            title = getattr(entry, "title", entry.id)
            items.append(ProfileItem(entry.id, section[:-1], title, []))
            items += [
                ProfileItem(b.id, f"{section[:-1]} {title}", b.text, b.skills)
                for b in entry.bullets
            ]
    return items


def applicable(jd: JobDescription, level: Seniority | None) -> tuple[list[Requirement], list[str]]:
    """Split requirements into those for this applicant and those for another level."""
    keep, excluded = [], []
    for r in jd.all_requirements():
        (keep if r.level is None or level is None or r.level == level else excluded).append(r)
    return keep, [r.id for r in excluded]


def judge_match(
    jd: JobDescription,
    profile: Profile,
    *,
    client: LLMClient,
    level: Seniority | None,
    weights: Weights | None = None,
    run_id: str | None = None,
) -> MatchResult:
    weights = weights or load_weights()
    requirements, excluded = applicable(jd, level)
    if not requirements:
        return MatchResult(score=0, level=level, requirements=[], excluded=excluded)

    items = profile_items(profile)
    must_ids = {r.id for r in jd.must_have}
    prompt = load_prompt(NODE, client.config.prompt_for(NODE))
    judgement = client.parse(
        MatchJudgement,
        node=NODE,
        prompt_version=prompt.id,
        system=prompt.system,
        user=prompt.render(
            requirements="\n".join(
                f"{r.id} | {'must' if r.id in must_ids else 'nice'} | {r.category} | {r.text}"
                + (f" | any_of: {', '.join(r.any_of)}" if r.any_of else "")
                for r in requirements
            ),
            profile="\n".join(
                f"{i.id} | {i.where} | {i.text}" + (f" | {', '.join(i.skills)}" if i.skills else "")
                for i in items
            ),
        ),
        run_id=run_id,
    )

    judged = check_judgement(judgement, requirements, {i.id for i in items})
    return MatchResult(
        score=compute_score(jd, judged, weights),
        level=level,
        requirements=judged,
        excluded=excluded,
    )


def check_judgement(
    judgement: MatchJudgement, requirements: list[Requirement], valid_ids: set[str]
) -> list[RequirementMatch]:
    """Grounding for match: every cited id must exist; every requirement judged once."""
    by_id: dict[str, RequirementMatch] = {}
    wanted = {r.id for r in requirements}
    for r in judgement.requirements:
        if r.req_id not in wanted or r.req_id in by_id:
            continue
        evidence = [e for e in dict.fromkeys(r.evidence) if e in valid_ids][:MAX_EVIDENCE]
        status, note = r.status, r.note
        if status in ("met", "partial") and not evidence:
            status, note = "missing", f"{note} [cited ids not found in profile]"
        if status in ("missing", "unverifiable"):
            evidence = []
        by_id[r.req_id] = RequirementMatch(
            req_id=r.req_id, status=status, evidence=evidence, note=note
        )

    return [
        by_id.get(r.id)
        or RequirementMatch(
            req_id=r.id, status="missing", evidence=[], note="not judged by the model"
        )
        for r in requirements
    ]


def match_job(
    job: Job,
    profile: Profile,
    *,
    profile_version: str,
    client: LLMClient,
    session: Session,
    level: Seniority | None,
    weights: Weights | None = None,
    run_id: str | None = None,
) -> MatchOutcome:
    if not job.parsed_json:
        raise ValueError(f"job {job.id} has not been parsed yet")
    jd = JobDescription.model_validate(job.parsed_json)
    prompt_id = load_prompt(NODE, client.config.prompt_for(NODE)).id

    previous = session.exec(
        select(Match).where(
            Match.job_id == job.id,
            Match.profile_version == profile_version,
            Match.prompt_version == prompt_id,
        )
    ).all()
    for row in previous:
        result = MatchResult.model_validate(row.result_json)
        if result.level == level:
            # The judgement is cached; the score is recomputed so weight changes apply.
            score = compute_score(jd, result.requirements, weights or load_weights())
            return MatchOutcome(
                match=row, result=result.model_copy(update={"score": score}), cached=True
            )

    result = judge_match(jd, profile, client=client, level=level, weights=weights, run_id=run_id)
    row = Match(
        job_id=job.id,
        profile_version=profile_version,
        prompt_version=prompt_id,
        score=result.score,
        result_json=result.model_dump(mode="json"),
    )
    session.add(row)
    session.commit()
    session.refresh(row)
    return MatchOutcome(match=row, result=result, cached=False)
