from pathlib import Path

from factfit.agent.fit import fit_one_page
from factfit.grounding.rules import load_kb
from factfit.render.compile import TailoredRender
from factfit.schemas.job import JobDescription
from factfit.schemas.profile import Metric, Profile
from factfit.schemas.tailored import TailoredCV

PROFILE = Profile.model_validate(
    {
        "basics": {"name": "A", "email": "a@x.y"},
        "experiences": [
            {"id": "e1", "org": "Acme", "role": "Intern", "start": "2026-01", "end": "present",
             "bullets": [{"id": "x1", "text": "Built a YOLOv8 tracker", "skills": ["YOLOv8"]},
                         {"id": "x2", "text": "Wrote docs"}]},
        ],
        "projects": [
            {"id": "p1", "name": "RAG bot", "bullets": [
                {"id": "y1", "text": "Built a RAG bot with LangChain", "skills": ["LangChain"]}]},
            {"id": "p2", "name": "Game", "bullets": [{"id": "z1", "text": "Made a game"}]},
        ],
        "skills": [{"name": "YOLOv8", "evidence": ["x1"]},
                   {"name": "LangChain", "evidence": ["y1"]}],
    }
)  # fmt: skip
JD = JobDescription.model_validate(
    {"title": "T", "company": None, "seniority": "junior", "language": "en", "location": None,
     "must_have": [], "nice_to_have": [], "responsibilities": [], "keywords": ["YOLOv8"]}
)  # fmt: skip
CV = TailoredCV.model_validate(
    {
        "sections": [
            {"type": "experience", "ref": "e1", "bullets": [
                {"text": "Built a YOLOv8 tracker", "source_bullet_ids": ["x1"],
                 "grounding": "pass"},
                {"text": "Wrote docs", "source_bullet_ids": ["x2"], "grounding": "pass"}]},
            {"type": "project", "ref": "p1", "bullets": [
                {"text": "Built a RAG bot", "source_bullet_ids": ["y1"], "grounding": "pass"}]},
            {"type": "project", "ref": "p2", "bullets": [
                {"text": "Made a game", "source_bullet_ids": ["z1"], "grounding": "pass"}]},
        ],
        "skills": ["YOLOv8", "LangChain"],
    }
)  # fmt: skip
SCORES = {"x1": 5.0, "x2": 0.5, "y1": 2.0, "z1": 1.0}


def fits_with(max_bullets: int):
    calls = []

    def render(profile, cv, out_dir, keywords):
        calls.append(len(cv.all_bullets()))
        pages = 1 if len(cv.all_bullets()) <= max_bullets else 2
        return TailoredRender(pages=pages, issues=[], warnings=[])

    return render, calls


def fit(max_bullets: int):
    render, calls = fits_with(max_bullets)
    out = fit_one_page(PROFILE, CV, JD, load_kb(), SCORES, Path("unused"), render)
    return out, calls


def test_a_cv_that_fits_is_left_alone():
    out, calls = fit(4)
    assert out.cv == CV and out.removed == [] and calls == [4]


def test_the_weakest_bullet_goes_first_then_the_weakest_project():
    out, calls = fit(2)
    assert calls == [4, 3, 2]
    kept = [b.source_bullet_ids[0] for b in out.cv.all_bullets()]
    assert kept == ["x1", "y1"]  # x2 (score 0.5), then the game project (1.0)
    assert "Wrote docs" in out.removed[0] and out.removed[1] == "project Game"


def test_skills_follow_the_bullets_that_remain():
    out, _ = fit(1)
    assert [s.ref for s in out.cv.sections] == ["e1"]  # experiences are never removed
    assert out.cv.skills == ["YOLOv8"]  # LangChain lost its evidence with the RAG project


def test_when_nothing_is_left_to_cut_the_page_count_stays_a_problem():
    out, calls = fit(0)
    assert out.render.pages == 2 and len(out.cv.all_bullets()) == 1


def test_on_a_tie_a_bullet_with_a_verified_number_stays():
    profile = PROFILE.model_copy(deep=True)
    measured = Metric(name="fps", value=25, verified=True)
    profile.experiences[0].bullets[0].metrics = [measured]  # x1 is measured
    render, _ = fits_with(3)
    tie = {**SCORES, "x1": 0.5}  # x1 and x2 now score the same; x1 comes first
    out = fit_one_page(profile, CV, JD, load_kb(), tie, Path("unused"), render)
    assert "Wrote docs" in out.removed[0]
