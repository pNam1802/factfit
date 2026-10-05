import json
from datetime import date

from factfit.agent.nodes.rewrite import allowed_keywords, rewrite_and_check
from factfit.agent.nodes.select import TailorConfig, select_bullets
from factfit.agent.tailor import pick_skills, tailor
from factfit.grounding.rules import load_kb
from factfit.llm import LLMClient
from factfit.llm.backends import RawOutput
from factfit.llm.config import LLMConfig
from factfit.schemas.job import JobDescription
from factfit.schemas.match import MatchResult
from factfit.schemas.profile import Profile

PROFILE = Profile.model_validate(
    {
        "basics": {"name": "A", "email": "a@x.y"},
        "summary_variants": [
            {"id": "sum_cv", "tags": ["computer-vision"], "text": "CV engineer"},
            {"id": "sum_llm", "tags": ["llm", "agent"], "text": "LLM engineer"},
        ],
        "experiences": [
            {
                "id": "exp_a",
                "org": "Acme",
                "role": "Intern",
                "start": "2026-01",
                "end": "present",
                "bullets": [
                    {"id": "b1", "text": "Built a YOLOv8 tracker running at 25 fps",
                     "skills": ["YOLOv8"],
                     "metrics": [{"name": "fps", "value": 25, "verified": True}]},
                    {"id": "b2", "text": "Contributed to a FastAPI service", "skills": ["FastAPI"]},
                    {"id": "b3", "text": "Organised team lunches"},
                    {"id": "b4", "text": "Wrote internal docs"},
                ],
            }
        ],
        "projects": [
            {"id": "p_rag", "name": "RAG bot", "bullets": [
                {"id": "b5", "text": "Built a RAG chatbot with LangChain", "skills": ["LangChain"]}]},  # noqa: E501
            {"id": "p_old", "name": "Old game", "end": "2020-01", "bullets": [
                {"id": "b6", "text": "Made a snake game"}]},
        ],
        "skills": [
            {"name": "YOLOv8", "evidence": ["b1"]},
            {"name": "LangChain", "evidence": ["b5"]},
            {"name": "Snake", "evidence": ["b6"]},
        ],
    }
)  # fmt: skip


def req(id_, text):
    return {"id": id_, "text": text, "category": "skill", "any_of": [], "level": None}


JD = JobDescription.model_validate(
    {
        "title": "LLM Engineer",
        "company": None,
        "seniority": "junior",
        "language": "en",
        "location": None,
        "must_have": [req("r1", "LLM apps"), req("r2", "Object detection")],
        "nice_to_have": [req("r3", "APIs")],
        "responsibilities": [],
        "keywords": ["LangChain", "LLM", "FastAPI", "YOLOv8"],
    }
)

MATCH = MatchResult(
    score=80,
    requirements=[
        {"req_id": "r1", "status": "met", "evidence": ["b5"], "note": ""},
        {"req_id": "r2", "status": "met", "evidence": ["b1"], "note": ""},
        {"req_id": "r3", "status": "partial", "evidence": ["b2"], "note": ""},
    ],
)

CONFIG = TailorConfig.model_validate(
    {
        "limits": {"bullets_per_experience": 2, "projects": 1, "bullets_per_project": 1},
        "weights": {
            "evidence": {"must_met": 3, "must_partial": 2, "nice_met": 1.5, "nice_partial": 1},
            "keyword": 0.5,
            "keyword_cap": 2,
            "recency": 1,
            "recency_years": 4,
        },
    }
)


def test_select_keeps_best_bullets_in_profile_order_and_best_project():
    selected = select_bullets(PROFILE, JD, MATCH, config=CONFIG, today=date(2026, 10, 5))
    exp, proj = selected
    assert [s.bullet.id for s in exp.bullets] == ["b1", "b2"]  # lunches and docs dropped
    assert proj.entry.id == "p_rag"  # the old game loses the single project slot
    assert any("evidence for r2" in r for r in exp.bullets[0].reasons)


def test_allowed_keywords_are_only_those_the_sources_support():
    entry = select_bullets(PROFILE, JD, MATCH, config=CONFIG)[0]
    assert allowed_keywords(entry, JD, load_kb()) == ["FastAPI", "YOLOv8"]


def test_skills_listed_only_with_evidence_on_the_cv():
    assert pick_skills(PROFILE, {"b1", "b5"}, JD, load_kb()) == ["YOLOv8", "LangChain"]


class ScriptedBackend:
    """Rewrite answers scripted per attempt; the judge objects to the word 'robust'."""

    def __init__(self, rewrites: list[dict[str, str]]):
        self.rewrites = rewrites
        self.prompts: list[str] = []

    def complete(self, *, node, model, system, user, schema, options):
        if schema.__name__ == "RewriteOutput":
            self.prompts.append(user)
            answer = self.rewrites.pop(0)
            bullets = [{"source_id": k, "text": v} for k, v in answer.items() if k in user]
            return RawOutput(text=json.dumps({"bullets": bullets}))
        claims = []
        if "robust" in user.split("REWRITE:")[-1]:
            claims = [{"claim": "robust", "kind": "inference", "why": "not in source"}]
        return RawOutput(text=json.dumps({"unsupported_claims": claims}))


LLM_CONFIG = LLMConfig.model_validate(
    {
        "models": {"s": "m"},
        "nodes": {"rewrite": {"model": "s", "prompt": "v1"}, "judge": {"model": "s"}},
    }
)


def test_retry_with_feedback_then_fallback():
    selected = select_bullets(PROFILE, JD, MATCH, config=CONFIG)[:1]  # exp_a: b1, b2
    backend = ScriptedBackend(
        [
            {  # attempt 1: b1 changes the number (rules), b2 adds a quality (judge)
                "b1": "Built a YOLOv8 tracker running at 30 fps",
                "b2": "Contributed to a robust FastAPI service",
            },
            {  # attempt 2: b1 fixed; b2 still invents
                "b1": "Built a 25 fps YOLOv8 tracker",
                "b2": "Contributed to a robust, production FastAPI service",
            },
        ]
    )
    drafts = rewrite_and_check(
        selected, jd=JD, match=MATCH, client=LLMClient(backend, LLM_CONFIG), workers=1
    )
    b1, b2 = drafts
    assert b1.passed and not b1.fallback and b1.text == "Built a 25 fps YOLOv8 tracker"
    assert b2.fallback and b2.text == PROFILE.experiences[0].bullets[1].text
    # the retry prompt told the model what was wrong
    assert "30" in backend.prompts[1] and "robust" in backend.prompts[1]


def test_tailor_assembles_an_exportable_cv():
    backend = ScriptedBackend(
        [
            {"b1": "Built a YOLOv8 tracker at 25 fps", "b2": "Contributed to a FastAPI service"},
            {"b5": "Built a LangChain RAG chatbot"},
        ]
    )
    run = tailor(PROFILE, JD, MATCH, client=LLMClient(backend, LLM_CONFIG), config=CONFIG)
    assert run.cv.ready_to_export()
    assert run.cv.summary.source_ids == ["sum_llm"]
    assert [s.ref for s in run.cv.sections] == ["exp_a", "p_rag"]
    assert run.cv.sections[1].bullets[0].source_bullet_ids == ["b5"]
