import json

import pytest
from langgraph.types import Command
from sqlmodel import Session

from factfit.agent.graph import build_graph, open_checkpointer, run_config, run_status
from factfit.agent.nodes.parse_jd import jd_hash
from factfit.db import Job, init_db, make_engine
from factfit.llm import LLMClient
from factfit.llm.backends import RawOutput
from factfit.llm.config import LLMConfig
from factfit.schemas.profile import Profile

PROFILE = Profile.model_validate(
    {
        "basics": {"name": "A", "email": "a@x.y"},
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
                ],
            }
        ],
        "skills": [{"name": "YOLOv8", "evidence": ["b1"]}],
    }
)  # fmt: skip

JD = {
    "title": "CV Engineer",
    "company": "ACME",
    "seniority": "junior",
    "language": "en",
    "location": None,
    "must_have": [
        {"id": "r1", "text": "Object detection", "category": "skill", "any_of": [], "level": None}
    ],
    "nice_to_have": [],
    "responsibilities": [],
    "keywords": ["YOLOv8"],
}

CONFIG = LLMConfig.model_validate(
    {
        "models": {"s": "m"},
        "nodes": {
            "parse_jd": {"model": "s", "prompt": "v2"},
            "match": {"model": "s", "prompt": "v1"},
            "rewrite": {"model": "s", "prompt": "v1"},
            "judge": {"model": "s", "prompt": "v2"},
        },
    }
)


class FakeBackend:
    """Match: r1 met by b1. Rewrite: `rewrites(source_id)`. Judge: objects to 'robust'."""

    def __init__(self, rewrites=lambda sid, text: text.replace("Built", "Developed")):
        self.rewrites = rewrites
        self.calls: list[str] = []

    def complete(self, *, node, model, system, user, schema, options):
        self.calls.append(schema.__name__)
        if schema.__name__ == "MatchJudgement":
            reqs = [{"req_id": "r1", "status": "met", "evidence": ["b1"], "note": "YOLOv8"}]
            return RawOutput(text=json.dumps({"requirements": reqs}))
        if schema.__name__ == "RewriteOutput":
            bullets = [
                {"source_id": b.id, "text": self.rewrites(b.id, b.text)}
                for b in PROFILE.experiences[0].bullets
                if f"{b.id} |" in user
            ]
            return RawOutput(text=json.dumps({"bullets": bullets}))
        rewrite = user.split("REWRITE:")[-1]
        claims = (
            [{"claim": "robust", "kind": "inference", "why": "w"}] if "robust" in rewrite else []
        )
        return RawOutput(text=json.dumps({"unsupported_claims": claims}))


@pytest.fixture
def env(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'app.db'}")
    init_db(engine)
    raw = "CV Engineer\nObject detection"
    with Session(engine) as s:
        job = Job(title="CV Engineer", raw_text=raw, text_hash=jd_hash(raw),
                  parsed_json=JD, parse_prompt_version="parse_jd/v2")  # fmt: skip
        s.add(job)
        s.commit()
        job_id = job.id
    return engine, tmp_path / "checkpoints.db", job_id


def start(env, backend, run_id="run1", client=None, output_dir=None):
    engine, ckpt, job_id = env
    graph = build_graph(
        client=client or LLMClient(backend, CONFIG),
        engine=engine,
        checkpointer=open_checkpointer(ckpt),
        output_dir=output_dir,
    )
    state = {
        "job_id": job_id,
        "profile": PROFILE.model_dump(mode="json"),
        "profile_version": "p1",
        "level": None,
        "use_judge": True,
    }
    graph.invoke(state, run_config(run_id))
    return graph


def resume(graph, decisions, run_id="run1"):
    graph.invoke(Command(resume=decisions), run_config(run_id))
    return run_status(graph, run_id)


def test_pauses_for_review_then_finishes(env):
    backend = FakeBackend()
    graph = start(env, backend)
    status = run_status(graph, "run1")
    assert status["status"] == "waiting_review"
    drafts = {d["source_id"]: d for d in status["review"]["drafts"]}
    assert drafts["b1"]["text"] == "Developed a YOLOv8 tracker running at 25 fps"
    assert drafts["b1"]["original"] == "Built a YOLOv8 tracker running at 25 fps"

    done = resume(graph, [])  # no decisions = accept everything
    assert done["status"] == "done"
    assert done["cv"]["sections"][0]["bullets"][0]["grounding"] == "pass"


def test_resumes_after_a_restart(env):
    start(env, FakeBackend())
    engine, ckpt, _ = env
    # A fresh graph and checkpointer, as after closing and reopening the app.
    graph = build_graph(
        client=LLMClient(FakeBackend(), CONFIG), engine=engine, checkpointer=open_checkpointer(ckpt)
    )
    assert run_status(graph, "run1")["status"] == "waiting_review"
    assert resume(graph, [{"source_id": "b1", "action": "accept"}])["status"] == "done"


def test_a_fabricating_edit_is_sent_back_until_rejected(env):
    graph = start(env, FakeBackend())
    edit = {"source_id": "b2", "action": "edit", "text": "Contributed to a robust FastAPI service"}
    status = resume(graph, [edit])
    assert status["status"] == "waiting_review"
    assert "robust" in status["review"]["review_errors"]["b2"][0]

    # "Accept" does not wave a failing edit through.
    status = resume(graph, [{"source_id": "b2", "action": "accept"}])
    assert status["status"] == "waiting_review"

    done = resume(graph, [{"source_id": "b2", "action": "reject"}])
    assert done["status"] == "done"
    b2 = done["cv"]["sections"][0]["bullets"][1]
    assert b2["text"] == "Contributed to a FastAPI service"


def test_an_edit_changing_a_number_is_caught_by_the_rules(env):
    graph = start(env, FakeBackend())
    edit = {"source_id": "b1", "action": "edit", "text": "Built a YOLOv8 tracker at 60 fps"}
    status = resume(graph, [edit])
    assert "'60'" in status["review"]["review_errors"]["b1"][0]


def test_rewrites_that_keep_fabricating_fall_back_to_the_source(env):
    backend = FakeBackend(rewrites=lambda sid, text: f"{text}, robust")
    graph = start(env, backend)
    drafts = run_status(graph, "run1")["review"]["drafts"]
    assert all(d["fallback"] and d["text"] == d["original"] for d in drafts)
    assert backend.calls.count("RewriteOutput") == 2  # first try + one retry


def test_a_persons_edit_gets_every_problem_at_once(env):
    graph = start(env, FakeBackend())
    edit = {"source_id": "b1", "action": "edit", "text": "Built a robust YOLOv8 tracker at 60 fps"}
    errors = resume(graph, [edit])["review"]["review_errors"]["b1"]
    # the rule problem (60) and the judge problem (robust) in the same round
    assert any("'60'" in e for e in errors) and any("robust" in e for e in errors)


def test_every_llm_call_is_logged_under_the_run_id(env):
    logged = []
    backend = FakeBackend()
    graph = start(env, backend, client=LLMClient(backend, CONFIG, log=logged.append))
    resume(graph, [{"source_id": "b2", "action": "edit", "text": "Helped with a FastAPI app"}])
    assert logged and {c.run_id for c in logged} == {"run1"}
    assert {c.node for c in logged} >= {"match", "rewrite", "judge"}


def test_a_rewrite_equal_to_its_source_skips_the_judge(env):
    backend = FakeBackend(rewrites=lambda sid, text: text + ".")  # only a full stop added
    graph = start(env, backend)
    drafts = run_status(graph, "run1")["review"]["drafts"]
    assert all(d["passed"] and not d["fallback"] for d in drafts)
    assert "JudgeVerdict" not in backend.calls


def test_the_pdf_is_built_when_the_review_finishes(env, tmp_path, monkeypatch):
    from factfit.render.compile import TailoredRender

    built = []

    def fake_render(profile, cv, out_dir, keywords):
        built.append((out_dir, keywords))
        return TailoredRender(pages=1, issues=[], warnings=["w"])

    monkeypatch.setattr("factfit.agent.graph.render_tailored", fake_render)
    graph = start(env, FakeBackend(), output_dir=tmp_path / "out")
    done = resume(graph, [])
    assert done["render"] == {"pages": 1, "issues": [], "warnings": ["w"]}
    assert built == [(tmp_path / "out" / "runs" / "run1", ["YOLOv8"])]


def test_a_failed_compile_keeps_the_reviewed_cv(env, tmp_path, monkeypatch):
    from factfit.render.compile import RenderError

    def broken(*args):
        raise RenderError("tectonic not found on PATH")

    monkeypatch.setattr("factfit.agent.graph.render_tailored", broken)
    graph = start(env, FakeBackend(), output_dir=tmp_path / "out")
    done = resume(graph, [])
    assert done["status"] == "done" and done["cv"]
    assert "not found" in done["render"]["error"]
