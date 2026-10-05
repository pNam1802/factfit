"""The tailoring agent as a LangGraph graph (PRD §6).

    parse -> match -> select -> rewrite -> check --(failed, retry left)--> rewrite
                                             |--(failed, no retry left)--> fallback --+
                                             +--(all passed)------------------------> review
    review (pause for the person) -> apply_review --(an edit fails the checks)--> review
                                                  +--(ok)--> assemble -> END

The state holds JSON only (dicts, lists, strings), so the SQLite checkpointer can save it and
a paused run can be resumed after a restart. Each node rebuilds objects from that JSON and
calls the same functions the CLI uses (factfit.agent.nodes.*); no logic lives here.
The profile is copied into the state when the run starts: editing profile.yaml during a
review does not change a run already in progress.
"""

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from sqlalchemy import Engine
from sqlmodel import Session

from factfit.agent.nodes.match import match_job
from factfit.agent.nodes.parse_jd import parse_jd
from factfit.agent.nodes.rewrite import MAX_ATTEMPTS, Draft, check_draft, rewrite_entry
from factfit.agent.nodes.select import ScoredBullet, SelectedEntry, select_bullets
from factfit.agent.tailor import assemble
from factfit.db.models import Job
from factfit.grounding.rules import load_kb
from factfit.llm import LLMClient
from factfit.schemas.base import Strict
from factfit.schemas.job import JobDescription
from factfit.schemas.match import MatchResult
from factfit.schemas.profile import Profile

DEFAULT_CHECKPOINTS = Path("data/checkpoints.db")
WORKERS = 6  # parallel LLM calls inside the rewrite and check nodes


class TailorState(TypedDict, total=False):
    job_id: int
    profile: dict  # Profile, copied when the run starts
    profile_version: str
    level: str | None
    use_judge: bool
    jd: dict  # JobDescription
    match: dict  # MatchResult
    selected: list[dict]  # [{section, entry_id, bullets: [{id, score, reasons}]}]
    drafts: list[dict]  # one per selected bullet, see _draft_to_dict
    decisions: list[dict]  # what the person sent at the last review
    review_errors: dict[str, list[str]]  # source_id -> why an edit was refused
    cv: dict  # TailoredCV, once assembled


class ReviewDecision(Strict):
    source_id: str
    action: Literal["accept", "edit", "reject"]  # reject = keep the original bullet
    text: str | None = None  # the new text, for "edit"


# --- JSON <-> objects ------------------------------------------------------------------------


def _profile(state: TailorState) -> Profile:
    return Profile.model_validate(state["profile"])


def _selected(state: TailorState, profile: Profile) -> list[SelectedEntry]:
    entries = {e.id: e for e in [*profile.experiences, *profile.projects]}
    out = []
    for s in state["selected"]:
        entry = entries[s["entry_id"]]
        bullets = {b.id: b for b in entry.bullets}
        scored = [ScoredBullet(bullets[b["id"]], b["score"], b["reasons"]) for b in s["bullets"]]
        out.append(SelectedEntry(s["section"], entry, scored))
    return out


def _drafts(state: TailorState, selected: list[SelectedEntry]) -> list[Draft]:
    by_entry = {s.entry.id: s for s in selected}
    drafts = []
    for d in state["drafts"]:
        entry = by_entry[d["entry_id"]]
        source = next(s.bullet for s in entry.bullets if s.bullet.id == d["source_id"])
        drafts.append(
            Draft(
                entry=entry,
                source=source,
                text=d["text"],
                attempts=d["attempts"],
                issues=list(d["issues"]),
                history=[(t, list(p)) for t, p in d["history"]],
                passed=d["passed"],
                fallback=d["fallback"],
            )
        )
    return drafts


def _draft_to_dict(d: Draft, extra: dict | None = None) -> dict:
    return {
        "source_id": d.source.id,
        "entry_id": d.entry.entry.id,
        "original": d.source.text,
        "text": d.text,
        "attempts": d.attempts,
        "issues": d.issues,
        "history": [[t, p] for t, p in d.history],
        "passed": d.passed,
        "fallback": d.fallback,
        **(extra or {}),
    }


def _save_drafts(state: TailorState, drafts: list[Draft]) -> list[dict]:
    """Keep review flags (edited / rejected) that Draft does not carry."""
    old = {d["source_id"]: d for d in state.get("drafts", [])}
    keep = ("user_edited", "user_rejected")
    return [
        _draft_to_dict(d, {k: old.get(d.source.id, {}).get(k, False) for k in keep}) for d in drafts
    ]


# --- the graph -------------------------------------------------------------------------------


def build_graph(*, client: LLMClient, engine: Engine, checkpointer=None):
    kb = load_kb()

    def parse(state: TailorState) -> dict:
        with Session(engine) as session:
            job = session.get(Job, state["job_id"])
            if job is None:
                raise ValueError(f"job {state['job_id']} not found")
            result = parse_jd(job.raw_text, client=client, session=session)
        return {"jd": result.jd.model_dump(mode="json")}

    def match(state: TailorState) -> dict:
        jd = JobDescription.model_validate(state["jd"])
        level = state.get("level") or (jd.seniority if jd.seniority != "unknown" else None)
        with Session(engine) as session:
            outcome = match_job(
                session.get(Job, state["job_id"]), _profile(state),
                profile_version=state["profile_version"], client=client, session=session,
                level=level,
            )  # fmt: skip
        return {"match": outcome.result.model_dump(mode="json"), "level": level}

    def select(state: TailorState) -> dict:
        profile = _profile(state)
        jd = JobDescription.model_validate(state["jd"])
        selected = select_bullets(profile, jd, MatchResult.model_validate(state["match"]), kb=kb)
        drafts = [Draft(entry, s.bullet) for entry in selected for s in entry.bullets]
        return {
            "selected": [
                {
                    "section": s.section,
                    "entry_id": s.entry.id,
                    "bullets": [
                        {"id": b.bullet.id, "score": b.score, "reasons": b.reasons}
                        for b in s.bullets
                    ],
                }
                for s in selected
            ],
            "drafts": [_draft_to_dict(d) for d in drafts],
        }

    def rewrite(state: TailorState) -> dict:
        profile = _profile(state)
        selected = _selected(state, profile)
        drafts = _drafts(state, selected)
        jd = JobDescription.model_validate(state["jd"])
        match_result = MatchResult.model_validate(state["match"])
        pending = [d for d in drafts if not d.passed]
        groups = [(e, [d for d in pending if d.entry is e]) for e in selected]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:  # one LLM call per entry
            list(
                pool.map(
                    lambda g: rewrite_entry(
                        g[0], g[1], jd=jd, match=match_result, client=client, kb=kb
                    ),
                    [g for g in groups if g[1]],
                )
            )
        return {"drafts": _save_drafts(state, drafts)}

    def check(state: TailorState) -> dict:
        drafts = _drafts(state, _selected(state, _profile(state)))
        pending = [d for d in drafts if not d.passed]
        use_judge = state.get("use_judge", True)
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(
                pool.map(
                    lambda d: check_draft(d, client=client, kb=kb, use_judge=use_judge), pending
                )
            )
        for d in pending:
            if not d.passed:
                d.history.append((d.text, d.issues))
        return {"drafts": _save_drafts(state, drafts)}

    def after_check(state: TailorState) -> str:
        pending = [d for d in state["drafts"] if not d["passed"]]
        if not pending:
            return "review"
        if all(d["attempts"] >= MAX_ATTEMPTS for d in pending):
            return "fallback"
        return "rewrite"

    def fallback(state: TailorState) -> dict:
        drafts = _drafts(state, _selected(state, _profile(state)))
        for d in drafts:
            if not d.passed:
                d.text, d.fallback, d.passed = d.source.text, True, True
        return {"drafts": _save_drafts(state, drafts)}

    def review(state: TailorState) -> dict:
        # Runs again from the top when resumed: build the payload only, no side effects.
        decisions = interrupt(
            {
                "drafts": state["drafts"],
                "review_errors": state.get("review_errors", {}),
                "instructions": "Send a decision per source_id: accept, edit (with text) or "
                "reject (keep the original). Bullets without a decision are accepted.",
            }
        )
        return {"decisions": decisions or []}

    def apply_review(state: TailorState) -> dict:
        decisions = {d.source_id: d for d in map(ReviewDecision.model_validate, state["decisions"])}
        drafts = _drafts(state, _selected(state, _profile(state)))
        flags = {d["source_id"]: d for d in state["drafts"]}
        errors: dict[str, list[str]] = {}
        out = []
        for d in drafts:
            extra = {k: flags[d.source.id].get(k, False) for k in ("user_edited", "user_rejected")}
            decision = decisions.get(d.source.id)
            if decision and decision.action == "reject":
                d.text, d.passed = d.source.text, True
                extra = {"user_edited": False, "user_rejected": True}
            elif decision and decision.action == "edit":
                d.text = (decision.text or "").strip()
                extra = {"user_edited": True, "user_rejected": False}
                # A person's edit can add claims too: it passes the same checks, and every
                # problem is reported at once rather than one layer per round trip.
                check_draft(
                    d,
                    client=client,
                    kb=kb,
                    use_judge=state.get("use_judge", True),
                    all_problems=True,
                )
            # "accept" leaves the draft as it is. Any draft still failing, including an edit
            # refused earlier and now "accepted", sends the run back to review.
            if not d.passed:
                errors[d.source.id] = d.issues
            out.append(_draft_to_dict(d, extra))
        return {"drafts": out, "review_errors": errors, "decisions": []}

    def after_review(state: TailorState) -> str:
        return "review" if state.get("review_errors") else "assemble"

    def assemble_node(state: TailorState) -> dict:
        profile = _profile(state)
        selected = _selected(state, profile)
        drafts = _drafts(state, selected)
        cv = assemble(selected, drafts, profile, JobDescription.model_validate(state["jd"]), kb)
        return {"cv": cv.model_dump(mode="json")}

    g = StateGraph(TailorState)
    for name, fn in [
        ("parse", parse), ("match", match), ("select", select), ("rewrite", rewrite),
        ("check", check), ("fallback", fallback), ("review", review),
        ("apply_review", apply_review), ("assemble", assemble_node),
    ]:  # fmt: skip
        g.add_node(name, fn)
    g.add_edge(START, "parse")
    g.add_edge("parse", "match")
    g.add_edge("match", "select")
    g.add_edge("select", "rewrite")
    g.add_edge("rewrite", "check")
    g.add_conditional_edges("check", after_check, ["rewrite", "fallback", "review"])
    g.add_edge("fallback", "review")
    g.add_edge("review", "apply_review")
    g.add_conditional_edges("apply_review", after_review, ["review", "assemble"])
    g.add_edge("assemble", END)
    return g.compile(checkpointer=checkpointer)


def open_checkpointer(path: str | Path = DEFAULT_CHECKPOINTS) -> SqliteSaver:
    """Checkpoints live in their own SQLite file, apart from the app database."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


# --- running and inspecting a run ------------------------------------------------------------


def initial_state(
    job_id: int, profile_path: str | Path, level: str | None = None, use_judge: bool = True
) -> TailorState:
    """Load and validate the profile now; raises ProfileError if it is not valid."""
    from factfit.profile import load_profile, profile_version

    profile = load_profile(profile_path)
    return {
        "job_id": job_id,
        "profile": profile.model_dump(mode="json"),
        "profile_version": profile_version(profile_path),
        "level": level,
        "use_judge": use_judge,
    }


def run_config(run_id: str) -> dict:
    return {"configurable": {"thread_id": run_id}}


def run_status(graph, run_id: str) -> dict[str, Any]:
    """Where a run is: not_found, running, waiting_review or done, with what the UI needs."""
    snapshot = graph.get_state(run_config(run_id))
    values = snapshot.values or {}
    if not values:
        return {"status": "not_found"}
    if snapshot.interrupts:
        return {"status": "waiting_review", "review": snapshot.interrupts[0].value}
    if snapshot.next:
        return {"status": "running", "next": list(snapshot.next)}
    return {"status": "done", "cv": values.get("cv"), "drafts": values.get("drafts", [])}
