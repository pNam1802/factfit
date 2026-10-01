# 0001. Use LangGraph to orchestrate the tailoring agent

- **Status:** accepted
- **Date:** 2026-10-01

## Context

Tailoring a CV is a short, mostly linear pipeline:
`parse_jd → match → select → rewrite → ground_check → human_review → render`.
Two parts make it more than a chain of function calls:

1. **A bounded repair loop.** Bullets that fail the grounding check go back to `rewrite`
   once, with the list of problems; if they still fail, they fall back to the original bullet.
2. **A mandatory human stop.** The run must pause at `human_review`, possibly for hours
   (or across a laptop restart), and resume with the user's accept / edit / reject decisions.

Options considered:

| Option | For | Against |
| --- | --- | --- |
| Plain Python functions + own state handling | No dependency, nothing hidden | Pause/resume, persistence and per-step tracing all have to be built and tested by hand |
| LangChain chains (LCEL) | Familiar, many integrations | Built for straight pipelines; loops and human pauses are awkward |
| **LangGraph** | Explicit graph with typed state; `interrupt` for human-in-the-loop; checkpointers persist state per run (SQLite available); step-by-step streaming for the UI | Extra dependency and concepts for a graph that is nearly linear; API still evolving |
| Agent framework with free tool choice | Flexible | Wrong fit: the order of steps is known, and letting a model pick steps makes grounding harder to guarantee |

## Decision

Use LangGraph with a typed `TailorState` (Pydantic), one node per step, a conditional edge
for the single repair loop, `interrupt` at `human_review`, and a SQLite checkpointer keyed by
`run_id` so a paused run survives restarts.

The graph stays deliberately simple: no node lets the model choose which tool or step comes
next. The first place the agent decides whether to call a tool is company research (P2).

## Consequences

- Pause/resume and persistence come from the library instead of custom code, which is where
  most bugs would be in a hand-written version.
- Node-level streaming feeds the UI progress bar (`GET /runs/{id}/events`) with little work.
- Nodes stay plain functions over `TailorState`, so they are unit-tested without the graph,
  and could be moved to another orchestrator if needed.
- Cost: one more framework to learn and keep up with; for a graph this small, a hand-written
  loop would also have worked. Revisit if LangGraph upgrades break the graph more than once,
  or if the human-review pause turns out to be simpler to model as two separate API calls.
