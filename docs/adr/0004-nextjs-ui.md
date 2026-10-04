# 0004. Build the UI with Next.js and shadcn/ui, not Streamlit

- **Status:** accepted
- **Date:** 2026-10-04

## Context

The screen used most is the review screen: for every requirement of a job, see whether the
profile meets it and which bullets prove it, and later accept, edit or reject rewritten bullets
one by one. A tailoring run also takes 30–60 seconds, so the UI must show progress instead of
looking frozen. The API already exists (FastAPI) and is the only place that runs the agent.

Options considered:

| Option | For | Against |
| --- | --- | --- |
| Streamlit | All Python, fastest first screen | Reruns the whole script on every click, so a long list of rows with expand/edit states feels slow; layout and keyboard control are limited |
| Gradio / NiceGUI | Python, more reactive than Streamlit | Same ceiling on custom layouts; less common in job postings |
| **Next.js + Tailwind + shadcn/ui** | Instant interactions, full control of layout and keyboard shortcuts, typed client generated from the API schema; a separate frontend like most real systems | A second language and toolchain (Node, npm); about 3–5 more days than Streamlit |

## Decision

A Next.js app in `ui/` (App Router, TypeScript, Tailwind, shadcn/ui components built on Base UI).

- The browser calls `/api/...` on the Next.js server, which forwards to FastAPI
  (`rewrites` in `next.config.ts`): same origin, no CORS setup.
- Types are generated, never written by hand: `uv run factfit export-openapi` writes
  `ui/openapi.json`, `npm run gen:api` turns it into `src/lib/api-types.ts`, and a test fails
  when `openapi.json` is stale.
- `uv run factfit dev` starts both servers and stops both on Ctrl+C, the same on Windows and
  Unix (a Makefile would not be).

## Consequences

- The UI holds no business logic; the CLI and the UI call the same API functions.
- Found while building: the Next.js rewrite proxy cuts requests after 30 s by default, shorter
  than a real parse (36 s measured) or match (31 s). `experimental.proxyTimeout` is raised to
  120 s; streaming progress (`GET /runs/{id}/events`) will replace long blocking requests later.
- Cost: two dependency trees to keep updated (`uv.lock`, `package-lock.json`), and `npm audit`
  reports advisories in dev-only tooling that cannot be fixed without breaking upgrades.
- Revisit if the UI stays a single read-only page: then the extra toolchain is not paying off.
