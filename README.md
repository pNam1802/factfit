# factfit

> LangGraph agent that tailors LaTeX CVs to job descriptions — every bullet traced to a verified profile, with a 3-layer hallucination check.

**Status:** 🚧 Work in progress (week 0 of 6). Metrics, demo, and setup instructions will land here once the MVP runs end-to-end.

## Why

Tailoring a CV for each job takes 30–60 minutes by hand. Generic chatbots are faster but invent numbers, skills, and roles you never had — which falls apart in the interview. factfit only rephrases, selects, and reorders what is already in your own verified profile, and refuses to ship anything it cannot trace back to a source.

## How it works

1. **Master profile** — all experiences, projects, and metrics live in one YAML file you own. Every bullet has an id.
2. **Parse JD** — extract must-have / nice-to-have requirements from a job description (English or Vietnamese).
3. **Match & gap analysis** — map each requirement to evidence bullets; score computed by code, not by the LLM.
4. **Select & rewrite** — choose the most relevant bullets and rephrase them toward the JD's keywords, only where the source already supports it.
5. **Grounding check (3 layers)**
   - structure: every output bullet points to existing source ids;
   - rules: numbers, technologies (via an alias table), and role level must match the source;
   - LLM judge: flags unsupported claims the rules cannot catch.
6. **Human review** — accept / edit / reject each bullet in a side-by-side diff before export.
7. **Render** — Jinja2 → LaTeX → PDF, plus an ATS readability check on the generated PDF.
8. **Track** — every application keeps an immutable snapshot of the exact CV that was sent.

## Planned stack

Python · LangGraph · FastAPI · Pydantic · SQLite · OpenAI API · Jinja2 + tectonic · Next.js + shadcn/ui

## Roadmap

- [ ] Week 1 — profile schema, LLM client, data model
- [ ] Week 2 — JD parsing and matching, with evals
- [ ] Week 3 — rewrite + grounding check, with a 200+ case eval set
- [ ] Week 4 — review UI, LaTeX export, application tracker
- [ ] Week 5–6 — eval report (ablation, model comparison), Docker demo without API key

## License

[MIT](LICENSE) © 2026 Nguyễn Phương Nam
