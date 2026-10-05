# Grounding test set

Pairs of (source bullet, rewritten bullet, label) used to measure how well factfit catches
fabricated CV bullets (PRD §8.3). Public on purpose: every source bullet in
[`sources.yaml`](sources.yaml) is fictional.

| Type | Label | Origin | What the rewrite does |
| --- | --- | --- | --- |
| number_change | fabricated | code | changes one number |
| tech_injection | fabricated | code | adds a technology the source does not use |
| role_inflation | fabricated | code | claims leadership ("Led the team that ...") |
| scale_inflation | fabricated | LLM | claims bigger scope in words only (production, company-wide) |
| unsupported_outcome | fabricated | LLM | claims a result the source does not state, words only |
| valid_paraphrase | grounded | LLM | faithful rewording; used to measure false positives |
| subtle_claim | fabricated | hand | one claim woven into the sentence: an inferred quality, an extra task, a purpose |
| hard_paraphrase | grounded | hand | faithful rewording with heavy restructuring |

Generated cases are in [`cases.jsonl`](cases.jsonl); hand-written ones in
[`hand_cases.yaml`](hand_cases.yaml), which a rebuild never overwrites.

Build: `uv run factfit build-grounding-cases` (code mutations are reproducible; LLM cases
use `prompts/gen_cases/v1.md` and change on every build). Score the rules alone with
`uv run factfit eval-grounding`, or the full ablation with
`uv run factfit eval-grounding --judges gpt-5-mini:low gpt-5-mini:minimal gpt-5.4-mini:low --write-report`.

**Results: [RESULTS.md](RESULTS.md).** Why the product runs rules first, then the judge:
[ADR 0003](../../docs/adr/0003-grounding-code-first.md).

## Review (2026-10-04)

All 136 LLM-written cases were read against their source.

- Dropped `c0153`, labelled `valid_paraphrase`: "Deployed a Streamlit demo" became "Built
  and deployed ...", which adds a claim. It fit no fabricated type cleanly.
- Dropped `c0110` on 2026-10-05 for the same reason ("Trained" became "Built and trained").
  The first review missed it; the gpt-5.4-mini judge flagged it, and it was right.
- The 48 hand-written cases (2026-10-05) were written by the same author as the judge prompt,
  which may make them easier for that prompt than cases written by someone else.
- LLM fabrications that also changed a number, technology or role verb are dropped at build
  time, so the words-only types test only words. Building also surfaced two rule false
  positives ("leading to ...", "drove adoption"), fixed before this version of the set.

## Known limitations

- LLM fabrications are blunt: most append "company-wide", "production use" or "cost
  savings" at the end. Real fabrications in rewrites are often subtler (an inferred
  "real-time" inside the sentence). A hand-written set of subtle cases is still to come.
- One known rule false positive is kept and not fixed against this set, to avoid tuning on
  the test data: `c0194` "Was on a hackathon team that created ..." reads "created" as the
  role because "was on" is not a listed role phrase.
- Small samples: report the 95% interval, not only the rate.
