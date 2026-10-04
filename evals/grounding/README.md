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

Build: `uv run factfit build-grounding-cases` (code mutations are reproducible; LLM cases
use `prompts/gen_cases/v1.md` and change on every build). Score:
`uv run factfit eval-grounding --show-misses`.

## Review (2026-10-04)

All 136 LLM-written cases were read against their source.

- Dropped `c0153`, labelled `valid_paraphrase`: "Deployed a Streamlit demo" became "Built
  and deployed ...", which adds a claim. It fit no fabricated type cleanly.
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
