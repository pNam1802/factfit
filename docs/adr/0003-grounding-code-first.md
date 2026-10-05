# 0003. Check grounding with code first, then an LLM judge

- **Status:** accepted
- **Date:** 2026-10-05

## Context

Every rewritten CV bullet must add nothing its source bullets do not support. Two ways to
check: rules in code (numbers, technologies via an alias table, role-verb levels), or an LLM
asked to list unsupported claims. Measured on the public test set in
[`evals/grounding`](../../evals/grounding/RESULTS.md) (268 cases: 186 fabricated in six
kinds, 82 faithful paraphrases):

| Checker | Recall, fabricated | False alarms | $ / 1,000 bullets | Median latency |
| --- | ---: | ---: | ---: | ---: |
| rules only | 46% | 1% | 0 | ~0 s |
| judge gpt-5-mini, low effort | **100%** | **0%** | 0.52 | 2.8 s |
| judge gpt-5-mini, minimal effort | 98% (misses 3 of 32 subtle claims) | 0% | 0.28 | 1.7 s |
| judge gpt-5.4-mini, low effort | 100% | 1% | 0.82 | 1.5 s |
| rules + judge gpt-5-mini low | 100% | 1% | 0.52 | 2.8 s |

Rules catch every number, technology and role change and nothing else: scale, outcome and
subtle claims are words, invisible to them. The judge catches all six kinds, and on this set
adding the rules contributes no extra recall while adding one false alarm.

## Decision

Run the rule checks first, then the LLM judge (gpt-5-mini, low reasoning effort, prompt
`judge/v1`). A bullet passes only if both pass. If the rules already fail a bullet, skip the
judge for that attempt: the rewrite goes back with the rule issues.

Why keep the rules although the judge alone scored as well here:

1. **Guarantees that do not drift.** A changed number is the costliest fabrication in an
   interview. The rule check catches it every time, whatever model or prompt the judge runs
   with next month; the judge's 100% is one run of one model on one set.
2. **Exact feedback for the rewrite loop.** A rule issue names the number, technology or verb
   to fix, so the retry is targeted.
3. **Free and instant.** Failing rewrites skip a paid judge call.
4. **Regression tests in CI.** The rule layer runs on every commit without an API key; the
   judge cannot.

## Consequences

- One known rule false alarm ("Was on a team that created ..." read as a role claim) costs
  about 1% false alarms; it is left unfixed against this set to avoid tuning on test data.
- The set is now too easy to tell good judges apart (three configs at or near 100%). The
  hand-written subtle cases were the only ones that separated minimal from low effort. A
  harder set, ideally written by someone other than the prompt author, is needed before
  trusting small differences.
- Cost: about $0.0005 per judged bullet, a few cents per CV.
- Revisit if a harder set shows the rules adding recall (keep), or if rule false alarms grow
  with real use (narrow the rules to numbers and technologies only).
