# 0005. Use the OpenAI API with Structured Outputs behind our own LLMClient

- **Status:** accepted (model choice per node to be revisited after the model comparison, PRD §10.1)
- **Date:** 2026-10-01

## Context

Every LLM step in factfit must return data that code can check: a parsed job description,
a match result, rewritten bullets with source ids, a judge verdict. Free text that is "usually
JSON" is not good enough, because the grounding check runs on these fields.

Other needs:

- a cheap model for frequent calls (parse, match, judge) and a stronger one for rewriting;
- cost per tailoring run under $0.05;
- every call logged (tokens, cost, latency, errors) for debugging and for the project's metrics;
- tests and a public demo that run without an API key.

## Decision

1. **Provider: OpenAI**, through the Responses API with **Structured Outputs**
   (`text.format = json_schema`, `strict: true`). The JSON schema is generated from the same
   Pydantic models the rest of the code uses, so there is one definition of each shape.
2. **Our own thin `LLMClient`** is the only place that talks to the provider. It:
   - validates the answer with the Pydantic model, which also runs our own rules that JSON
     Schema cannot express (e.g. unique requirement ids, "met needs evidence");
   - retries network errors with backoff, and invalid answers with the validation errors sent
     back to the model, at most 2 extra attempts;
   - logs every attempt, including failed ones, to `llm_calls`.
   The SDK's own retries are disabled so that every attempt is visible and priced.
3. **Models and prices live in `config/llm.yaml`**, per node, never in code.
   Defaults: `gpt-5-mini` for parse / match / judge, `gpt-5` for rewrite.
4. **Mock and record backends**: `record` saves real answers as fixtures, `mock` replays them,
   so CI and the Docker demo need no key.

Alternatives considered: other hosted providers (similar structured-output features; OpenAI
chosen for familiarity and because both a small and a large model come from one API), local
open-weight models (free per call but weaker structured output on available hardware; kept as a
candidate in the model comparison), and frameworks that wrap many providers (more abstraction
than needed, and they hide the per-attempt logging we want).

## Evidence so far

First real call (`factfit llm-check`, gpt-5-mini, 2026-10-01):

| Setting | Output tokens | Latency | Cost |
| --- | --- | --- | --- |
| default reasoning | 231 | 5.2 s | $0.000476 |
| `reasoning_effort: minimal` | 26 | 1.7 s | $0.000066 |

gpt-5 family models are reasoning models: hidden reasoning tokens are billed as output and add
latency. Reasoning effort is therefore a per-node setting, to be chosen with evals (week 2),
not guessed.

Estimated rewrite step (~3k tokens in, ~3k out): about $0.03 with `gpt-5` versus about $0.10
with `gpt-5.5`, which is why newer models are priced in the config but not the default.

## Consequences

- Malformed JSON is practically eliminated; what remains to handle is answers that are
  well-formed but break our rules, which the retry-with-errors loop addresses.
- Switching provider means writing one new backend; nodes and prompts do not change.
- Using `openai.lib._pydantic.to_strict_json_schema`, a private helper of the SDK, saves
  reimplementing OpenAI's strict-schema rules but may break on an SDK upgrade; the backend test
  checks the request shape so such a break is caught.
- Prices are copied by hand from public trackers (the official page blocks automated reads) and
  must be kept up to date; a model without a price logs `cost_usd = null` instead of a wrong number.
- Revisit after the model comparison: if a cheaper or local model reaches the grounding
  thresholds, change the defaults and record the result here.
