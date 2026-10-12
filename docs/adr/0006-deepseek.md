# 0006. Switch the default provider to DeepSeek, keep OpenAI as an option

- **Status:** accepted (supersedes the provider choice in 0005; the rest of 0005 still holds)
- **Date:** 2026-10-12

## Context

The OpenAI credit for this project ran out in week 5. A DeepSeek key was available. DeepSeek
serves an OpenAI-compatible Chat Completions API, so the same `openai` SDK works with a
different `base_url`, but it differs in two ways that matter here:

- **No Structured Outputs.** Only JSON mode (`response_format: json_object`): the answer is
  valid JSON, but nothing holds it to our schema.
- **No Responses API.** Only Chat Completions. Reasoning is set with `reasoning_effort`
  (low / high / max) and thinking can be turned off with `thinking: {type: disabled}`.

Options: wait for OpenAI credit (blocks the week-5 evals and the first real application),
a multi-provider framework (rejected in 0005 for the same reasons), or one more backend.

## Decision

Add a `DeepSeekBackend` next to `OpenAIBackend` and choose between them with
`provider: openai | deepseek` in `config/llm.yaml`, next to the model names and prices, which
are provider-specific anyway. `FACTFIT_LLM_MODE` is now `live | mock | record`.

The JSON Schema of the Pydantic model goes into the system prompt. Answers that do not match it
fail Pydantic validation and are retried with the errors, through the loop `LLMClient` already
has. Models: `deepseek-flash` (small) and `deepseek-v4-pro` (large). The effort `minimal`
turns thinking off; other efforts are passed through.

## Evidence so far

`factfit llm-check` (2026-10-12): `deepseek-flash`, thinking off, 1.0 s, $0.00004.

`factfit eval-parse --prompts v2 --efforts low`, 4 dev JDs:

| Model | Precision | Recall | any_of | $/JD | s/JD | Failed |
| --- | --- | --- | --- | --- | --- | --- |
| gpt-5-mini (2026-10-05) | 97% | 96% | 68% | 0.0051 | 22.8 | 0 |
| deepseek-flash (2026-10-12) | 98% | 87% | 71% | 0.0070* | 19.6 | 0 |

\* priced at peak, cache-miss rates, so an upper bound. The alignment judge of the eval also
moved to `deepseek-flash`, so the two rows are not measured by the same judge.

## Consequences

- No node, prompt or schema changed; only the backend and config did, as 0005 expected.
- Schema adherence is now enforced by validation and retries, not by the API: watch the
  `fail` and retry counts in evals and in `llm_calls`.
- Prompts and efforts were tuned on gpt-5 / gpt-5-mini. Parse recall already dropped
  (96% -> 87%); the grounding judge must be re-measured with `factfit eval-grounding` before
  its results are trusted, since it is the layer that stops invented claims.
- Profile and JD text now go to DeepSeek's servers instead of OpenAI's.
- Switching back is one line (`provider: openai` plus the old model names).
