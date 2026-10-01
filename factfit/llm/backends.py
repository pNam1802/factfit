"""Backends do the actual request. `LLMClient` adds retries, validation and logging on top.

- OpenAIBackend:    calls the OpenAI Responses API with Structured Outputs.
- MockBackend:      answers from saved fixtures; no network, no cost. For tests and demos.
- RecordingBackend: calls a real backend and saves each answer as a fixture for MockBackend.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

DEFAULT_FIXTURES = Path("evals/fixtures")


@dataclass
class RawOutput:
    text: str  # the model's JSON, not yet validated
    tokens_in: int = 0
    tokens_out: int = 0


class TransientLLMError(Exception):
    """A failure worth retrying: network error, timeout, rate limit, server error."""


class Backend(Protocol):
    def complete(
        self,
        *,
        node: str,
        model: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        options: dict[str, Any],
    ) -> RawOutput: ...


class OpenAIBackend:
    def __init__(self, client: Any = None, timeout_s: float = 60):
        if client is None:
            from openai import OpenAI

            # The SDK's own retries are turned off: LLMClient retries and logs every attempt.
            client = OpenAI(timeout=timeout_s, max_retries=0)
        self.client = client

    def complete(self, *, node, model, system, user, schema, options) -> RawOutput:
        import openai
        from openai.lib._pydantic import to_strict_json_schema

        try:
            response = self.client.responses.create(
                model=model,
                instructions=system,
                input=user,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema.__name__,
                        "schema": to_strict_json_schema(schema),
                        "strict": True,
                    }
                },
                **options,
            )
        except (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError) as e:
            raise TransientLLMError(f"{type(e).__name__}: {e}") from e

        usage = response.usage
        return RawOutput(
            text=response.output_text,
            tokens_in=usage.input_tokens if usage else 0,
            tokens_out=usage.output_tokens if usage else 0,
        )


def fixture_path(root: Path, node: str, system: str, user: str) -> Path:
    """Same prompt -> same file, so a recorded answer can be found again."""
    key = hashlib.sha256(f"{system}\n---\n{user}".encode()).hexdigest()[:16]
    return root / node / f"{key}.json"


class MockBackend:
    def __init__(self, fixtures: Path = DEFAULT_FIXTURES):
        self.fixtures = Path(fixtures)

    def complete(self, *, node, model, system, user, schema, options) -> RawOutput:
        path = fixture_path(self.fixtures, node, system, user)
        if not path.exists():
            raise FileNotFoundError(
                f"no fixture for this prompt at {path}. Record one first with "
                "FACTFIT_LLM_MODE=record (needs an API key)."
            )
        return RawOutput(text=json.loads(path.read_text(encoding="utf-8"))["output"])


class RecordingBackend:
    def __init__(self, inner: Backend, fixtures: Path = DEFAULT_FIXTURES):
        self.inner = inner
        self.fixtures = Path(fixtures)

    def complete(self, *, node, model, system, user, schema, options) -> RawOutput:
        raw = self.inner.complete(
            node=node, model=model, system=system, user=user, schema=schema, options=options
        )
        path = fixture_path(self.fixtures, node, system, user)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"node": node, "model": model, "schema": schema.__name__, "output": raw.text}
        path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return raw
