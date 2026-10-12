"""The only door between factfit and an LLM.

    client = make_client()
    jd = client.parse(JobDescription, node="parse_jd", prompt_version="parse_jd/v1",
                      system=SYSTEM_PROMPT, user=raw_jd_text)

Every call:
- asks for JSON matching the Pydantic schema (Structured Outputs),
- validates it with that schema, including our own rules (e.g. unique requirement ids),
- retries on network errors and on invalid output, telling the model what was wrong,
- logs every attempt, failed or not, to the `llm_calls` table.
"""

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError
from sqlalchemy import Engine
from sqlmodel import Session

from factfit.db.models import LLMCall
from factfit.llm.backends import (
    Backend,
    DeepSeekBackend,
    MockBackend,
    OpenAIBackend,
    RecordingBackend,
    TransientLLMError,
)
from factfit.llm.config import DEFAULT_CONFIG, LLMConfig, load_llm_config

T = TypeVar("T", bound=BaseModel)
Logger = Callable[[LLMCall], None]

RETRY_NOTE = (
    "\n\nYour previous answer was rejected because it did not pass validation:\n"
    "{errors}\nReturn a corrected answer that fixes these problems."
)


class LLMError(Exception):
    """The call failed for good: retries are used up, or retrying cannot help."""


class LLMClient:
    def __init__(
        self,
        backend: Backend,
        config: LLMConfig,
        log: Logger | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.backend = backend
        self.config = config
        self.log = log or (lambda call: None)
        self.sleep = sleep

    def parse(
        self,
        schema: type[T],
        *,
        node: str,
        prompt_version: str,
        system: str,
        user: str,
        run_id: str | None = None,
    ) -> T:
        model = self.config.model_for(node)
        options = self.config.api_options(node)
        attempts = 1 + self.config.max_retries
        prompt = user
        last_error = ""

        for attempt in range(attempts):
            call = LLMCall(run_id=run_id, node=node, model=model, prompt_version=prompt_version)
            start = time.perf_counter()
            try:
                raw = self.backend.complete(
                    node=node,
                    model=model,
                    system=system,
                    user=prompt,
                    schema=schema,
                    options=options,
                )
            except TransientLLMError as e:
                last_error = str(e)
                self._finish(call, start, error=last_error)
                if attempt < attempts - 1:
                    self.sleep(2**attempt)  # wait 1s, 2s, 4s ... before the next try
                continue
            except Exception as e:
                # Wrong API key, unknown model, bad request: trying again will not help.
                self._finish(call, start, error=f"{type(e).__name__}: {e}")
                raise LLMError(f"{node}: {type(e).__name__}: {e}") from e

            call.tokens_in, call.tokens_out = raw.tokens_in, raw.tokens_out
            call.cost_usd = self.config.cost(model, raw.tokens_in, raw.tokens_out)
            try:
                result = schema.model_validate_json(raw.text)
            except ValidationError as e:
                last_error = _summarise(e)
                self._finish(call, start, error=f"invalid output: {last_error}")
                prompt = user + RETRY_NOTE.format(errors=last_error)
                continue

            self._finish(call, start)
            return result

        raise LLMError(
            f"{node}: no valid answer after {attempts} attempts. Last error: {last_error}"
        )

    def _finish(self, call: LLMCall, start: float, error: str | None = None) -> None:
        call.latency_ms = round((time.perf_counter() - start) * 1000)
        call.ok = error is None
        call.error = error
        self.log(call)


def _summarise(error: ValidationError, limit: int = 10) -> str:
    lines = []
    for err in error.errors()[:limit]:
        where = ".".join(str(p) for p in err["loc"]) or "(root)"
        lines.append(f"- {where}: {err['msg']}")
    return "\n".join(lines)


def db_logger(engine: Engine) -> Logger:
    """Write each call to the `llm_calls` table."""

    def log(call: LLMCall) -> None:
        with Session(engine) as session:
            session.add(call)
            session.commit()

    return log


# Older .env files named the provider as the mode; the provider now lives in config/llm.yaml.
LIVE_ALIASES = ("openai", "deepseek")

KEY_VARS = {"openai": "OPENAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}


def make_client(
    mode: str | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    engine: Engine | None = None,
) -> LLMClient:
    """Build a client from config/llm.yaml and .env.

    mode (or env FACTFIT_LLM_MODE): "live" (default), "mock" or "record".
    "live" and "record" call the `provider` set in config/llm.yaml.
    """
    load_dotenv()
    mode = mode or os.environ.get("FACTFIT_LLM_MODE") or "live"
    if mode in LIVE_ALIASES:
        mode = "live"
    config = load_llm_config(config_path)

    if mode == "mock":
        backend: Backend = MockBackend()
    elif mode in ("live", "record"):
        backend = _live_backend(config)
        if mode == "record":
            backend = RecordingBackend(backend)
    else:
        raise LLMError(f"unknown FACTFIT_LLM_MODE '{mode}': use live, mock or record")

    return LLMClient(backend, config, log=db_logger(engine) if engine else None)


def _live_backend(config: LLMConfig) -> Backend:
    key_var = KEY_VARS[config.provider]
    key = os.environ.get(key_var)
    if not key:
        raise LLMError(
            f"{key_var} is not set (provider '{config.provider}' in config/llm.yaml). "
            "Copy .env.example to .env and add your key, or use FACTFIT_LLM_MODE=mock."
        )
    if config.provider == "deepseek":
        return DeepSeekBackend(timeout_s=config.timeout_s, api_key=key)
    return OpenAIBackend(timeout_s=config.timeout_s)
