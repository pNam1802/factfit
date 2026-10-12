import json
from types import SimpleNamespace

import pytest

from factfit.db.models import LLMCall
from factfit.llm import LLMClient, LLMError
from factfit.llm.backends import (
    DeepSeekBackend,
    MockBackend,
    OpenAIBackend,
    RawOutput,
    RecordingBackend,
    TransientLLMError,
    deepseek_options,
)
from factfit.llm.client import make_client
from factfit.llm.config import LLMConfig
from factfit.schemas.job import JobDescription

CONFIG = LLMConfig.model_validate(
    {
        "models": {"small": "small-model"},
        "nodes": {"parse_jd": {"model": "small"}},
        "pricing": {"small-model": {"input": 1.0, "output": 2.0}},
        "max_retries": 2,
    }
)

VALID_JD = json.dumps(
    {
        "title": "AI Engineer",
        "company": None,
        "seniority": "junior",
        "language": "en",
        "location": None,
        "must_have": [
            {"id": "r1", "text": "Python", "category": "skill", "any_of": [], "level": None}
        ],
        "nice_to_have": [],
        "responsibilities": [],
        "keywords": [],
    }
)
# Valid JSON shape, but breaks our own rule: requirement ids must be unique.
DUPLICATE_IDS_JD = json.dumps(
    {
        **json.loads(VALID_JD),
        "nice_to_have": [
            {"id": "r1", "text": "Go", "category": "skill", "any_of": [], "level": None}
        ],
    }
)


class ScriptedBackend:
    """Plays back a list of answers; an Exception in the list is raised instead."""

    def __init__(self, *script):
        self.script = list(script)
        self.prompts: list[str] = []

    def complete(self, *, node, model, system, user, schema, options):
        self.prompts.append(user)
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        return RawOutput(text=step, tokens_in=1000, tokens_out=500)


def make(backend, config=CONFIG):
    calls: list[LLMCall] = []
    client = LLMClient(backend, config, log=calls.append, sleep=lambda s: None)
    return client, calls


def parse(client):
    return client.parse(
        JobDescription, node="parse_jd", prompt_version="parse_jd/v1", system="sys", user="JD text"
    )


def test_success_first_try_logs_cost():
    client, calls = make(ScriptedBackend(VALID_JD))
    assert parse(client).title == "AI Engineer"
    (call,) = calls
    assert call.ok and call.model == "small-model" and call.prompt_version == "parse_jd/v1"
    assert call.cost_usd == pytest.approx((1000 * 1.0 + 500 * 2.0) / 1_000_000)


def test_invalid_output_is_retried_with_the_error_shown_to_the_model():
    backend = ScriptedBackend(DUPLICATE_IDS_JD, VALID_JD)
    client, calls = make(backend)
    parse(client)
    assert [c.ok for c in calls] == [False, True]
    assert "repeated" in calls[0].error
    assert "rejected" in backend.prompts[1] and "repeated" in backend.prompts[1]


def test_broken_json_is_retried():
    client, calls = make(ScriptedBackend("{not json", VALID_JD))
    parse(client)
    assert [c.ok for c in calls] == [False, True]


def test_network_error_is_retried():
    client, calls = make(ScriptedBackend(TransientLLMError("timeout"), VALID_JD))
    parse(client)
    assert [c.ok for c in calls] == [False, True]


def test_gives_up_after_max_retries_and_logs_every_attempt():
    client, calls = make(ScriptedBackend("{bad", "{bad", "{bad"))
    with pytest.raises(LLMError, match="after 3 attempts"):
        parse(client)
    assert len(calls) == 3 and not any(c.ok for c in calls)


def test_non_retryable_error_stops_immediately():
    client, calls = make(ScriptedBackend(PermissionError("invalid api key"), VALID_JD))
    with pytest.raises(LLMError, match="invalid api key"):
        parse(client)
    assert len(calls) == 1


def test_model_without_price_logs_null_cost():
    config = CONFIG.model_copy(update={"pricing": {}})
    client, calls = make(ScriptedBackend(VALID_JD), config)
    parse(client)
    assert calls[0].cost_usd is None


def test_unknown_node_is_a_clear_error():
    client, _ = make(ScriptedBackend(VALID_JD))
    with pytest.raises(KeyError, match="rewrite"):
        client.parse(JobDescription, node="rewrite", prompt_version="v1", system="", user="")


def test_record_then_replay_with_mock(tmp_path):
    recorder = RecordingBackend(ScriptedBackend(VALID_JD), fixtures=tmp_path)
    client, _ = make(recorder)
    first = parse(client)

    replay, _ = make(MockBackend(fixtures=tmp_path))
    assert parse(replay) == first


def test_mock_without_fixture_explains_how_to_record(tmp_path):
    client, _ = make(MockBackend(fixtures=tmp_path))
    with pytest.raises(LLMError, match="FACTFIT_LLM_MODE=record"):
        parse(client)


def test_openai_backend_sends_a_strict_schema():
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        return SimpleNamespace(output_text=VALID_JD, usage=usage)

    fake_client = SimpleNamespace(responses=SimpleNamespace(create=create))
    raw = OpenAIBackend(client=fake_client).complete(
        node="parse_jd", model="m", system="sys", user="u", schema=JobDescription, options={}
    )
    fmt = sent["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    assert fmt["name"] == "JobDescription"
    assert fmt["schema"]["additionalProperties"] is False
    assert sent["instructions"] == "sys" and sent["input"] == "u"
    assert (raw.tokens_in, raw.tokens_out) == (10, 5)


def test_deepseek_backend_puts_the_schema_in_the_prompt_and_asks_for_json():
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        message = SimpleNamespace(content=VALID_JD)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    raw = DeepSeekBackend(client=fake_client).complete(
        node="parse_jd",
        model="m",
        system="sys",
        user="u",
        schema=JobDescription,
        options={"reasoning": {"effort": "low"}},
    )
    system, user = sent["messages"]
    assert system["role"] == "system" and system["content"].startswith("sys")
    assert "JSON Schema" in system["content"] and '"must_have"' in system["content"]
    assert user == {"role": "user", "content": "u"}
    assert sent["response_format"] == {"type": "json_object"}
    assert sent["reasoning_effort"] == "low"
    assert (raw.text, raw.tokens_in, raw.tokens_out) == (VALID_JD, 10, 5)


def test_deepseek_options_turn_thinking_off_for_minimal_effort():
    assert deepseek_options({}) == {}
    assert deepseek_options({"reasoning": {"effort": "high"}}) == {"reasoning_effort": "high"}
    assert deepseek_options({"reasoning": {"effort": "minimal"}}) == {
        "extra_body": {"thinking": {"type": "disabled"}}
    }


def test_live_mode_uses_the_provider_from_config_and_its_key(tmp_path, monkeypatch):
    config = tmp_path / "llm.yaml"
    config.write_text(
        "provider: deepseek\nmodels: {small: m}\nnodes: {parse_jd: {model: small}}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("factfit.llm.client.load_dotenv", lambda: None)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(LLMError, match="DEEPSEEK_API_KEY"):
        make_client(mode="live", config_path=config)

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    for mode in ("live", "deepseek", "openai"):  # old provider-named modes still mean live
        assert isinstance(make_client(mode=mode, config_path=config).backend, DeepSeekBackend)
