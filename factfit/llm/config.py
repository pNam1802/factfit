"""Load `config/llm.yaml`: which model each node uses and what each model costs."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from factfit.schemas.base import Strict

DEFAULT_CONFIG = Path("config/llm.yaml")


class Price(Strict):
    input: float = Field(ge=0)  # USD per 1M input tokens
    output: float = Field(ge=0)  # USD per 1M output tokens


class NodeSettings(Strict):
    model: str  # an alias from `models` or a model name
    reasoning_effort: Literal["minimal", "low", "medium", "high"] | None = None


class LLMConfig(Strict):
    models: dict[str, str]
    nodes: dict[str, NodeSettings]
    pricing: dict[str, Price] = {}
    max_retries: int = Field(default=2, ge=0, le=5)
    timeout_s: float = 60

    def model_for(self, node: str) -> str:
        if node not in self.nodes:
            raise KeyError(f"node '{node}' has no entry under `nodes` in config/llm.yaml")
        name = self.nodes[node].model
        return self.models.get(name, name)

    def api_options(self, node: str) -> dict:
        """Extra arguments for the API call, only those that are set."""
        effort = self.nodes[node].reasoning_effort
        return {"reasoning": {"effort": effort}} if effort else {}

    def cost(self, model: str, tokens_in: int, tokens_out: int) -> float | None:
        price = self.pricing.get(model)
        if price is None:
            return None
        return (tokens_in * price.input + tokens_out * price.output) / 1_000_000


def load_llm_config(path: str | Path = DEFAULT_CONFIG) -> LLMConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return LLMConfig.model_validate(data)
