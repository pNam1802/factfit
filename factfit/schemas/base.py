from pydantic import BaseModel, ConfigDict


class Strict(BaseModel):
    """Base for every schema: unknown fields are an error, not silently dropped.

    For LLM output this also matters: OpenAI Structured Outputs requires
    `additionalProperties: false`, which this setting produces.
    """

    model_config = ConfigDict(extra="forbid")
