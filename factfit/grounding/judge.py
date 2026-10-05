"""Grounding layer 3: an LLM judge for claims code cannot check (PRD §8.2).

Scale, impact, inferred qualities and added work are written in words, so the rule layer
cannot see them. The judge lists each unsupported claim with a kind and a reason, which the
rewrite step can feed back to the model.
"""

from typing import Literal

from factfit.llm import LLMClient
from factfit.prompts import load_prompt
from factfit.schemas.base import Strict
from factfit.schemas.profile import Bullet

NODE = "judge"

ClaimKind = Literal["scale", "outcome", "inference", "extra_work", "role", "number", "technology"]


class UnsupportedClaim(Strict):
    claim: str  # the words in the rewrite that make the claim
    kind: ClaimKind
    why: str  # one short sentence


class JudgeVerdict(Strict):
    unsupported_claims: list[UnsupportedClaim]

    @property
    def grounded(self) -> bool:
        return not self.unsupported_claims


def judge_bullet(
    new_text: str, sources: list[Bullet], client: LLMClient, run_id: str | None = None
) -> JudgeVerdict:
    prompt = load_prompt(NODE, client.config.prompt_for(NODE))
    return client.parse(
        JudgeVerdict,
        node=NODE,
        prompt_version=prompt.id,
        system=prompt.system,
        user=prompt.render(
            sources="\n".join(_source_line(b, with_skills=prompt.version != "v1") for b in sources),
            rewrite=new_text,
        ),
        run_id=run_id,
    )


def _source_line(bullet: Bullet, with_skills: bool) -> str:
    # v1 saw the sentence only, and flagged technologies listed in the bullet's skills that
    # the rule layer accepts. v2 sees the skills too. v1 keeps its input so its measured
    # results stay reproducible.
    if with_skills and bullet.skills:
        return f"- {bullet.text} (skills used: {', '.join(bullet.skills)})"
    return f"- {bullet.text}"
