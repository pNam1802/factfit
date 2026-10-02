"""Load versioned prompts from `prompts/<node>/<version>.md`.

A prompt file has a system part, then a line `=== USER ===`, then the user message template.
The template uses `$name` placeholders (Python string.Template), so JSON braces in prompts
need no escaping. Prompts are never edited in place once used: a change is a new version
file, so every stored result can be traced back to the exact prompt that produced it.
"""

from dataclasses import dataclass
from pathlib import Path
from string import Template

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
USER_MARKER = "=== USER ==="


@dataclass(frozen=True)
class Prompt:
    node: str
    version: str
    system: str
    user_template: str

    @property
    def id(self) -> str:
        """Stored in llm_calls and next to results, e.g. "parse_jd/v1"."""
        return f"{self.node}/{self.version}"

    def render(self, **values: str) -> str:
        return Template(self.user_template).substitute(values)


def load_prompt(node: str, version: str, root: Path | None = None) -> Prompt:
    path = (root or PROMPTS_DIR) / node / f"{version}.md"
    text = path.read_text(encoding="utf-8")
    if USER_MARKER not in text:
        raise ValueError(f"{path}: missing the '{USER_MARKER}' line between system and user parts")
    system, user = text.split(USER_MARKER, 1)
    return Prompt(node=node, version=version, system=system.strip(), user_template=user.strip())
