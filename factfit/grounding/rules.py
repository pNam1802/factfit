"""Grounding checks done by code (PRD §8.2, layers 1 and 2).

Given a rewritten bullet and the profile bullets it claims to come from, report anything the
rewrite ADDS that its sources do not support:

- number_unsupported: a number not in the sources' metrics or text
- tech_unsupported:   a technology (from data/aliases.yaml) not in the sources' text or skills
- role_inflated:      a role verb of a higher level (data/role_verbs.yaml) than the sources use

Leaving something out is fine: a shorter bullet is not a fabricated one. What code cannot
check (scale, business impact, technologies missing from the alias table) is left to the LLM
judge, layer 3.
"""

import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal

import yaml

from factfit.schemas.profile import Bullet
from factfit.text.numbers import find_numbers, guess_lang

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
# A source with no listed role verb is treated as plain hands-on work.
DEFAULT_SOURCE_LEVEL = 2
CAUSAL_VERBS = {"lead", "led", "leading"}  # ignored when followed by "to"
_NEXT_IS_TO = re.compile(r"\s+to\b", re.IGNORECASE)

IssueKind = Literal["no_source", "number_unsupported", "tech_unsupported", "role_inflated"]


@dataclass(frozen=True)
class GroundingIssue:
    kind: IssueKind
    detail: str  # the offending number, technology or verb
    message: str


class _Lexicon:
    """Finds phrases from a fixed list in text: case-insensitive, whole words, longest first."""

    def __init__(self, phrase_to_value: dict[str, object]):
        self.lookup = {p.casefold(): v for p, v in phrase_to_value.items()}
        alternatives = sorted(self.lookup, key=len, reverse=True)
        body = "|".join(re.escape(p) for p in alternatives) or r"(?!x)x"
        # \w on both sides would split "C++" or "Next.js"; these boundaries keep them whole.
        self.pattern = re.compile(rf"(?<![\w+#])({body})(?![\w+#])", re.IGNORECASE)

    def find(self, text: str) -> list[tuple[str, object]]:
        """Return (phrase as written, value) for every match."""
        return [
            (m.group(1), self.lookup[m.group(1).casefold()]) for m in self.pattern.finditer(text)
        ]


@dataclass(frozen=True)
class KnowledgeBase:
    tech: _Lexicon  # any spelling -> canonical name
    verbs: _Lexicon  # verb phrase -> level 1..4
    tech_names: tuple[str, ...]  # every spelling, used to mask version digits ("YOLOv8")

    def canonical(self, name: str) -> str:
        """Canonical technology name, or the name itself (casefolded) if not in the table."""
        return str(self.tech.lookup.get(name.casefold(), name.casefold()))


def load_kb(data_dir: Path = DATA_DIR) -> KnowledgeBase:
    return _load_kb_cached(str(data_dir))


@cache
def _load_kb_cached(data_dir: str) -> KnowledgeBase:
    root = Path(data_dir)
    aliases = yaml.safe_load((root / "aliases.yaml").read_text(encoding="utf-8")) or {}
    spellings: dict[str, object] = {}
    for canonical, others in aliases.items():
        for spelling in [canonical, *(others or [])]:
            spellings[str(spelling)] = str(canonical)

    levels = yaml.safe_load((root / "role_verbs.yaml").read_text(encoding="utf-8")) or {}
    verbs: dict[str, object] = {}
    for level, by_lang in levels.items():
        for phrases in by_lang.values():
            for phrase in phrases:
                verbs[str(phrase)] = int(level)

    return KnowledgeBase(
        tech=_Lexicon(spellings), verbs=_Lexicon(verbs), tech_names=tuple(spellings)
    )


# --- the checks ------------------------------------------------------------------------------


def check_bullet(
    new_text: str, sources: list[Bullet], kb: KnowledgeBase | None = None
) -> list[GroundingIssue]:
    kb = kb or load_kb()
    if not sources:
        return [GroundingIssue("no_source", "", "the bullet has no source bullet")]
    return [
        *check_numbers(new_text, sources, kb),
        *check_tech(new_text, sources, kb),
        *check_role(new_text, sources, kb),
    ]


def check_numbers(new_text: str, sources: list[Bullet], kb: KnowledgeBase) -> list[GroundingIssue]:
    mask = [*kb.tech_names, *(s for b in sources for s in b.skills)]
    allowed: set[float] = {m.value for b in sources for m in b.metrics}
    for b in sources:
        for mention in find_numbers(b.text, lang=guess_lang(b.text), mask=mask):
            allowed |= mention.values

    issues = []
    # "2.000" is two thousand in Vietnamese and two in English.
    for mention in find_numbers(new_text, lang=guess_lang(new_text), mask=mask):
        if not any(abs(v - a) < 1e-9 for v in mention.values for a in allowed):
            issues.append(
                GroundingIssue(
                    "number_unsupported",
                    mention.raw,
                    f"'{mention.raw}' is not in the source bullets' metrics or text",
                )
            )
    return issues


def check_tech(new_text: str, sources: list[Bullet], kb: KnowledgeBase) -> list[GroundingIssue]:
    in_sources = {str(canon) for b in sources for _, canon in kb.tech.find(b.text)}
    in_sources |= {kb.canonical(s) for b in sources for s in b.skills}

    issues, reported = [], set()
    for written, canon in kb.tech.find(new_text):
        if canon not in in_sources and canon not in reported:
            reported.add(canon)
            issues.append(
                GroundingIssue(
                    "tech_unsupported",
                    str(canon),
                    f"'{written}' does not appear in the source bullets",
                )
            )
    return issues


def role_level(text: str, kb: KnowledgeBase) -> tuple[str, int] | None:
    """The role a bullet claims: its first verb, raised by any level 3-4 verb anywhere.

    Taking every verb would misread descriptions as roles: in "Contributed to a tracker
    deployed with Docker" the role is "contributed", while "deployed" describes the tracker.
    "Designed" or "led" anywhere is a role claim, so those always count.
    """
    found = [
        (m.group(1), kb.verbs.lookup[m.group(1).casefold()])
        for m in kb.verbs.pattern.finditer(text)
        # "led to faster responses" states a cause, not leadership
        if not (m.group(1).casefold() in CAUSAL_VERBS and _NEXT_IS_TO.match(text, m.end()))
    ]
    if not found:
        return None
    claims = [found[0], *(pair for pair in found if pair[1] >= 3)]
    return max(claims, key=lambda pair: pair[1])


def check_role(new_text: str, sources: list[Bullet], kb: KnowledgeBase) -> list[GroundingIssue]:
    claim = role_level(new_text, kb)
    if claim is None:
        return []
    written, new_level = claim
    source_levels = [lvl for b in sources if (c := role_level(b.text, kb)) for lvl in [c[1]]]
    source_level = max(source_levels, default=DEFAULT_SOURCE_LEVEL)
    if new_level > source_level:
        return [
            GroundingIssue(
                "role_inflated",
                written,
                f"'{written}' (level {new_level}) claims a bigger role than the sources "
                f"(level {source_level})",
            )
        ]
    return []
