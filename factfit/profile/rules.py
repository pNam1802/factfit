"""Rules that look across the whole profile (PRD §7.1).

1. Every id is unique in the file.
2. Every skill points to at least one bullet that proves it, and those bullets exist.
3. Every number written in a text is backed by a verified metric of the same item.

`check_rules` returns all problems instead of stopping at the first one.
"""

from collections.abc import Iterator
from dataclasses import dataclass

from factfit.schemas.profile import SECTIONS, Bullet, Metric, Profile, SummaryVariant
from factfit.text.numbers import Lang, find_numbers

Path = tuple[str | int, ...]


@dataclass
class Issue:
    path: Path
    message: str
    line: int | None = None

    def format(self, file: str = "") -> str:
        if file:
            where = f"{file}:{self.line}" if self.line else file
        else:
            where = f"line {self.line}" if self.line else ""
        dotted = format_path(self.path)
        prefix = ": ".join(part for part in (where, dotted) if part)
        return f"{prefix}: {self.message}" if prefix else self.message


def format_path(path: Path) -> str:
    out = ""
    for part in path:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else part)
    return out


def check_rules(profile: Profile) -> list[Issue]:
    return [*_unique_ids(profile), *_skill_evidence(profile), *_numbers_backed(profile)]


# --- walking the profile -------------------------------------------------------------------


def _items_with_text(profile: Profile) -> Iterator[tuple[Path, Bullet | SummaryVariant]]:
    for i, summary in enumerate(profile.summary_variants):
        yield ("summary_variants", i), summary
    for section in SECTIONS:
        for i, entry in enumerate(getattr(profile, section)):
            for j, bullet in enumerate(entry.bullets):
                yield (section, i, "bullets", j), bullet


def _all_ids(profile: Profile) -> Iterator[tuple[Path, str]]:
    for i, summary in enumerate(profile.summary_variants):
        yield ("summary_variants", i, "id"), summary.id
    for section in SECTIONS:
        for i, entry in enumerate(getattr(profile, section)):
            yield (section, i, "id"), entry.id
            for j, bullet in enumerate(entry.bullets):
                yield (section, i, "bullets", j, "id"), bullet.id


# --- rule 1 ----------------------------------------------------------------------------------


def _unique_ids(profile: Profile) -> Iterator[Issue]:
    first_seen: dict[str, Path] = {}
    for path, id_ in _all_ids(profile):
        if id_ in first_seen:
            yield Issue(
                path, f"duplicate id '{id_}' (first used at {format_path(first_seen[id_])})"
            )
        else:
            first_seen[id_] = path


# --- rule 2 ----------------------------------------------------------------------------------


def _skill_evidence(profile: Profile) -> Iterator[Issue]:
    bullet_ids = {b.id for _, b in _items_with_text(profile) if isinstance(b, Bullet)}
    for i, skill in enumerate(profile.skills):
        for j, ref in enumerate(skill.evidence):
            if ref not in bullet_ids:
                yield Issue(
                    ("skills", i, "evidence", j),
                    f"skill '{skill.name}' cites '{ref}', which is not a bullet id in this profile",
                )


# --- rule 3 ----------------------------------------------------------------------------------


def _numbers_backed(profile: Profile) -> Iterator[Issue]:
    global_names = [s.name for s in profile.skills]
    for path, item in _items_with_text(profile):
        mask = global_names + (item.skills if isinstance(item, Bullet) else [])
        texts: list[tuple[str, str | None, Lang]] = [
            ("text", item.text, "en"),
            ("text_vi", item.text_vi, "vi"),
        ]
        for field, text, lang in texts:
            if text:
                yield from _check_text(path + (field,), text, lang, item.metrics, mask)


def _check_text(
    path: Path, text: str, lang: Lang, metrics: list[Metric], mask: list[str]
) -> Iterator[Issue]:
    for mention in find_numbers(text, lang=lang, mask=mask):
        matching = [m for m in metrics if _same(m.value, mention.values)]
        if not matching:
            yield Issue(
                path,
                f"number '{mention.raw}' is not in this item's metrics. Add it under metrics "
                "(with verified: true once checked), or, if it is part of a technology name "
                "such as 'Python 3.11', list that name under skills.",
            )
        elif not any(m.verified for m in matching):
            yield Issue(
                path,
                f"number '{mention.raw}' matches metric '{matching[0].name}', which is not "
                "verified yet. Check it, then set verified: true.",
            )


def _same(value: float, candidates: frozenset[float]) -> bool:
    return any(abs(value - c) < 1e-9 for c in candidates)
