"""Find the numeric claims in a piece of text.

A "numeric claim" is a number a reader would take as a fact about your work:
"25 fps", "cut latency by 40%", "1,000 images". Numbers that are only part of a
name are not claims and are ignored:

- glued to a word before it:  YOLOv8, ResNet50, GPT-4o
- glued to letters after it that are not a unit:  3D, 4o
- inside a name the caller asks us to mask:  "Python 3.11"
- a year in a date context:  "in 2025", "Mar 2026", "(2024)"

Used for profile validation now and for the grounding check later (PRD §8.2).
"""

import re
from dataclasses import dataclass
from typing import Literal

Lang = Literal["en", "vi"]

# Units that may be glued to a number ("25fps", "120ms", "3x").
# Anything else glued to a number ("3D") is treated as part of a name.
TIME_UNITS = {"ms", "s", "sec", "secs", "min", "mins", "h", "hr", "hrs"}
SIZE_UNITS = {"kb", "mb", "gb", "tb"}
UNITS = {"%", "fps", "x", "k", "m", "b"} | TIME_UNITS | SIZE_UNITS
# Case-sensitive on purpose: "5M users" is a million, "5m" is minutes or metres.
MULTIPLIERS = {"k": 1e3, "K": 1e3, "M": 1e6, "B": 1e9}

_NUMBER = re.compile(
    r"(?<![\w.,/])"  # not glued to a preceding word, decimal point or slash
    r"(?<![A-Za-z]-)"  # not the "4" in "GPT-4"
    r"(?P<num>\d+(?:[.,]\d+)*)"
    r"(?P<suffix>%|[A-Za-z]+)?"
)

_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"
_YEAR_CONTEXT = re.compile(
    rf"(?:\b(?:in|since|from|until|to|năm|tháng|(?:{_MONTHS})[a-z]*\.?)\s+|\(|/)$",
    re.IGNORECASE,
)

_MASK = "⁣"  # invisible separator; never matched by _NUMBER


@dataclass(frozen=True)
class NumberMention:
    raw: str  # exactly as written, e.g. "1k", "25fps"
    values: frozenset[float]  # every value this mention may stand for, e.g. {1.0, 1000.0}


_VIETNAMESE = re.compile("[ăâđêôơưạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]", re.IGNORECASE)


def guess_lang(text: str) -> Lang:
    """ "vi" if the text uses letters only Vietnamese has, else "en"."""
    return "vi" if _VIETNAMESE.search(text) else "en"


def parse_number(raw: str, lang: Lang = "en") -> float:
    """Turn "1,000" / "0,85" / "1.000" / "3.11" into a float.

    English uses "," for thousands and "." for decimals; Vietnamese is the reverse.
    When a separator could be either, groups of exactly 3 digits mean thousands.
    """
    has_comma, has_dot = "," in raw, "." in raw
    if has_comma and has_dot:
        decimal = "," if raw.rfind(",") > raw.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        return float(raw.replace(thousands, "").replace(decimal, "."))

    sep = "," if has_comma else "." if has_dot else None
    if sep is None:
        return float(raw)

    groups = raw.split(sep)
    looks_like_thousands = all(len(g) == 3 for g in groups[1:])
    sep_is_thousands = (sep == ",") if lang == "en" else (sep == ".")
    if looks_like_thousands and (sep_is_thousands or len(groups) > 2):
        return float("".join(groups))
    return float(raw.replace(sep, "."))


def find_numbers(
    text: str, *, lang: Lang = "en", mask: list[str] | tuple[str, ...] = ()
) -> list[NumberMention]:
    """Return the numeric claims in ``text``.

    ``mask`` lists names whose digits must be ignored, e.g. ["Python 3.11"].
    """
    for name in sorted(mask, key=len, reverse=True):
        if any(c.isdigit() for c in name):
            text = re.sub(re.escape(name), _MASK * len(name), text, flags=re.IGNORECASE)

    mentions = []
    for m in _NUMBER.finditer(text):
        raw_num, suffix = m.group("num"), m.group("suffix")
        if suffix and suffix.lower() not in UNITS:
            continue  # part of a name, e.g. "3D"
        value = parse_number(raw_num, lang)
        if _is_year(raw_num, value, text[: m.start()]):
            continue
        values = {value}
        if suffix in MULTIPLIERS:
            values.add(value * MULTIPLIERS[suffix])
        mentions.append(NumberMention(raw=m.group(0), values=frozenset(values)))
    return mentions


def _is_year(raw: str, value: float, before: str) -> bool:
    return (
        len(raw) == 4
        and raw.isdigit()
        and 1950 <= value <= 2100
        and _YEAR_CONTEXT.search(before) is not None
    )
