"""Turn profile text into safe LaTeX (PRD F6).

All escaping happens here, in Python; the template only places finished strings. A bullet
may contain a link written as [text](https://...), which becomes \\href{url}{text}.
"""

import re

# One pass over the text, so a replacement is never escaped again: replacing "\" first and
# then "{" would break the "\textbackslash{}" just inserted.
_REPLACEMENTS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "^": r"\textasciicircum{}",
    "~": r"$\sim$",  # "~95%" means "about 95%"
    "∼": r"$\sim$",
    "→": r"$\rightarrow$",
    "←": r"$\leftarrow$",
    "|": r"$|$",
    "<": r"$<$",
    ">": r"$>$",
    "“": "``",
    "”": "''",
    "‘": "`",
    "’": "'",
    "–": "--",
    "—": "---",
    " ": "~",  # no-break space
}
_PATTERN = re.compile("|".join(re.escape(k) for k in _REPLACEMENTS))
_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")


def tex_escape(text: str) -> str:
    """Escape plain text for LaTeX. Links in [text](url) form become \\href."""
    out, pos = [], 0
    for m in _LINK.finditer(text):
        out.append(_escape_plain(text[pos : m.start()]))
        out.append(rf"\href{{{tex_url(m.group(2))}}}{{{_escape_plain(m.group(1))}}}")
        pos = m.end()
    out.append(_escape_plain(text[pos:]))
    return "".join(out)


def _escape_plain(text: str) -> str:
    # Keep a number with its unit on one line ("512 MB", "25 fps"): a no-break space.
    text = _NUMBER_UNIT.sub("\\1\u00a0\\2", text)
    return _PATTERN.sub(lambda m: _REPLACEMENTS[m.group(0)], text)


_NUMBER_UNIT = re.compile(r"(\d) (KB|MB|GB|TB|ms|s|min|fps|k|M|x)\b")


def tex_url(url: str) -> str:
    """Inside \\href{...}, only % and # need escaping."""
    return url.replace("\\", "/").replace("%", r"\%").replace("#", r"\#")


def strip_links(text: str) -> str:
    """[demo](https://...) -> demo, for checks that compare plain text."""
    return _LINK.sub(lambda m: m.group(1), text)


_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def format_month(value: str | None) -> str:
    """'2026-09' -> 'Sep 2026', '2026' -> '2026', 'present' -> 'Present'."""
    if not value:
        return ""
    if value == "present":
        return "Present"
    if len(value) == 7:
        return f"{_MONTHS[int(value[5:7]) - 1]} {value[:4]}"
    return value


def format_range(start: str | None, end: str | None) -> str:
    """'Sep 2026 -- Present'; one side alone when the other is missing."""
    parts = [p for p in (format_month(start), format_month(end)) if p]
    return " -- ".join(parts)
