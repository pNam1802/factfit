"""ATS check: read the PDF back the way an applicant tracking system would (PRD F6).

A CV can look right and still read wrong to a machine: broken ligatures, text in the wrong
order, glyphs without a Unicode mapping. After compiling, extract the text with two
different extractors (pypdf and pdfminer.six; real ATS products differ too) and check:

1. no character is broken (ligature glyphs, replacement characters, "(cid:..)" leftovers);
2. every bullet, the summary and every listed skill can be read back;
3. sections come in the order the template prints them;
4. every JD keyword the CV content contains is still found in the PDF text.

Each check runs on both extractors; any failure is an issue that blocks export.
Comparisons ignore whitespace, hyphens and quote / dash styles, because LaTeX breaks lines,
hyphenates long words and typesets quotes. A keyword that one extractor splits inside a
word ("T raffic") is reported as a warning: it depends on the extractor, not the PDF.

Found with this check: Latin Modern's "ff" ligature was read as U+FB00 ("EﬀicientNet"), so
ligatures are switched off in the template for XeTeX builds.
"""

import io
import re
from collections.abc import Callable
from contextlib import redirect_stderr
from dataclasses import dataclass, field
from pathlib import Path

from pdfminer.high_level import extract_text as pdfminer_text
from pypdf import PdfReader

from factfit.render.latex import strip_links
from factfit.schemas.profile import Profile
from factfit.schemas.tailored import TailoredCV

LIGATURES = "ﬀﬁﬂﬃﬄﬅﬆ"  # ﬀ ﬁ ﬂ ﬃ ﬄ ﬅ ﬆ
_BROKEN = re.compile(rf"[{LIGATURES}�]|\(cid:\d+\)")
_FOLD = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "`": "'",
        "∼": "~",
        "–": "-",
        "—": "-",
        " ": " ",
    }  # fmt: skip
)
SECTION_ORDER = [
    "Summary",
    "Technical Skills",
    "Experience",
    "Projects",
    "Education",
    "Certifications",
]


def _pypdf(pdf_path: Path) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(pdf_path).pages)


def _pdfminer(pdf_path: Path) -> str:
    with redirect_stderr(io.StringIO()):  # pdfminer is chatty about harmless font details
        return pdfminer_text(str(pdf_path))


EXTRACTORS: dict[str, Callable[[Path], str]] = {"pypdf": _pypdf, "pdfminer": _pdfminer}


@dataclass
class AtsReport:
    texts: dict[str, str]  # extractor name -> extracted text
    issues: list[str] = field(default_factory=list)  # block export
    warnings: list[str] = field(default_factory=list)  # shown, do not block

    @property
    def ok(self) -> bool:
        return not self.issues


def normalize(text: str) -> str:
    """For comparison only: fold quotes and dashes, then drop case, spaces and hyphens."""
    text = text.translate(_FOLD).casefold()
    return re.sub(r"[\s\-]+", "", text).replace("''", '"')


def extract_text(pdf_path: Path, extractor: str = "pypdf") -> str:
    return EXTRACTORS[extractor](pdf_path)


def check_pdf(
    pdf_path: Path, profile: Profile, cv: TailoredCV, jd_keywords: list[str] | None = None
) -> AtsReport:
    report = AtsReport(texts={name: fn(pdf_path) for name, fn in EXTRACTORS.items()})
    entries = {e.id: e for e in [*profile.experiences, *profile.projects]}
    headings = []  # roles, organisations, project names and their tech lines
    for s in cv.sections:
        e = entries[s.ref]
        headings += [getattr(e, a, None) or "" for a in ("role", "org", "name")]
        headings += getattr(e, "tech", [])
    content = normalize(
        " ".join(strip_links(b.text) for s in cv.sections for b in s.bullets)
        + " ".join(cv.skills)
        + " ".join(headings)
        + (cv.summary.text if cv.summary else "")
    )
    expected_sections = [s for s in SECTION_ORDER if _section_present(s, profile, cv)]

    for name, text in report.texts.items():
        flat = normalize(text)
        problems: list[str] = []

        broken = sorted(set(_BROKEN.findall(text)))
        if broken:
            problems.append(f"broken characters in the PDF text: {broken}")
        for section in cv.sections:
            for b in section.bullets:
                plain = strip_links(b.text)
                if normalize(plain) not in flat:
                    problems.append(f"bullet not readable as written: {plain[:80]}")
        if cv.summary and normalize(cv.summary.text) not in flat:
            problems.append("summary not readable as written")
        for skill in cv.skills:
            if normalize(skill) not in flat:
                problems.append(f"skill not found in PDF text: {skill}")

        positions = [text.find(s) for s in expected_sections]
        missing = [s for s, p in zip(expected_sections, positions, strict=True) if p < 0]
        if missing:
            problems.append(f"section headings not found: {missing}")
        else:
            found = [s for _, s in sorted(zip(positions, expected_sections, strict=True))]
            if found != expected_sections:
                problems.append(f"sections read in the wrong order: {found}")

        words = _words(text)
        for kw in jd_keywords or []:
            if normalize(kw) not in content:
                continue  # the CV does not claim it; nothing to look for
            if normalize(kw) not in flat:
                problems.append(f"JD keyword in the CV content but not in the PDF text: {kw}")
            elif _words(kw) and not _contains_words(words, _words(kw)):
                report.warnings.append(f"{name} reads keyword '{kw}' split inside a word")

        report.issues += [f"[{name}] {p}" for p in problems]
    return report


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.translate(_FOLD).casefold())


def _contains_words(words: list[str], target: list[str]) -> bool:
    n = len(target)
    return any(words[i : i + n] == target for i in range(len(words) - n + 1))


def _section_present(name: str, profile: Profile, cv: TailoredCV) -> bool:
    return {
        "Summary": cv.summary is not None,
        "Technical Skills": bool(cv.skills),
        "Experience": any(s.type == "experience" for s in cv.sections),
        "Projects": any(s.type == "project" for s in cv.sections),
        "Education": bool(profile.education),
        "Certifications": bool(profile.certifications),
    }[name]
