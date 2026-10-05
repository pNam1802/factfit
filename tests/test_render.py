import shutil
from pathlib import Path

import pytest
from pypdf import PdfReader

from factfit.profile import load_profile
from factfit.render.compile import compile_pdf, render_tex
from factfit.render.latex import format_month, format_range, strip_links, tex_escape
from factfit.render.view import build_view, full_profile_cv
from factfit.schemas.tailored import TailoredBullet, TailoredCV, TailoredSection

EXAMPLE = Path(__file__).parent.parent / "data" / "profile.example.yaml"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("R&D", r"R\&D"),
        ("50%", r"50\%"),
        ("$3", r"\$3"),
        ("#1", r"\#1"),
        ("snake_case", r"snake\_case"),
        ("{a}", r"\{a\}"),
        ("a\\b", r"a\textbackslash{}b"),  # not re-escaped into \textbackslash\{\}
        ("x^2", r"x\textasciicircum{}2"),
        ("~95%", r"$\sim$95\%"),
        ("A → B", r"A $\rightarrow$ B"),
        ("a | b", r"a $|$ b"),
        ("“AI Thuc Chien”", "``AI Thuc Chien''"),
        ("company’s", "company's"),
        ("2022–2026", "2022--2026"),
        ("512 MB", "512~MB"),  # number and unit stay on one line
        ("25 fps", "25~fps"),
    ],
)
def test_tex_escape(text, expected):
    assert tex_escape(text) == expected


def test_links_become_href_with_escaped_text_and_url():
    out = tex_escape("see [demo & docs](https://x.io/a%20b#top) now")
    assert out == r"see \href{https://x.io/a\%20b\#top}{demo \& docs} now"
    assert strip_links("the ([demo](https://x.io)) flow") == "the (demo) flow"


def test_dates():
    assert format_month("2026-09") == "Sep 2026"
    assert format_month("2026") == "2026"
    assert format_range("2026-09", "present") == "Sep 2026 -- Present"
    assert format_range(None, "2026-01") == "Jan 2026"


def test_view_from_example_profile():
    profile = load_profile(EXAMPLE)
    view = build_view(profile, full_profile_cv(profile))
    assert r"\href{mailto:a@example.com}" in view.contact
    assert view.headline == "" and view.name == "Nguyen Van A"
    project = view.projects[0]
    # no `tech` given: the project's bullet skills are shown, without duplicates
    assert project.title.endswith(r"\emph{Python, LangChain, Qdrant}")
    assert [h for h, _ in view.skill_groups] == ["Skills"]


def test_template_leaves_no_jinja_markers_and_one_item_per_bullet():
    profile = load_profile(EXAMPLE)
    tex = render_tex(build_view(profile, full_profile_cv(profile)))
    assert "<<" not in tex and "<%" not in tex and "<#" not in tex
    bullets = sum(len(e.bullets) for e in [*profile.experiences, *profile.projects])
    assert tex.count(r"\resumeItem{") == bullets
    assert r"\ifPDFTeX" in tex  # pdfLaTeX-only lines are guarded


@pytest.mark.skipif(shutil.which("tectonic") is None, reason="tectonic not installed")
def test_tricky_text_compiles_and_reads_back(tmp_path):
    profile = load_profile(EXAMPLE)
    tricky = "Cut cost 50% & time ~30% for R&D_team #1 {beta} → done; see [demo](https://x.io/a#b)"
    cv = TailoredCV(
        sections=[
            TailoredSection(
                type="experience",
                ref="exp_acme",
                bullets=[
                    TailoredBullet(text=tricky, source_bullet_ids=["b_acme_01"], grounding="pass")
                ],
            )
        ],
        skills=["Python"],
    )
    result = compile_pdf(render_tex(build_view(profile, cv)), tmp_path)
    assert result.pages == 1
    text = PdfReader(result.pdf_path).pages[0].extract_text()
    assert "50% & time" in text and "R&D_team #1 {beta}" in text and "demo" in text
