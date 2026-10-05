import shutil
from pathlib import Path

import pytest

from factfit.profile import load_profile
from factfit.render import ats
from factfit.render.compile import render_and_check
from factfit.render.view import full_profile_cv
from factfit.schemas.tailored import TailoredBullet, TailoredCV, TailoredSection

EXAMPLE = Path(__file__).parent.parent / "data" / "profile.example.yaml"
PROFILE = load_profile(EXAMPLE)
CV = TailoredCV(
    sections=[
        TailoredSection(
            type="experience",
            ref="exp_acme",
            bullets=[
                TailoredBullet(
                    text="Built an EfficientNet tracker at 25 fps", source_bullet_ids=["b_acme_01"]
                )
            ],
        )
    ],
    skills=["Python"],
)
GOOD = (
    "Nguyen Van A\nTechnical Skills\nSkills: Python\nExperience\nAI Engineer Intern\n"
    "Built an EfficientNet tracker at\n25 fps\nEducation\nB.Sc. in Computer Science"
)


def fake_extractors(monkeypatch, text_by_name):
    monkeypatch.setattr(ats, "EXTRACTORS", {n: (lambda p, t=t: t) for n, t in text_by_name.items()})


def test_clean_text_passes_despite_line_breaks(monkeypatch):
    fake_extractors(monkeypatch, {"a": GOOD, "b": GOOD})
    assert ats.check_pdf(Path("x.pdf"), PROFILE, CV, ["EfficientNet"]).ok


def test_ligature_is_an_issue_even_if_casefold_would_hide_it(monkeypatch):
    fake_extractors(monkeypatch, {"a": GOOD.replace("EfficientNet", "EﬀicientNet"), "b": GOOD})
    report = ats.check_pdf(Path("x.pdf"), PROFILE, CV)
    assert not report.ok and "broken characters" in report.issues[0]
    assert report.issues[0].startswith("[a]")  # names the extractor that failed


def test_missing_bullet_and_wrong_order(monkeypatch):
    swapped = (
        "Nguyen Van A\nExperience\nAI Engineer Intern\nTechnical Skills\nSkills: Python\nEducation"
    )
    fake_extractors(monkeypatch, {"a": swapped})
    issues = ats.check_pdf(Path("x.pdf"), PROFILE, CV).issues
    assert any("bullet not readable" in i for i in issues)
    assert any("wrong order" in i for i in issues)


def test_keyword_split_inside_a_word_is_a_warning(monkeypatch):
    fake_extractors(monkeypatch, {"a": GOOD.replace("EfficientNet", "E fficientNet")})
    report = ats.check_pdf(Path("x.pdf"), PROFILE, CV, ["EfficientNet"])
    assert report.ok
    assert report.warnings == ["a reads keyword 'EfficientNet' split inside a word"]


def test_keywords_the_cv_does_not_claim_are_not_required(monkeypatch):
    fake_extractors(monkeypatch, {"a": GOOD})
    assert ats.check_pdf(Path("x.pdf"), PROFILE, CV, ["Kubernetes"]).ok


@pytest.mark.skipif(shutil.which("tectonic") is None, reason="tectonic not installed")
def test_real_pdf_reads_back_without_ligatures(tmp_path):
    # "Efficient" and "traffic" were read with a U+FB00 ligature before ligatures were disabled.
    cv = full_profile_cv(PROFILE)
    cv.sections[0].bullets[0].text = "Built an EfficientNet model for store traffic, offline"
    _, report = render_and_check(PROFILE, cv, tmp_path, ["EfficientNet"])
    assert report.ok, report.issues
    for text in report.texts.values():
        assert "EfficientNet" in text and "ﬀ" not in text
