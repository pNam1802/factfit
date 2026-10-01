from pathlib import Path

import pytest

from factfit.profile import ProfileError, load_profile

EXAMPLE = Path(__file__).parent.parent / "data" / "profile.example.yaml"

# Smallest valid profile; tests change one thing at a time.
BASE = """\
basics:
  name: Test User
  email: t@example.com
experiences:
  - id: exp_a
    org: Org
    role: Intern
    start: 2026-01
    end: present
    bullets:
      - id: b_a_01
        text: {text}
        skills: [YOLOv8]
        metrics:
          - {{name: fps, value: 25, unit: fps, verified: {verified}}}
skills:
  - {{name: Python, evidence: [{evidence}]}}
"""


def load(tmp_path, text="Ran YOLOv8 at 25 fps", verified="true", evidence="b_a_01", extra=""):
    file = tmp_path / "profile.yaml"
    file.write_text(
        BASE.format(text=text, verified=verified, evidence=evidence) + extra, encoding="utf-8"
    )
    return load_profile(file)


def problems(tmp_path, **kwargs) -> list[str]:
    with pytest.raises(ProfileError) as exc:
        load(tmp_path, **kwargs)
    return [i.format() for i in exc.value.issues]


def test_example_profile_is_valid():
    profile = load_profile(EXAMPLE)
    assert profile.basics.name == "Nguyen Van A"


def test_base_profile_is_valid(tmp_path):
    assert load(tmp_path).experiences[0].bullets[0].id == "b_a_01"


def test_number_not_in_metrics(tmp_path):
    (msg,) = problems(tmp_path, text="Ran YOLOv8 at 30 fps")
    assert "number '30' is not in this item's metrics" in msg
    assert msg.startswith("line 12: experiences[0].bullets[0].text")


def test_metric_not_verified(tmp_path):
    (msg,) = problems(tmp_path, verified="false")
    assert "not verified" in msg


def test_skill_evidence_must_be_a_bullet(tmp_path):
    (msg,) = problems(tmp_path, evidence="b_missing")
    assert "'b_missing', which is not a bullet id" in msg


def test_duplicate_id(tmp_path):
    extra = """\
projects:
  - id: b_a_01
    name: Clash
    bullets:
      - {id: b_p_01, text: A project}
"""
    (msg,) = problems(tmp_path, extra=extra)
    assert "duplicate id 'b_a_01'" in msg


def test_typo_in_field_name_is_reported_with_line(tmp_path):
    file = tmp_path / "profile.yaml"
    file.write_text(
        BASE.format(text="Ran at 25 fps", verified="true", evidence="b_a_01").replace(
            "metrics:", "metircs:"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProfileError) as exc:
        load_profile(file)
    typo = next(i for i in exc.value.issues if "metircs" in str(i.path))
    assert typo.line == 14


def test_bad_date_format(tmp_path):
    file = tmp_path / "profile.yaml"
    file.write_text(
        BASE.format(text="Ran at 25 fps", verified="true", evidence="b_a_01").replace(
            "start: 2026-01", "start: Jan 2026"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ProfileError) as exc:
        load_profile(file)
    assert any(i.path[-1] == "start" and i.line == 8 for i in exc.value.issues)


def test_all_problems_reported_at_once(tmp_path):
    msgs = problems(tmp_path, text="Ran at 30 fps", evidence="b_missing")
    assert len(msgs) == 2


def test_year_only_dates_and_certifications(tmp_path):
    extra = """\
education:
  - {id: edu_u, school: Uni, degree: B.S., start: 2022, end: 2026}
certifications:
  - {id: cert_ml, name: Machine Learning Specialization, date: 2025}
"""
    profile = load(tmp_path, extra=extra)
    assert profile.education[0].start == "2022"
    assert profile.certifications[0].date == "2025"


def test_certification_ids_are_checked_for_duplicates(tmp_path):
    extra = """\
certifications:
  - {id: exp_a, name: Clash}
"""
    (msg,) = problems(tmp_path, extra=extra)
    assert "duplicate id 'exp_a'" in msg
