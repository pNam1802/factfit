import pytest

from factfit.text.numbers import find_numbers, parse_number


def raws(text: str, **kwargs) -> list[str]:
    return [m.raw for m in find_numbers(text, **kwargs)]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("runs at 25 fps", ["25"]),
        ("runs at 25fps", ["25fps"]),
        ("cut latency by 40%", ["40%"]),
        ("3 pilot stores", ["3"]),
        ("2-3 cameras", ["2", "3"]),
        ("30–40% faster", ["30", "40%"]),
    ],
)
def test_finds_real_claims(text, expected):
    assert raws(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "trained YOLOv8",
        "fine-tuned ResNet50",
        "evaluated GPT-4o",
        "built a 3D viewer",
        "hybrid search with BM25",
        "shipped in 2025",
        "from Mar 2026",
        "(2024)",
    ],
)
def test_ignores_names_and_years(text):
    assert raws(text) == []


def test_mask_hides_versions_inside_names():
    assert raws("migrated to Python 3.11 and Django 4.2", mask=["Python 3.11"]) == ["4.2"]


def test_a_bare_large_number_is_not_mistaken_for_a_year():
    assert raws("processed 2000 images") == ["2000"]


@pytest.mark.parametrize(
    "raw, lang, value",
    [
        ("1,000", "en", 1000),
        ("1.000", "vi", 1000),
        ("0,85", "vi", 0.85),
        ("0.85", "en", 0.85),
        ("3.11", "en", 3.11),
        ("1,234,567", "en", 1234567),
        ("1.234,5", "vi", 1234.5),
    ],
)
def test_parse_number(raw, lang, value):
    assert parse_number(raw, lang) == pytest.approx(value)


def test_multiplier_suffix_gives_both_readings():
    (mention,) = find_numbers("serving 5M requests")
    assert mention.values == {5.0, 5_000_000.0}
