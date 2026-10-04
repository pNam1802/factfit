import pytest

from factfit.grounding import check_bullet, load_kb
from factfit.schemas.profile import Bullet

SOURCE = Bullet.model_validate(
    {
        "id": "b1",
        "text": "Contributed to a multi-camera tracking pipeline with YOLOv8 and ByteTrack in "
        "torch, processing 1,200 clips at 25 fps",
        "skills": ["YOLOv8", "ByteTrack", "Python"],
        "metrics": [
            {"name": "fps", "value": 25, "unit": "fps", "verified": True},
            {"name": "clips", "value": 1200, "verified": True},
        ],
    }
)


def kinds(text: str, sources=(SOURCE,)) -> list[tuple[str, str]]:
    return [(i.kind, i.detail) for i in check_bullet(text, list(sources))]


# --- must pass: rewrites that add nothing ------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # same facts, new wording
        "Contributed to a YOLOv8 + ByteTrack multi-camera tracker running at 25 fps",
        # alias: the source says "torch", the rewrite says "PyTorch"
        "Helped build a PyTorch tracking pipeline over 1.2k clips",
        # number written differently: 1,200 == 1200 == 1.2k
        "Contributed to processing 1200 clips with YOLOv8",
        # dropping facts is allowed
        "Contributed to a multi-camera tracking pipeline",
        # Python is in the source's skills even though not in its text
        "Contributed to a Python tracking pipeline",
    ],
)
def test_faithful_rewrites_pass(text):
    assert kinds(text) == []


# --- must fail: each kind of fabrication -------------------------------------------------------


def test_changed_number_is_caught():
    assert kinds("Contributed to a YOLOv8 tracker running at 30 fps") == [
        ("number_unsupported", "30")
    ]


def test_new_percentage_is_caught():
    assert ("number_unsupported", "40%") in kinds("Contributed to cutting latency by 40%")


def test_version_in_a_name_is_not_a_number():
    # "YOLOv8" holds an 8, "ResNet50" a 50: neither is a claim about results.
    assert kinds("Contributed to a YOLOv8 tracker") == []


def test_added_technology_is_caught():
    assert kinds("Contributed to a YOLOv8 tracker deployed with Docker on Kubernetes") == [
        ("tech_unsupported", "Docker"),
        ("tech_unsupported", "Kubernetes"),
    ]


def test_specific_is_not_accepted_for_generic():
    # The source says YOLOv8; "YOLO26" is a different model, not a synonym.
    assert kinds("Contributed to a YOLO26 tracker") == [("tech_unsupported", "YOLO26")]


def test_sql_inside_postgresql_is_not_sql():
    source = Bullet(id="b2", text="Stored results in PostgreSQL", skills=["PostgreSQL"])
    assert kinds("Built storage on PostgreSQL", [source]) == []
    assert kinds("Built storage with SQL and PostgreSQL", [source]) == [("tech_unsupported", "SQL")]


@pytest.mark.parametrize("verb", ["Led", "Designed", "Spearheaded", "Architected"])
def test_role_inflation_is_caught(verb):
    issues = kinds(f"{verb} a multi-camera tracking pipeline with YOLOv8")
    assert issues and issues[0][0] == "role_inflated"


def test_same_or_lower_role_passes():
    assert kinds("Supported a multi-camera tracking pipeline") == []


def test_vietnamese_role_verbs():
    source = Bullet(id="b3", text="Tham gia xây dựng pipeline theo dõi người")
    assert kinds("Tham gia phát triển pipeline theo dõi người", [source]) == []
    # "took part in building" -> "built": the same inflation as "contributed to" -> "built"
    assert kinds("Xây dựng pipeline theo dõi người", [source])[0][0] == "role_inflated"
    assert kinds("Dẫn dắt nhóm xây dựng pipeline theo dõi người", [source])[0][0] == (
        "role_inflated"
    )


def test_technical_phrases_are_not_role_claims():
    source = Bullet(id="b4", text="Built a supervised learning model for a directed graph")
    assert kinds("Built a supervised learning model for a directed graph", [source]) == []


def test_several_sources_are_combined():
    other = Bullet(
        id="b5",
        text="Deployed the tracker with Docker",
        skills=["Docker"],
    )
    assert kinds("Contributed to a YOLOv8 tracker deployed with Docker", [SOURCE, other]) == []


def test_no_source_is_an_issue():
    assert kinds("Anything", []) == [("no_source", "")]


def test_alias_table_has_no_conflicting_spellings():
    # A spelling mapped to two canonical names would make results depend on file order.
    import yaml

    from factfit.grounding.rules import DATA_DIR

    seen: dict[str, str] = {}
    data = yaml.safe_load((DATA_DIR / "aliases.yaml").read_text(encoding="utf-8"))
    for canonical, others in data.items():
        for spelling in [canonical, *(others or [])]:
            key = str(spelling).casefold()
            assert seen.setdefault(key, canonical) == canonical, f"{spelling} maps twice"
    assert load_kb().canonical("torch") == "PyTorch"


def test_verbs_describing_objects_are_not_role_claims():
    # "deployed" describes the tracker; the role is still "contributed to".
    assert kinds("Contributed to a YOLOv8 tracker deployed with ByteTrack") == []


def test_leadership_later_in_the_bullet_still_counts():
    source = Bullet(id="b6", text="Built a tracking pipeline")
    assert kinds("Built a tracking pipeline and led a team", [source])[0] == (
        "role_inflated",
        "led",
    )
