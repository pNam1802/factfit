import json

import pytest
from sqlmodel import Session, select

from factfit.agent.nodes.parse_jd import jd_hash, normalize_jd, parse_jd
from factfit.db import Company, Job, init_db, make_engine
from factfit.llm import LLMClient
from factfit.llm.backends import RawOutput
from factfit.llm.config import LLMConfig
from factfit.prompts import load_prompt

JD_TEXT = """THỰC TẬP SINH AI
Yêu cầu
•\tThành thạo Python
Điểm cộng
•\tCó kinh nghiệm triển khai REST API
"""

ANSWER = json.dumps(
    {
        "title": "Thực tập sinh AI",
        "company": "Mekong HT Holding",
        "seniority": "intern",
        "language": "vi",
        "location": "Hà Nội",
        "must_have": [
            {
                "id": "r1",
                "text": "Thành thạo Python",
                "category": "skill",
                "any_of": [],
                "level": None,
            }
        ],
        "nice_to_have": [
            {
                "id": "r2",
                "text": "Kinh nghiệm triển khai REST API",
                "category": "experience",
                "any_of": [],
                "level": None,
            }
        ],
        "responsibilities": ["Phát triển giải pháp LLM"],
        "keywords": ["Python", "REST API"],
    }
)


class CountingBackend:
    def __init__(self):
        self.calls: list[str] = []

    def complete(self, *, node, model, system, user, schema, options):
        self.calls.append(user)
        return RawOutput(text=ANSWER, tokens_in=500, tokens_out=200)


def config(prompt="v1"):
    return LLMConfig.model_validate(
        {"models": {"small": "m"}, "nodes": {"parse_jd": {"model": "small", "prompt": prompt}}}
    )


@pytest.fixture
def session(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    with Session(engine) as s:
        yield s


def test_normalize_ignores_spacing_differences():
    a = "Title\n\n\n\n•\tPython   and  SQL  \n"
    b = "Title\n\n• Python and SQL"
    assert normalize_jd(a) == normalize_jd(b) == "Title\n\n• Python and SQL"
    assert jd_hash(a) == jd_hash(b)


def test_prompt_file_loads_and_renders():
    prompt = load_prompt("parse_jd", "v1")
    assert prompt.id == "parse_jd/v1"
    assert "must_have" in prompt.system
    user = prompt.render(company="ABC", jd_text="Python")
    assert "ABC" in user and "Python" in user


def test_parse_stores_job_and_company(session):
    backend = CountingBackend()
    result = parse_jd(JD_TEXT, client=LLMClient(backend, config()), session=session)

    assert not result.cached
    assert result.jd.must_have[0].text == "Thành thạo Python"
    job = session.exec(select(Job)).one()
    assert job.parse_prompt_version == "parse_jd/v1"
    assert job.parsed_json["seniority"] == "intern"
    assert session.get(Company, job.company_id).name == "Mekong HT Holding"
    assert "Thành thạo Python" in backend.calls[0]  # the JD text reached the model


def test_same_jd_is_served_from_cache(session):
    backend = CountingBackend()
    client = LLMClient(backend, config())
    parse_jd(JD_TEXT, client=client, session=session)
    again = parse_jd(JD_TEXT.replace("\t", "  "), client=client, session=session)

    assert again.cached
    assert len(backend.calls) == 1
    assert len(session.exec(select(Job)).all()) == 1


def test_new_prompt_version_parses_again(session, tmp_path, monkeypatch):
    for version in ("v1", "v2"):
        path = tmp_path / "parse_jd" / f"{version}.md"
        path.parent.mkdir(exist_ok=True)
        path.write_text(f"rules {version}\n=== USER ===\n$company $jd_text", encoding="utf-8")
    monkeypatch.setattr("factfit.prompts.PROMPTS_DIR", tmp_path)

    backend = CountingBackend()
    parse_jd(JD_TEXT, client=LLMClient(backend, config("v1")), session=session)
    result = parse_jd(JD_TEXT, client=LLMClient(backend, config("v2")), session=session)

    assert not result.cached
    assert len(backend.calls) == 2
    assert session.exec(select(Job)).one().parse_prompt_version == "parse_jd/v2"


def test_company_given_by_user_wins(session):
    result = parse_jd(
        JD_TEXT, client=LLMClient(CountingBackend(), config()), session=session, company="Mekong"
    )
    assert session.get(Company, result.job.company_id).name == "Mekong"


def test_empty_jd_is_rejected(session):
    with pytest.raises(ValueError, match="empty"):
        parse_jd("  \n ", client=LLMClient(CountingBackend(), config()), session=session)
