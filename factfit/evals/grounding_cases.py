"""Build the grounding test set (PRD §8.3): pairs of (source bullet, rewrite, label).

Two origins:
- code: deterministic mutations the rule checks are designed to catch
    number_change    one number changed
    tech_injection   a technology the source does not use is added
    role_inflation   the role verb raised (contributed -> led)
- llm: written by a model from a prompt, then reviewed by a person
    scale_inflation      bigger scope, words only (no new digits or technologies)
    unsupported_outcome  a result the source does not state, words only
    valid_paraphrase     faithful rewording, label "grounded": measures false positives
- hand: written by a person in evals/grounding/hand_cases.yaml (never overwritten by a build)
    subtle_claim     one unsupported claim woven into the sentence, words only
    hard_paraphrase  faithful rewording with heavy restructuring, label "grounded"

Generated cases live in evals/grounding/cases.jsonl. The set is public; sources are fictional.
"""

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import yaml

from factfit.grounding.rules import KnowledgeBase, check_bullet, load_kb
from factfit.llm import LLMClient
from factfit.prompts import load_prompt
from factfit.schemas.base import Strict
from factfit.schemas.profile import Bullet
from factfit.text.numbers import find_numbers, guess_lang

GROUNDING_DIR = Path("evals/grounding")
GEN_NODE = "gen_cases"

MutationType = Literal[
    "number_change",
    "tech_injection",
    "role_inflation",
    "scale_inflation",
    "unsupported_outcome",
    "valid_paraphrase",
    "subtle_claim",
    "hard_paraphrase",
]
CODE_TYPES = ("number_change", "tech_injection", "role_inflation")
LLM_TYPES = ("scale_inflation", "unsupported_outcome", "valid_paraphrase")
HAND_TYPES = ("subtle_claim", "hard_paraphrase")
GROUNDED_TYPES = ("valid_paraphrase", "hard_paraphrase")

TECH_POOL = ["Docker", "Kubernetes", "PyTorch", "Redis", "AWS", "TensorRT", "Kafka", "MLflow"]
LEAD_PREFIXES = {
    "en": ["Led the team that", "Headed the effort that", "Spearheaded a project that"],
    "vi": ["Dẫn dắt nhóm", "Trưởng nhóm"],
}


@dataclass
class Case:
    id: str
    source_ids: list[str]
    text: str
    label: Literal["grounded", "fabricated"]
    mutation_type: MutationType
    origin: Literal["code", "llm", "hand"]
    note: str = ""  # what a hand case adds, or a reviewer's remark


def load_sources(path: Path = GROUNDING_DIR / "sources.yaml") -> dict[str, Bullet]:
    items = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {b["id"]: Bullet.model_validate(b) for b in items}


def load_cases(root: Path = GROUNDING_DIR) -> list[Case]:
    """The whole test set: generated cases plus hand-written ones."""
    return load_generated(root / "cases.jsonl") + load_hand(root / "hand_cases.yaml")


def load_generated(path: Path = GROUNDING_DIR / "cases.jsonl") -> list[Case]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Case(**json.loads(line)) for line in lines if line.strip()]


def load_hand(path: Path = GROUNDING_DIR / "hand_cases.yaml") -> list[Case]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cases = []
    for kind in HAND_TYPES:
        for i, item in enumerate(data.get(kind, []), start=1):
            # An unquoted comma inside a {...} entry silently splits the text into an extra
            # key; reject anything but the expected keys so that cannot go unnoticed.
            extra = set(item) - {"source", "adds", "text"}
            if extra:
                raise ValueError(f"{path}: {kind} #{i} has unexpected keys {extra}; quote the text")
            label = "grounded" if kind in GROUNDED_TYPES else "fabricated"
            case_id = f"h{'s' if kind == 'subtle_claim' else 'p'}{i:03d}"
            cases.append(
                Case(
                    case_id,
                    [item["source"]],
                    item["text"],
                    label,
                    kind,
                    "hand",
                    item.get("adds", ""),
                )
            )
    return cases


def save_cases(cases: list[Case], path: Path = GROUNDING_DIR / "cases.jsonl") -> None:
    text = "".join(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in cases)
    path.write_text(text, encoding="utf-8", newline="\n")


# --- code mutations --------------------------------------------------------------------------


def code_mutations(
    sources: dict[str, Bullet], kb: KnowledgeBase | None = None, seed: int = 7
) -> list[Case]:
    kb = kb or load_kb()
    rng = random.Random(seed)
    cases: list[Case] = []
    for sid, source in sources.items():
        for kind, text in (
            ("number_change", _change_number(source, kb)),
            ("tech_injection", _inject_tech(source, kb, rng)),
            ("role_inflation", _inflate_role(source, kb, rng)),
        ):
            if text and text != source.text:
                cases.append(Case("", [sid], text, "fabricated", kind, "code"))
    return cases


def _change_number(source: Bullet, kb: KnowledgeBase) -> str | None:
    lang = guess_lang(source.text)
    mentions = find_numbers(source.text, lang=lang, mask=[*kb.tech_names, *source.skills])
    if not mentions:
        return None
    raw = mentions[0].raw
    num = re.match(r"[\d.,]+", raw).group(0)
    value = min(mentions[0].values)  # "10k" -> 10, so the suffix stays meaningful
    bump = max(1, round(value * 0.2))
    new = value - bump if raw.endswith("%") and value + bump > 99 else value + bump
    new_num = f"{int(new):,}" if ("," in num or "." in num) and new >= 1000 else str(int(new))
    if lang == "vi":
        new_num = new_num.replace(",", ".")
    return source.text.replace(raw, raw.replace(num, new_num, 1), 1)


def _inject_tech(source: Bullet, kb: KnowledgeBase, rng: random.Random) -> str:
    present = {kb.canonical(s) for s in source.skills} | {
        str(canon) for _, canon in kb.tech.find(source.text)
    }
    tech = rng.choice([t for t in TECH_POOL if t not in present])
    if guess_lang(source.text) == "vi":
        return f"{source.text}, triển khai bằng {tech}"
    return rng.choice([f"{source.text}, deployed with {tech}", f"{source.text} using {tech}"])


def _inflate_role(source: Bullet, kb: KnowledgeBase, rng: random.Random) -> str | None:
    """Claim leadership of the work: "Led the team that <source>".

    Swapping the verb in place reads badly ("Architected inference latency..."), and an
    unnatural sentence is an easier test than a real fabrication; a prefix always reads well.
    """
    found = kb.verbs.find(source.text)
    if found and found[0][1] >= 4:
        return None  # already a leadership claim: nothing higher to add
    rest = source.text[0].lower() + source.text[1:]
    if guess_lang(source.text) == "vi":
        return f"{rng.choice(LEAD_PREFIXES['vi'])} {rest}"
    return f"{rng.choice(LEAD_PREFIXES['en'])} {rest}"


# --- LLM-written cases -----------------------------------------------------------------------


class Variants(Strict):
    scale_inflation: str
    unsupported_outcome: str
    valid_paraphrases: list[str]


def llm_variants(sources: dict[str, Bullet], client: LLMClient) -> list[Case]:
    prompt = load_prompt(GEN_NODE, client.config.prompt_for(GEN_NODE))
    cases: list[Case] = []
    for sid, source in sources.items():
        out = client.parse(
            Variants,
            node=GEN_NODE,
            prompt_version=prompt.id,
            system=prompt.system,
            user=prompt.render(source=source.text),
        )
        cases.append(Case("", [sid], out.scale_inflation, "fabricated", "scale_inflation", "llm"))
        cases.append(
            Case("", [sid], out.unsupported_outcome, "fabricated", "unsupported_outcome", "llm")
        )
        for text in out.valid_paraphrases[:2]:
            cases.append(Case("", [sid], text, "grounded", "valid_paraphrase", "llm"))
    return cases


def impure(case: Case, sources: dict[str, Bullet], kb: KnowledgeBase | None = None) -> list[str]:
    """For a words-only fabrication: rule issues mean it also changed numbers, tech or role,
    so it would not test what its type claims to test."""
    issues = check_bullet(case.text, [sources[s] for s in case.source_ids], kb)
    return [f"{i.kind}: {i.detail}" for i in issues]


def number_ids(cases: list[Case]) -> list[Case]:
    for i, case in enumerate(cases, start=1):
        case.id = f"c{i:04d}"
    return cases
