"""Regression guard on the public grounding test set (evals/grounding/cases.jsonl).

The rule checker must keep catching what it is built to catch, without starting to flag
faithful rewrites. Words-only fabrications (scale, outcome) are not asserted here: catching
them is the LLM judge's job, measured separately.
"""

import pytest

from factfit.evals import grounding_cases as gc
from factfit.evals import grounding_eval as ge
from factfit.evals.stats import wilson


@pytest.fixture(scope="module")
def data():
    return gc.load_cases(), gc.load_sources()


@pytest.fixture(scope="module")
def results(data):
    cases, sources = data
    return {r.mutation_type: r for r in ge.evaluate(cases, sources, ge.rules_checker)}


def test_test_set_is_well_formed(data):
    cases, sources = data
    assert len(cases) >= 200
    assert sum(c.label == "fabricated" for c in cases) >= 100
    assert len({c.id for c in cases}) == len(cases)
    assert all(s in sources for c in cases for s in c.source_ids)
    assert all((c.label == "grounded") == (c.mutation_type == "valid_paraphrase") for c in cases)


@pytest.mark.parametrize("kind", gc.CODE_TYPES)
def test_rules_catch_what_they_are_built_for(results, kind):
    assert results[kind].rate >= 0.95, [c.id for c in results[kind].misses]


def test_rules_rarely_flag_faithful_rewrites(results):
    assert results["valid_paraphrase"].rate <= 0.10, [
        c.id for c in results["valid_paraphrase"].misses
    ]


def test_code_mutations_are_reproducible(data):
    _, sources = data
    a = [(c.mutation_type, c.text) for c in gc.code_mutations(sources)]
    b = [(c.mutation_type, c.text) for c in gc.code_mutations(sources)]
    assert a == b


def test_wilson_interval():
    lo, hi = wilson(34, 34)
    assert 0.89 < lo < 0.91 and hi == 1.0
    lo, hi = wilson(0, 34)
    assert lo == 0.0 and 0.09 < hi < 0.11
    assert wilson(0, 0) is None
