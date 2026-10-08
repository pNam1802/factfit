from datetime import UTC, datetime, timedelta

from sqlmodel import Session

from factfit.db import LLMCall, init_db, make_engine
from factfit.evals.cost import run_costs


def test_run_costs_groups_calls_by_run_and_node(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path / 'app.db'}")
    init_db(engine)
    t0 = datetime(2026, 10, 8, tzinfo=UTC)

    def call(run, node, cost, minutes, model="m"):
        return LLMCall(run_id=run, node=node, model=model, prompt_version="x", cost_usd=cost,
                       created_at=t0 + timedelta(minutes=minutes))  # fmt: skip

    with Session(engine) as s:
        s.add_all([
            call("old", "rewrite", 0.03, 0), call("old", "judge", 0.01, 1),
            call("new", "parse_jd", 0.005, 10), call("new", "rewrite", 0.04, 11),
            call("new", "judge", None, 12),  # model without a price
            call("eval", "judge", 0.02, 20),  # no rewrite: not a tailoring run
            call(None, "rewrite", 0.5, 30),  # not part of any run
        ])  # fmt: skip
        s.commit()
        runs = run_costs(s)

    assert [r.run_id for r in runs] == ["new", "old"]
    new = runs[0]
    assert round(new.cost, 4) == 0.045 and new.calls == 3 and new.unpriced == 1
    assert dict(new.by_node) == {"parse_jd": 0.005, "rewrite": 0.04}
