"""Command line entry point: `factfit <command>`."""

import argparse
import sys

from factfit.profile import ProfileError, load_profile
from factfit.schemas.profile import SECTIONS


def main(argv: list[str] | None = None) -> int:
    # Windows consoles may default to a legacy code page; make Vietnamese text printable.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="factfit")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-profile", help="check a profile.yaml for errors")
    validate.add_argument("path", nargs="?", default="data/profile.yaml")
    commands.add_parser("llm-check", help="send one tiny request to check your API key and model")
    parse = commands.add_parser("parse-jd", help="extract requirements from a job description file")
    parse.add_argument("path", help="text file with the job description")
    parse.add_argument("--company", help="company name, if the JD does not say")
    parse.add_argument("--url", help="where the JD was posted")
    ev = commands.add_parser("eval-parse", help="score parse_jd against hand labels")
    ev.add_argument("--split", default="dev", help="folder under evals/jds/ (dev or holdout)")
    ev.add_argument("--prompts", nargs="+", default=["v1"], help="prompt versions to compare")
    ev.add_argument(
        "--efforts",
        nargs="+",
        default=["default"],
        help="reasoning efforts to compare: default, minimal, low, medium, high",
    )
    m = commands.add_parser("match", help="check your profile against a job description file")
    m.add_argument("path", help="text file with the job description")
    m.add_argument("--profile", default="data/profile.yaml")
    m.add_argument("--company", help="company name, if the JD does not say")
    m.add_argument(
        "--level",
        choices=["intern", "fresher", "junior", "mid", "senior"],
        help="your level, for postings that hire several (default: lowest the JD accepts)",
    )
    dev = commands.add_parser("dev", help="run the API and the web UI together")
    dev.add_argument("--api-port", type=int, default=8000)
    dev.add_argument("--ui-port", type=int, default=3000)
    oa = commands.add_parser("export-openapi", help="write the API schema for the UI types")
    oa.add_argument("path", nargs="?", default="ui/openapi.json")
    args = parser.parse_args(argv)

    if args.command == "validate-profile":
        return _validate_profile(args.path)
    if args.command == "llm-check":
        return _llm_check()
    if args.command == "parse-jd":
        return _parse_jd(args.path, args.company, args.url)
    if args.command == "eval-parse":
        return _eval_parse(args.split, args.prompts, args.efforts)
    if args.command == "match":
        return _match(args.path, args.profile, args.company, args.level)
    if args.command == "dev":
        from factfit.devserver import run_dev

        return run_dev(api_port=args.api_port, ui_port=args.ui_port)
    if args.command == "export-openapi":
        from factfit.api.app import export_openapi

        export_openapi(args.path)
        print(f"Wrote {args.path}. Regenerate UI types with: cd ui && npm run gen:api")
        return 0
    return 2


def _match(path: str, profile_path: str, company: str | None, level: str | None) -> int:
    import uuid
    from pathlib import Path

    from sqlalchemy import func
    from sqlmodel import Session, select

    from factfit.agent.nodes.match import match_job, profile_items
    from factfit.agent.nodes.parse_jd import parse_jd
    from factfit.db import LLMCall, init_db, make_engine
    from factfit.llm import LLMError, make_client
    from factfit.profile import ProfileError, load_profile, profile_version

    try:
        profile = load_profile(profile_path)
    except ProfileError as e:
        print(f"{profile_path} is not valid yet ({len(e.issues)} problem(s)).")
        print(f"Run: uv run factfit validate-profile {profile_path}")
        return 1
    except FileNotFoundError:
        print(f"{profile_path}: file not found")
        return 1

    engine = make_engine()
    init_db(engine)
    run_id = uuid.uuid4().hex
    try:
        raw = Path(path).read_text(encoding="utf-8")
        client = make_client(engine=engine)
        with Session(engine) as session:
            parsed = parse_jd(raw, client=client, session=session, company=company, run_id=run_id)
            jd = parsed.jd
            # Without --level, judge as the lowest level the posting accepts.
            applicant_level = level or (jd.seniority if jd.seniority != "unknown" else None)
            outcome = match_job(
                parsed.job,
                profile,
                profile_version=profile_version(profile_path),
                client=client,
                session=session,
                level=applicant_level,
                run_id=run_id,
            )
            calls, cost, latency = session.exec(
                select(
                    func.count(), func.sum(LLMCall.cost_usd), func.sum(LLMCall.latency_ms)
                ).where(LLMCall.run_id == run_id)
            ).one()
    except (OSError, ValueError, LLMError) as e:
        print(f"FAILED: {e}")
        return 1

    result = outcome.result
    reqs = {r.id: (r, "must") for r in jd.must_have} | {r.id: (r, "nice") for r in jd.nice_to_have}
    item_text = {i.id: i.text for i in profile_items(profile)}
    icon = {"met": "[x]", "partial": "[~]", "missing": "[ ]", "unverifiable": "[?]"}

    print(f"{jd.title}  |  {jd.company or '?'}  |  judged as level: {result.level or 'any'}")
    print(f"Match score: {result.score}/100")
    for bucket in ("must", "nice"):
        rows = [r for r in result.requirements if reqs[r.req_id][1] == bucket]
        if not rows:
            continue
        met = sum(r.status == "met" for r in rows)
        part = sum(r.status == "partial" for r in rows)
        title = "Must have" if bucket == "must" else "Nice to have"
        print(f"\n{title}: {met} met, {part} partial, of {len(rows)}")
        for r in rows:
            print(f"  {icon[r.status]} {r.req_id:>4}  {reqs[r.req_id][0].text}")
            print(f"           {r.note}")
            for e in r.evidence:
                print(f"           <- {e}: {item_text.get(e, '')[:90]}")
    if result.excluded:
        print(f"\nSkipped (for another level): {', '.join(result.excluded)}")
    print("\nLegend: [x] met  [~] partial  [ ] missing  [?] a CV cannot show this")

    if outcome.cached and parsed.cached:
        print("\n(cached: no LLM call)")
    else:
        cost_text = f"${cost:.5f}" if cost is not None else "unknown"
        print(f"\nLLM calls: {calls}, total latency: {latency} ms, cost: {cost_text}")
    return 0


def _eval_parse(split: str, prompts: list[str], efforts: list[str]) -> int:
    import json
    from datetime import datetime
    from pathlib import Path

    from factfit.evals import parse_eval as pe
    from factfit.llm import LLMError, make_client

    cases = pe.load_cases(split)
    if not cases:
        print(f"No labelled job descriptions in {pe.JDS_DIR / split} (need <name>.labels.yaml).")
        return 1
    variants = [
        pe.Variant(prompt=p, effort=None if e == "default" else e) for p in prompts for e in efforts
    ]
    try:
        client = make_client()
    except LLMError as e:
        print(f"FAILED: {e}")
        return 1

    print(f"{len(cases)} job descriptions x {len(variants)} variants...")
    results = pe.run(
        cases, variants, backend=client.backend, base_config=client.config, progress=print
    )

    def pct(x):
        return "  -  " if x is None else f"{x:5.0%}"

    header = f"\n{'variant':<14}{'prec':>6}{'recall':>8}{'bucket':>8}{'any_of':>8}{'level':>7}"
    print(header + f"{'reqs/JD':>9}{'$/JD':>9}{'s/JD':>7}{'fail':>6}")
    for r in results:
        m, s, t = r.totals().metrics(), r.parse_stats(), r.totals()
        cost = "  ?" if s["cost_per_jd"] is None else f"{s['cost_per_jd']:.4f}"
        ok = len(r.cases) - s["failed"] or 1
        print(
            f"{r.variant.label:<14}{pct(m['precision']):>6}{pct(m['recall']):>8}"
            f"{pct(m['bucket']):>8}{pct(m['any_of']):>8}{pct(m['level']):>7}"
            f"{t.predicted / ok:>9.1f}{cost:>9}{s['latency_s_per_jd']:>7.1f}{s['failed']:>6}"
        )

    out = Path("evals/results") / f"parse-{split}-{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pe.to_json(results), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nFull results (per JD, with predictions): {out}")
    return 0


def _parse_jd(path: str, company: str | None, url: str | None) -> int:
    import uuid
    from pathlib import Path

    from sqlalchemy import func
    from sqlmodel import Session, select

    from factfit.agent.nodes.parse_jd import parse_jd
    from factfit.db import LLMCall, init_db, make_engine
    from factfit.llm import LLMError, make_client

    engine = make_engine()
    init_db(engine)
    run_id = uuid.uuid4().hex
    try:
        raw = Path(path).read_text(encoding="utf-8")
        client = make_client(engine=engine)
        with Session(engine) as session:
            result = parse_jd(
                raw, client=client, session=session, company=company, url=url, run_id=run_id
            )
            stats = session.exec(
                select(
                    func.count(), func.sum(LLMCall.cost_usd), func.sum(LLMCall.latency_ms)
                ).where(LLMCall.run_id == run_id)
            ).one()
    except (OSError, ValueError, LLMError) as e:
        print(f"FAILED: {e}")
        return 1

    jd = result.jd
    print(
        f"{jd.title}  |  {jd.company or '?'}  |  {jd.seniority}  |  {jd.language}  |  {jd.location}"
    )
    print(f"job id: {result.job.id}")
    for label, reqs in (("Must have", jd.must_have), ("Nice to have", jd.nice_to_have)):
        print(f"\n{label} ({len(reqs)}):")
        for r in reqs:
            print(f"  {r.id:>4}  [{r.category}]  {r.text}")
    print(f"\nResponsibilities ({len(jd.responsibilities)}):")
    for item in jd.responsibilities:
        print(f"  - {item}")
    print(f"\nKeywords: {', '.join(jd.keywords)}")

    if result.cached:
        print("\n(cached: same JD and prompt version as before, no LLM call)")
    else:
        calls, cost, latency = stats
        cost_text = f"${cost:.5f}" if cost is not None else "unknown"
        print(f"\nLLM calls: {calls}, total latency: {latency} ms, cost: {cost_text}")
    return 0


def _llm_check() -> int:
    from sqlmodel import Session, select

    from factfit.db import LLMCall, init_db, make_engine
    from factfit.llm import LLMError, make_client
    from factfit.schemas.base import Strict

    class Ping(Strict):
        reply: str

    engine = make_engine()
    init_db(engine)
    try:
        client = make_client(engine=engine)
        result = client.parse(
            Ping,
            node="llm_check",
            prompt_version="llm_check/v1",
            system="You are a health check. Answer briefly.",
            user="Reply with the single word: ok",
        )
    except LLMError as e:
        print(f"FAILED: {e}")
        return 1

    with Session(engine) as session:
        call = session.exec(select(LLMCall).order_by(LLMCall.id.desc())).first()
    cost = "unknown (add this model under `pricing` in config/llm.yaml)"
    if call.cost_usd is not None:
        cost = f"${call.cost_usd:.6f}"
    print(f"OK: model {call.model} replied '{result.reply}'")
    print(f"    tokens in/out: {call.tokens_in}/{call.tokens_out}, latency: {call.latency_ms} ms")
    print(f"    cost: {cost}")
    return 0


def _validate_profile(path: str) -> int:
    try:
        profile = load_profile(path)
    except FileNotFoundError:
        print(f"{path}: file not found")
        return 1
    except ProfileError as e:
        print(f"{len(e.issues)} problem(s) found:\n")
        for issue in e.issues:
            print(f"  {issue.format(e.file)}")
        if e.structural:
            print(
                "\nFix these first, then run again: checks for numbers, skill evidence "
                "and duplicate ids only run once the structure is valid."
            )
        return 1

    bullets = sum(len(entry.bullets) for section in SECTIONS for entry in getattr(profile, section))
    print(f"OK: {path} is valid ({bullets} bullets, {len(profile.skills)} skills).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
