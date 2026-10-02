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
    args = parser.parse_args(argv)

    if args.command == "validate-profile":
        return _validate_profile(args.path)
    if args.command == "llm-check":
        return _llm_check()
    if args.command == "parse-jd":
        return _parse_jd(args.path, args.company, args.url)
    return 2


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
