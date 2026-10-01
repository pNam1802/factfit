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
    args = parser.parse_args(argv)

    if args.command == "validate-profile":
        return _validate_profile(args.path)
    if args.command == "llm-check":
        return _llm_check()
    return 2


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
