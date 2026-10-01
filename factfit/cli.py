"""Command line entry point: `factfit <command>`."""

import argparse
import sys

from factfit.profile import ProfileError, load_profile


def main(argv: list[str] | None = None) -> int:
    # Windows consoles may default to a legacy code page; make Vietnamese text printable.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(prog="factfit")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-profile", help="check a profile.yaml for errors")
    validate.add_argument("path", nargs="?", default="data/profile.yaml")
    args = parser.parse_args(argv)

    if args.command == "validate-profile":
        return _validate_profile(args.path)
    return 2


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

    bullets = sum(
        len(entry.bullets)
        for section in ("experiences", "projects", "education", "publications", "awards")
        for entry in getattr(profile, section)
    )
    print(f"OK: {path} is valid ({bullets} bullets, {len(profile.skills)} skills).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
