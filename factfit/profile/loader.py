"""Load `profile.yaml`, validate it, and report every problem with its line number."""

from pathlib import Path as FilePath

import yaml
from pydantic import ValidationError

from factfit.profile.rules import Issue, Path, check_rules
from factfit.schemas.profile import Profile


class ProfileError(Exception):
    def __init__(self, file: str, issues: list[Issue], structural: bool = False):
        self.file = file
        self.issues = issues
        # Structural errors stop validation before the cross-file rules run.
        self.structural = structural
        super().__init__("\n".join(i.format(file) for i in issues))


def load_profile(file: str | FilePath) -> Profile:
    file = str(file)
    text = FilePath(file).read_text(encoding="utf-8")

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        line = mark.line + 1 if mark else None
        raise ProfileError(file, [Issue((), f"invalid YAML: {e}", line)]) from e

    lines = _line_map(text)

    try:
        profile = Profile.model_validate(data)
    except ValidationError as e:
        issues = [Issue(tuple(err["loc"]), _friendly(err)) for err in e.errors()]
        raise ProfileError(file, _with_lines(issues, lines), structural=True) from e

    issues = check_rules(profile)
    if issues:
        raise ProfileError(file, _with_lines(issues, lines))
    return profile


def _friendly(err: dict) -> str:
    """Replace Pydantic's regex messages with ones a person can act on."""
    if err["type"] != "string_pattern_mismatch":
        return err["msg"]
    field = next((p for p in reversed(err["loc"]) if isinstance(p, str)), "")
    if field in ("start", "end", "date"):
        return (
            "use YYYY-MM (e.g. 2026-03), or YYYY if the month is unknown; 'present' for an end date"
        )
    if field in ("id", "evidence"):
        return "ids use lowercase letters, digits and underscores, starting with a letter"
    return err["msg"]


def _line_map(text: str) -> dict[Path, int]:
    """Map each path in the YAML document, e.g. ("experiences", 0, "id"), to its line."""
    root = yaml.compose(text, Loader=yaml.SafeLoader)
    lines: dict[Path, int] = {}

    def walk(node: yaml.Node, path: Path) -> None:
        if isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                child = path + (key.value,)
                lines[child] = key.start_mark.line + 1
                walk(value, child)
        elif isinstance(node, yaml.SequenceNode):
            for i, item in enumerate(node.value):
                lines[path + (i,)] = item.start_mark.line + 1
                walk(item, path + (i,))

    if root is not None:
        walk(root, ())
    return lines


def _with_lines(issues: list[Issue], lines: dict[Path, int]) -> list[Issue]:
    for issue in issues:
        # Pydantic sometimes adds extra parts to the path (e.g. which union member failed),
        # so drop parts from the end until we find a path that exists in the file.
        path = issue.path
        while path and path not in lines:
            path = path[:-1]
        issue.line = lines.get(path)
    return sorted(issues, key=lambda i: i.line or 0)
