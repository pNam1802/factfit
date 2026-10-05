"""Render a CV to .tex with Jinja2, then to PDF with tectonic (PRD F6).

tectonic is a self-contained LaTeX engine that downloads the packages it needs on first use.
Install: https://tectonic-typesetting.github.io (one executable on PATH).
"""

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pypdf import PdfReader

from factfit.render.view import CVView

TEMPLATES = Path(__file__).resolve().parent.parent.parent / "templates"


class RenderError(Exception):
    pass


@dataclass
class RenderResult:
    tex_path: Path
    pdf_path: Path
    pages: int


def render_tex(view: CVView, template: str = "cv.tex.j2") -> str:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES),
        # LaTeX is full of {} and %, so Jinja uses other markers (see the template's header).
        block_start_string="<%",
        block_end_string="%>",
        variable_start_string="<<",
        variable_end_string=">>",
        comment_start_string="<#",
        comment_end_string="#>",
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,  # a missing field is an error, not an empty string
        autoescape=False,  # escaping is done in factfit.render.latex, before the template
    )
    return env.get_template(template).render(cv=view)


def compile_pdf(tex: str, out_dir: Path, name: str = "cv", timeout_s: int = 180) -> RenderResult:
    exe = shutil.which("tectonic")
    if exe is None:
        raise RenderError(
            "tectonic not found on PATH. Install it from https://tectonic-typesetting.github.io"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_path = out_dir / f"{name}.tex"
    tex_path.write_text(tex, encoding="utf-8", newline="\n")

    proc = subprocess.run(
        [exe, "--keep-logs", "--outdir", str(out_dir), str(tex_path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )
    pdf_path = out_dir / f"{name}.pdf"
    if proc.returncode != 0 or not pdf_path.exists():
        raise RenderError(_first_error(proc.stdout + proc.stderr))
    return RenderResult(tex_path, pdf_path, len(PdfReader(pdf_path).pages))


def _first_error(log: str) -> str:
    """The LaTeX error line and the line after it, which shows where it happened."""
    lines = log.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("!") or re.match(r"^error:", line):
            return "\n".join(lines[i : i + 3])
    return log[-1500:] or "tectonic failed without output"


def render_and_check(profile, cv, out_dir: Path, jd_keywords: list[str] | None = None):
    """Render, compile and run the ATS check: what every exported CV goes through."""
    from factfit.render.ats import check_pdf
    from factfit.render.view import build_view

    result = compile_pdf(render_tex(build_view(profile, cv)), out_dir)
    return result, check_pdf(result.pdf_path, profile, cv, jd_keywords)
