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
    tl = commands.add_parser("tailor", help="select and rewrite bullets for a job, with checks")
    tl.add_argument("path", help="text file with the job description")
    tl.add_argument("--profile", default="data/profile.yaml")
    tl.add_argument("--company", help="company name, if the JD does not say")
    tl.add_argument("--level", choices=["intern", "fresher", "junior", "mid", "senior"])
    tl.add_argument("--no-judge", action="store_true", help="rule checks only (no LLM judge)")
    dev = commands.add_parser("dev", help="run the API and the web UI together")
    dev.add_argument("--api-port", type=int, default=8000)
    dev.add_argument("--ui-port", type=int, default=3000)
    rd = commands.add_parser("render", help="render a CV to LaTeX and PDF")
    rd.add_argument("--profile", default="data/profile.yaml")
    rd.add_argument("--cv", help="tailored CV JSON (from factfit tailor); default: whole profile")
    rd.add_argument("--out", default="output/render", help="folder for cv.tex and cv.pdf")
    rd.add_argument(
        "--unchecked",
        action="store_true",
        help="preview a profile that fails the rules (structure is still checked); "
        "not allowed with --cv",
    )
    oa = commands.add_parser("export-openapi", help="write the API schema for the UI types")
    oa.add_argument("path", nargs="?", default="ui/openapi.json")
    cb = commands.add_parser("check-bullet", help="check a rewritten bullet against its sources")
    cb.add_argument("--source", action="append", required=True, help="source bullet id (repeat)")
    cb.add_argument("--text", required=True, help="the rewritten bullet")
    cb.add_argument("--profile", help="default: $FACTFIT_PROFILE or data/profile.yaml")
    commands.add_parser(
        "build-grounding-cases",
        help="(re)build evals/grounding/cases.jsonl; overwrites it, review the result after",
    )
    eg = commands.add_parser("eval-grounding", help="score grounding checks on the test set")
    eg.add_argument(
        "--judges",
        nargs="*",
        default=[],
        help="LLM judge configs to compare, as model:effort (e.g. gpt-5-mini:low)",
    )
    eg.add_argument("--show-misses", metavar="CHECKER", help="print misses of one checker")
    eg.add_argument("--write-report", action="store_true", help="write evals/grounding/RESULTS.md")
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
    if args.command == "tailor":
        return _tailor(args.path, args.profile, args.company, args.level, not args.no_judge)
    if args.command == "dev":
        from factfit.devserver import run_dev

        return run_dev(api_port=args.api_port, ui_port=args.ui_port)
    if args.command == "build-grounding-cases":
        return _build_grounding_cases()
    if args.command == "eval-grounding":
        return _eval_grounding(args.show_misses, args.judges, args.write_report)
    if args.command == "check-bullet":
        return _check_bullet(args.source, args.text, args.profile)
    if args.command == "render":
        return _render(args.profile, args.cv, args.out, args.unchecked)
    if args.command == "export-openapi":
        from factfit.api.app import export_openapi

        export_openapi(args.path)
        print(f"Wrote {args.path}. Regenerate UI types with: cd ui && npm run gen:api")
        return 0
    return 2


def _build_grounding_cases() -> int:
    from collections import Counter

    from factfit.evals import grounding_cases as gc
    from factfit.llm import LLMError, make_client

    sources = gc.load_sources()
    code = gc.code_mutations(sources)
    try:
        llm = gc.llm_variants(sources, make_client())
    except LLMError as e:
        print(f"FAILED: {e}")
        return 1

    # Words-only fabrications that also changed numbers, tech or role test the wrong thing.
    kept = []
    for case in llm:
        problems = gc.impure(case, sources) if case.label == "fabricated" else []
        if problems:
            print(f"dropped impure {case.mutation_type} for {case.source_ids[0]}: {problems}")
        else:
            kept.append(case)

    cases = gc.number_ids(code + kept)
    gc.save_cases(cases)
    counts = Counter(c.mutation_type for c in cases)
    print(f"\nWrote {len(cases)} cases to evals/grounding/cases.jsonl:")
    for kind, n in counts.items():
        print(f"  {kind:<22}{n}")
    print("\nNext: review the llm cases (labels must be right), then run factfit eval-grounding.")
    return 0


def _eval_grounding(show_misses: str | None, judges: list[str], write_report: bool) -> int:
    from factfit.evals import grounding_cases as gc
    from factfit.evals import grounding_eval as ge
    from factfit.llm import LLMError, make_client

    sources = gc.load_sources()
    cases = gc.load_cases()
    checkers: dict[str, ge.Checker] = {"rules": ge.rules_checker}
    costs: dict[str, dict] = {"rules": {"cost_per_1000": 0.0, "latency_p50_s": 0.0}}

    if judges:
        try:
            client = make_client()
        except LLMError as e:
            print(f"FAILED: {e}")
            return 1
        print(f"Judging {len(cases)} cases with {len(judges)} config(s)...")
        for spec in judges:
            cfg = ge.JudgeConfig.parse(spec)
            verdicts = ge.judge_cases(
                cases, sources, cfg, backend=client.backend, base_config=client.config,
                progress=print,
            )  # fmt: skip
            checkers[f"judge {cfg.label}"] = ge.judge_checker(verdicts)
            checkers[f"rules+judge {cfg.label}"] = ge.combined_checker(verdicts)
            costs[f"judge {cfg.label}"] = costs[f"rules+judge {cfg.label}"] = ge.cost_and_latency(
                verdicts
            )

    results = {name: ge.evaluate(cases, sources, fn) for name, fn in checkers.items()}
    names = list(checkers)
    short = {n: f"C{i}" for i, n in enumerate(names)}

    print(f"\nGrounding test set: {len(cases)} cases. Columns:")
    for n in names:
        print(f"  {short[n]} = {n}")
    header = f"\n{'type':<22}{'n':>4}" + "".join(f"{short[n]:>7}" for n in names)
    print(header)
    rows = list(zip(*(results[n] for n in names), strict=True))
    for row in rows:
        print(f"{row[0].mutation_type:<22}{row[0].n:>4}" + "".join(f"{r.rate:>7.0%}" for r in row))
    sums = {n: ge.summary(results[n]) for n in names}
    print(f"{'recall (fabricated)':<26}" + "".join(f"{sums[n]['recall']:>7.0%}" for n in names))
    print(
        f"{'false positives':<26}"
        + "".join(f"{sums[n]['false_positive_rate']:>7.0%}" for n in names)
    )
    print(
        f"{'$ per 1000 bullets':<26}"
        + "".join(f"{_money(costs[n]['cost_per_1000']):>7}" for n in names)
    )
    print("(rows = share flagged; for the two paraphrase rows, flagged means a false alarm)")

    if show_misses:
        target = next((n for n in names if show_misses in n), None)
        for r in results.get(target, []):
            for case in r.misses:
                src = " | ".join(sources[s].text for s in case.source_ids)
                print(f"\n[{target}] [{r.mutation_type}] {case.id}\n  source:  {src}")
                print(f"  rewrite: {case.text}")

    if write_report:
        _write_grounding_report(cases, names, results, sums, costs)
    return 0


def _money(value: float | None) -> str:
    return "?" if value is None else f"{value:.2f}"


def _write_grounding_report(cases, names, results, sums, costs) -> None:
    from datetime import date
    from pathlib import Path

    lines = [
        "# Grounding results",
        "",
        f"Generated by `uv run factfit eval-grounding --write-report` on {date.today()}, "
        f"on {len(cases)} cases (see [README](README.md) for how the set was built).",
        "",
        "Share of cases flagged as not grounded. For fabricated types higher is better "
        "(recall); for the two paraphrase types lower is better (false alarms). "
        "Brackets: 95% Wilson interval.",
        "",
        "| type | n | " + " | ".join(names) + " |",
        "| --- | ---: | " + " | ".join("---" for _ in names) + " |",
    ]
    for row in zip(*(results[n] for n in names), strict=True):
        cells = []
        for r in row:
            lo, hi = r.interval or (0, 0)
            cells.append(f"{r.rate:.0%} ({lo:.0%}-{hi:.0%})")
        lines.append(f"| {row[0].mutation_type} | {row[0].n} | " + " | ".join(cells) + " |")
    lines.append(
        "| **recall, all fabricated** | | "
        + " | ".join(f"**{sums[n]['recall']:.0%}**" for n in names)
        + " |"
    )
    lines.append(
        "| **false alarms, all paraphrases** | | "
        + " | ".join(f"**{sums[n]['false_positive_rate']:.0%}**" for n in names)
        + " |"
    )
    lines.append(
        "| $ per 1,000 bullets | | "
        + " | ".join(_money(costs[n]["cost_per_1000"]) for n in names)
        + " |"
    )
    lines.append(
        "| median latency (s) | | "
        + " | ".join(f"{costs[n]['latency_p50_s']:.1f}" for n in names)
        + " |"
    )
    path = Path("evals/grounding/RESULTS.md")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"\nWrote {path}")


def _check_bullet(source_ids: list[str], text: str, profile_path: str | None) -> int:
    import os

    from dotenv import load_dotenv

    from factfit.grounding import check_bullet
    from factfit.profile import ProfileError, load_profile

    load_dotenv()
    path = profile_path or os.environ.get("FACTFIT_PROFILE", "data/profile.yaml")
    try:
        profile = load_profile(path)
    except (ProfileError, FileNotFoundError):
        print(f"{path} is missing or not valid. Run: uv run factfit validate-profile {path}")
        return 1

    bullets = {b.id: b for s in SECTIONS for e in getattr(profile, s) for b in e.bullets}
    unknown = [i for i in source_ids if i not in bullets]
    if unknown:
        print(f"Unknown bullet id(s) in {path}: {', '.join(unknown)}")
        return 1
    sources = [bullets[i] for i in source_ids]

    for b in sources:
        print(f"source {b.id}: {b.text}")
    print(f"rewrite:   {text}\n")
    issues = check_bullet(text, sources)
    if not issues:
        print("PASS: nothing added that the sources do not support (numbers, tech, role).")
        return 0
    print(f"FAIL: {len(issues)} issue(s)")
    for i in issues:
        print(f"  [{i.kind}] {i.message}")
    return 1


def _tailor(
    path: str, profile_path: str, company: str | None, level: str | None, use_judge: bool
) -> int:
    import json
    import re
    import uuid
    from pathlib import Path

    from sqlalchemy import func
    from sqlmodel import Session, select

    from factfit.agent.nodes.match import match_job
    from factfit.agent.nodes.parse_jd import parse_jd
    from factfit.agent.tailor import tailor
    from factfit.db import LLMCall, init_db, make_engine
    from factfit.llm import LLMError, make_client
    from factfit.profile import ProfileError, load_profile, profile_version

    try:
        profile = load_profile(profile_path)
    except (ProfileError, FileNotFoundError):
        print(f"{profile_path} is missing or not valid. Run: uv run factfit validate-profile")
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
            applicant_level = level or (jd.seniority if jd.seniority != "unknown" else None)
            matched = match_job(
                parsed.job, profile, profile_version=profile_version(profile_path),
                client=client, session=session, level=applicant_level, run_id=run_id,
            )  # fmt: skip
            run = tailor(
                profile, jd, matched.result, client=client, use_judge=use_judge,
                run_id=run_id, progress=lambda m: print(f"  {m}"),
            )  # fmt: skip
            calls, cost, latency = session.exec(
                select(
                    func.count(), func.sum(LLMCall.cost_usd), func.sum(LLMCall.latency_ms)
                ).where(LLMCall.run_id == run_id)
            ).one()
    except (OSError, ValueError, LLMError) as e:
        print(f"FAILED: {e}")
        return 1

    print(f"\n{jd.title} | {jd.company or '?'} | match score {matched.result.score}/100")
    by_entry: dict[int, list] = {}
    for d in run.drafts:
        by_entry.setdefault(id(d.entry), []).append(d)
    for entry in run.selected:
        print(f"\n== {entry.title}")
        for d in by_entry.get(id(entry), []):
            if d.fallback:
                status = "FALLBACK (rewrites failed, original kept)"
            else:
                status = "pass" + (" after retry" if d.attempts > 1 else "")
            print(f"  [{d.source.id}] {status}")
            print(f"    original: {d.source.text}")
            if d.text != d.source.text:
                print(f"    tailored: {d.text}")
            for text, problems in d.history:
                print(f"    rejected: {text}")
                for p in problems:
                    print(f"      - {p}")
    if run.cv.summary:
        print(f"\nSummary ({run.cv.summary.source_ids[0]}): {run.cv.summary.text}")
    print(f"Skills: {', '.join(run.cv.skills)}")

    rewritten = sum(d.text != d.source.text for d in run.drafts)
    retried = sum(d.attempts > 1 and not d.fallback for d in run.drafts)
    fallbacks = sum(d.fallback for d in run.drafts)
    print(
        f"\n{len(run.drafts)} bullets: {rewritten} rewritten, {retried} passed after a retry, "
        f"{fallbacks} fell back to the original"
    )
    cost_text = f"${cost:.4f}" if cost is not None else "unknown"
    print(f"LLM calls: {calls}, summed latency: {latency} ms, cost: {cost_text}")

    out = Path("output") / f"tailored-{re.sub(r'[^a-z0-9]+', '-', jd.title.lower())[:40]}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(run.cv.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Tailored CV JSON: {out}")
    return 0


def _render(profile_path: str, cv_path: str | None, out: str, unchecked: bool) -> int:
    import json
    from pathlib import Path

    from factfit.profile import ProfileError, load_profile
    from factfit.render.compile import RenderError, compile_pdf, render_tex
    from factfit.render.view import build_view, full_profile_cv
    from factfit.schemas.tailored import TailoredCV

    if unchecked and cv_path:
        print("--unchecked is only for previewing a profile, not for a tailored CV.")
        return 1
    try:
        profile = load_profile(profile_path, check=not unchecked)
    except ProfileError as e:
        print(f"{profile_path}: {len(e.issues)} problem(s). Run: uv run factfit validate-profile")
        if not e.structural:
            print("To preview it anyway: uv run factfit render --unchecked")
        return 1
    if unchecked:
        print("WARNING: preview of an unchecked profile; do not send this CV.")

    if cv_path:
        cv = TailoredCV.model_validate(json.loads(Path(cv_path).read_text(encoding="utf-8")))
        if not cv.ready_to_export():
            print("This CV still has bullets that failed or were never checked; not exported.")
            return 1
    else:
        cv = full_profile_cv(profile)

    try:
        result = compile_pdf(render_tex(build_view(profile, cv)), Path(out))
    except RenderError as e:
        print(f"FAILED to compile:\n{e}")
        return 1
    print(f"Wrote {result.tex_path} and {result.pdf_path} ({result.pages} page(s))")
    # The one-page limit applies to a tailored CV to send (PRD F6), not to a full preview.
    if cv_path and result.pages != 1:
        print("FAILED: a tailored CV must fit on 1 page; select fewer bullets.")
        return 1
    return 0


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
