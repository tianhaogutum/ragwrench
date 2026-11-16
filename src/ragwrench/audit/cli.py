"""``ragwrench audit`` command line. Exit 0 clean, 1 findings at/above --fail-on, 2 usage/input error."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from . import SCHEMA, __version__
from ..lint.io import load_config
from ..lint.model import InputError
from ..lint.report import sarif
from .core import audit_file, find_inputs, fixes, summarize, worst_chunks

EPILOG = """\
exit status: 0 no finding at or above --fail-on, 1 findings at or above it,
2 usage or input error. Read-only: nothing is written; the repair preview runs
`ragwrench lint fix` in memory.

examples:
  ragwrench audit chunks.jsonl
  ragwrench audit chunks.jsonl --source handbook.md      # also checks coverage
  ragwrench audit exports/ --fail-on warning --format sarif > audit.sarif
"""


class Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # usage errors exit 2, like the other tools
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n")


def parser() -> argparse.ArgumentParser:
    cli = Parser(prog="ragwrench audit", formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EPILOG,
                 description="One read-only health report for a RAG chunk export: what is broken, "
                             "where, and the command that fixes it.")
    cli.add_argument("--version", action="version", version=f"ragwrench audit {__version__}")
    cli.add_argument("input", type=Path, help="chunk export (JSONL or JSON array), or a directory of them")
    cli.add_argument("--source", type=Path, help="original document the chunks came from (single-file input only)")
    cli.add_argument("--format", choices=("text", "json", "sarif"), default="text")
    cli.add_argument("--fail-on", choices=("error", "warning", "note", "none"), default=None,
                     help="lowest severity that makes the exit status 1 (default: error, or chunklint.toml)")
    cli.add_argument("--adapter", choices=("auto", "generic", "langchain", "llamaindex"), default="auto")
    cli.add_argument("--config", type=Path, help="chunklint.toml (default: ./chunklint.toml if present)")
    cli.add_argument("--disable", action="append", default=[], help="comma-separated rule IDs; repeatable")
    cli.add_argument("--min-chars", type=int)
    cli.add_argument("--max-chars", type=int)
    cli.add_argument("--worst", type=int, default=5, metavar="N", help="worst chunks to list (default 5)")
    cli.add_argument("--no-color", action="store_true", help="disable color (also: NO_COLOR, non-TTY stdout)")
    return cli


class Style:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def _wrap(self, code: str, text: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.enabled and text else text

    def bold(self, t): return self._wrap("1", t)
    def dim(self, t): return self._wrap("2", t)
    def red(self, t): return self._wrap("31", t)
    def green(self, t): return self._wrap("32", t)
    def yellow(self, t): return self._wrap("33", t)
    def cyan(self, t): return self._wrap("36", t)


def _flags(args) -> list[str]:
    """Options to repeat in suggested commands so they see the same configuration."""
    flags = []
    if args.adapter != "auto":
        flags += ["--adapter", args.adapter]
    if args.config:
        flags += ["--config", str(args.config)]
    for group in args.disable:
        flags += ["--disable", group]
    for name in ("min_chars", "max_chars"):
        if getattr(args, name) is not None:
            flags += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    return flags


def build(audits, skipped, args, fail_on) -> dict:
    flags = _flags(args)
    per_file = []
    all_fixes = []
    for audit in audits:
        steps = fixes(audit, flags)
        all_fixes += steps
        per_file.append({
            "path": str(audit.path), "source": str(audit.source) if audit.source else None,
            "chunks": len(audit.chunks), "stats": audit.report.stats, "auto_fixable": audit.fixable(),
            "fix_error": audit.fix_error, "notes": audit.notes, "gaps": audit.gaps,
            "issues": [issue.record() for issue in audit.report.issues], "fixes": steps})
    order = {"lint-fix": 0, "coverage": 1, "table-chunk": 2, "splitter": 3}
    all_fixes.sort(key=lambda f: (order[f["kind"]], -f["resolves"]))
    return {"schema": SCHEMA, "ragwrench_version": __version__, "input": str(args.input),
            "summary": summarize(audits, fail_on), "worst_chunks": worst_chunks(audits, max(args.worst, 0)),
            "top_fixes": all_fixes[:4], "files": per_file, "skipped": skipped}


def render_text(doc: dict, s: Style) -> str:
    out, summary, files = [], doc["summary"], doc["files"]
    multi = len(files) > 1
    out.append(s.bold("ragwrench audit") + "  " + doc["input"])
    facts = [f"{summary['chunks']:,} chunks", f"{summary['characters']:,} characters"]
    if multi:
        facts.insert(0, f"{summary['files']} files")
    covs = [f for f in files if f["source"]]
    if len(covs) == 1:
        facts.append(f"source {covs[0]['source']} ({covs[0]['stats']['source']['coverage']:.1%} covered)")
    out.append("  " + s.dim(" · ".join(facts)))
    out.append("")
    out.append(s.bold("Findings") + s.dim(f"{'':<27}rule   found  auto-fix  why it hurts retrieval"))
    for cat in summary["categories"]:
        rule, found = cat["rule"], cat["found"]
        skipped_rule = rule == "CL007" and not covs
        if found:
            mark = s.red("✗") if cat["severity"] == "error" else s.yellow("!")
            count = (s.red if cat["severity"] == "error" else s.yellow)(f"{found:>5}")
            auto = f"{cat['auto_fixable']:>8}" if cat["auto_fixable"] else s.dim(f"{'-':>8}")
            out.append(f"  {mark} {cat['name']:<30} {rule}  {count}  {auto}  {s.dim(cat['why'])}")
        else:
            tail = "needs --source" if skipped_rule else "ok"
            out.append(s.dim(f"  ✓ {cat['name']:<30} {rule}  {0:>5}  {'':>8}  {tail}"))
    if doc["worst_chunks"]:
        out += ["", s.bold("Worst chunks")]
        width = max(len((w["file"] + ":" if multi else "") + w["chunk_id"]) for w in doc["worst_chunks"])
        rwidth = max(len(" ".join(w["rules"])) for w in doc["worst_chunks"])
        for w in doc["worst_chunks"]:
            name = (w["file"] + ":" if multi else "") + w["chunk_id"]
            out.append(f"  {name:<{width}}  {' '.join(w['rules']):<{rwidth}}  {s.dim(chr(34) + w['preview'] + chr(34))}")
    gaps = [(f, g) for f in files for g in f["gaps"]]
    if gaps:
        out += ["", s.bold("Source text missing from every chunk")]
        for f, g in gaps[:5]:
            lines = f"{g['line_start']}" + (f"-{g['line_end']}" if g["line_end"] != g["line_start"] else "")
            out.append(f"  {s.red(f['source'] + ':' + lines)}  {s.dim(chr(34) + g['preview'] + chr(34))}")
    notes = [(f, n) for f in files for n in f["notes"]]
    for f, note in notes:
        out.append(s.yellow(f"  note: {f['path']}: {note}"))
    if doc["skipped"]:
        out += ["", s.bold("Skipped")]
        out += [s.dim(f"  {item['path']}: {item['error']}") for item in doc["skipped"]]
    out += ["", s.bold("Top fixes")]
    if not doc["top_fixes"]:
        out.append("  " + s.green("Nothing to fix."))
    for index, step in enumerate(doc["top_fixes"], 1):
        where = f" ({step['path']})" if multi else ""
        out.append(f"  {index}. {step['title']}{where}")
        out.append(f"     {s.dim(step['detail'])}")
        out += [f"     {s.cyan(command)}" for command in step["commands"]]
    out.append("")
    total = summary["findings"]
    if total:
        head = (s.red if summary["errors"] else s.yellow)(f"{total:,} findings")
        line = (f"{head} in {summary['chunks_with_findings']:,} of {summary['chunks']:,} chunks: "
                f"{summary['errors']:,} errors, {summary['warnings']:,} warnings; "
                f"{summary['auto_fixable']:,} fixable automatically.")
    else:
        line = s.green(f"No findings in {summary['chunks']:,} chunks.")
    verdict = s.red("FAIL (exit 1)") if summary["failed"] else s.green("pass (exit 0)")
    out.append(line)
    out.append(f"--fail-on {summary['fail_on']}: {verdict}")
    if not covs:
        out.append(s.dim("Add --source <document> to check source coverage (CL007)."))
    return "\n".join(out) + "\n"


def render_sarif(audits) -> dict:
    runs = [sarif(audit.report, audit.path, audit.source)["runs"][0] for audit in audits]
    return {"$schema": "https://json.schemastore.org/sarif-2.1.0.json", "version": "2.1.0", "runs": runs}


def _use_color(args) -> bool:
    if args.no_color or args.format != "text" or os.environ.get("NO_COLOR"):
        return False
    try:
        return sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if args.worst < 0:
        cli.error("--worst must be >= 0")
    try:
        config = load_config(args.config)
        for name in ("min_chars", "max_chars", "fail_on"):
            if getattr(args, name) is not None:
                setattr(config, name, getattr(args, name))
        config.disable = sorted(set(config.disable + [x for group in args.disable for x in group.split(",") if x]))
        config.validate()
        inputs = find_inputs(args.input)
        if args.source is not None and (len(inputs) > 1 or args.input.is_dir()):
            raise InputError("--source needs a single chunk file, not a directory")
        if args.source is not None and not args.source.is_file():
            raise InputError(f"{args.source}: source must be a regular file")
        audits, skipped = [], []
        for path in inputs:
            try:
                audits.append(audit_file(path, args.source, config, args.adapter))
            except (InputError, UnicodeError, RecursionError) as exc:
                if len(inputs) == 1:
                    raise
                skipped.append({"path": str(path), "error": str(exc).removeprefix(f"{path}: ")})
        if not audits:
            raise InputError(f"{args.input}: none of {len(inputs)} files is a readable chunk export")
    except (InputError, OSError, UnicodeError, RecursionError) as exc:
        print(f"ragwrench audit: {exc}", file=sys.stderr)
        return 2
    doc = build(audits, skipped, args, config.fail_on)
    try:
        if args.format == "json":
            sys.stdout.write(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
        elif args.format == "sarif":
            sys.stdout.write(json.dumps(render_sarif(audits), ensure_ascii=False, indent=2) + "\n")
        else:
            sys.stdout.write(render_text(doc, Style(_use_color(args))))
        sys.stdout.flush()
    except BrokenPipeError:
        try:
            sys.stdout = open(os.devnull, "w")
        except OSError:
            pass
    return 1 if doc["summary"]["failed"] else 0


__all__ = ["main", "parser"]

if __name__ == "__main__":
    raise SystemExit(main())
