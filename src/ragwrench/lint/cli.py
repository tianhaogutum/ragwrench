import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .bench import benchmark
from .fix import fix
from .io import MAX_BYTES, atomic_write, load_chunks, load_config, read_bytes, read_text
from .lint import lint
from .model import InputError
from .plan import create_plan, load_plan, preview_text, validate_plan
from .plan_validation import PlanValidationError, schema_text
from .report import failed, sarif, text_report


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def parser():
    cli = argparse.ArgumentParser(prog="ragwrench lint", description="Offline Markdown chunk QA and conservative repair")
    cli.add_argument("--version", action="version", version=f"ragwrench lint {__version__}")
    commands = cli.add_subparsers(dest="command", required=True)
    for name in ("check", "fix", "plan"):
        command = commands.add_parser(name, help={"check": "report chunk problems against the source; exit 1 at --fail-on",
                                                  "fix": "write conservatively repaired chunks and a change log",
                                                  "plan": "save a reviewable, versioned fix plan without changing chunks"}[name])
        command.add_argument("input", type=Path, help="JSONL or JSON array")
        command.add_argument("--adapter", choices=("auto", "generic", "langchain", "llamaindex"), default="auto")
        command.add_argument("--source", type=Path)
        command.add_argument("--config", type=Path)
        command.add_argument("--disable", action="append", default=[], help="comma-separated rule IDs; repeatable")
        command.add_argument("--min-chars", type=int)
        command.add_argument("--max-chars", type=int)
        command.add_argument("--fail-on", choices=("error", "warning", "note", "none"))
        command.add_argument("--format", choices=("text", "json", "sarif"), default="text")
        if name == "fix":
            command.add_argument("-o", "--output", type=Path)
            command.add_argument("--change-log", type=Path, help="default: OUTPUT.changes.json")
            command.add_argument("--dry-run", action="store_true", help="compute and report; write no files")
        elif name == "plan":
            command.add_argument("-o", "--output", type=Path, required=True, help="save a versioned fix plan")
    for name in ("preview", "apply"):
        command = commands.add_parser(name, help=f"{name} a saved fix plan against its original snapshots")
        command.add_argument("input", type=Path, help="original JSONL or JSON array")
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--source", type=Path, help="required if the plan used a source")
        command.add_argument("--format", choices=("text", "json"), default="text")
        if name == "apply":
            command.add_argument("-o", "--output", type=Path, required=True)
    command = commands.add_parser("validate-plan", help="validate a saved plan without reading chunks/source")
    command.add_argument("plan", type=Path, nargs="?", help="saved fix-plan JSON file")
    command.add_argument("--format", choices=("text", "json"), default=None)
    command.add_argument("--schema", action="store_true", help="print the packaged v1 JSON Schema instead")
    bench = commands.add_parser("bench", help="deterministic before/after experiment on packaged fixtures")
    bench.add_argument("--write", type=Path, help="also save the measured JSON result")
    return cli


def _check_outputs(outputs, protected):
    paths = [p.resolve() for p in outputs]
    if len(set(paths)) != len(paths):
        raise InputError("outputs must be different files (including output and change log)")
    for output, resolved in zip(outputs, paths):
        if resolved in protected:
            raise InputError(f"refusing to overwrite input, source, config or plan: {output}")
        if output.is_symlink():
            raise InputError(f"refusing symlink output: {output}")
        if not output.parent.is_dir() or output.is_dir():
            raise InputError(f"output parent must exist and output must be a file: {output}")


def _protected(args):
    paths = [args.input, getattr(args, "source", None), getattr(args, "plan", None),
             getattr(args, "config", None) or Path("chunklint.toml")]
    return {path.resolve() for path in paths if path is not None}


def _saved_plan(args, config=None):
    if args.command != "plan":
        # Plan diagnostics take precedence over unavailable input/source snapshots.
        plan = load_plan(args.plan)
    input_raw = read_bytes(args.input)
    source_raw = read_bytes(args.source) if args.source is not None else None
    if args.command == "plan":
        plan, result = create_plan(input_raw, source_raw, args.adapter, config)
        _check_outputs([args.output], _protected(args))
        plan_text = _json(plan)
        if len(plan_text.encode("utf-8")) > MAX_BYTES:
            raise InputError(f"fix plan exceeds {MAX_BYTES} byte input limit; no files written")
        atomic_write(args.output, plan_text)
    else:
        encoded, result = validate_plan(plan, input_raw, source_raw)
        if args.command == "apply":
            _check_outputs([args.output], _protected(args))
            # One replacement only: stale/invalid plans never create a log or partial output.
            atomic_write(args.output, encoded)
    value = {"plan": plan, **result.record()}
    if args.format == "json":
        rendered = _json(value)
    elif args.format == "sarif":
        rendered = _json(sarif(result.after, args.input, args.source))
    elif args.command == "apply":
        rendered = f"Applied {len(plan['edits'])} edits to {args.output}\n" + text_report(result.after)
    else:
        rendered = preview_text(plan, input_raw) + text_report(result.after)
    sys.stdout.write(rendered)
    return int(failed(result.after, plan["config"]["fail_on"]))


def _validate_plan(args):
    try:
        plan = load_plan(args.plan)
    except (InputError, OSError, UnicodeError, RecursionError) as exc:
        diagnostic = {"path": exc.path if isinstance(exc, PlanValidationError) else "$",
                      "message": exc.message if isinstance(exc, PlanValidationError) else str(exc)}
        if args.format == "json":
            sys.stdout.write(_json({"valid": False, "diagnostics": [diagnostic]}))
        else:
            print(f"ragwrench lint: invalid fix plan at {diagnostic['path']}: {diagnostic['message']}", file=sys.stderr)
        return 2
    if args.format == "json":
        sys.stdout.write(_json({"valid": True, "version": plan["version"], "edits": len(plan["edits"]),
                               "snapshots_checked": False}))
    else:
        sys.stdout.write(f"Valid fix plan v1: {len(plan['edits'])} edits (snapshots not checked)\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if args.command == "validate-plan":
        if args.schema:
            if args.plan is not None or args.format is not None:
                cli.error("validate-plan --schema accepts no plan or --format")
            sys.stdout.write(schema_text())
            return 0
        if args.plan is None:
            cli.error("validate-plan requires a plan filename or --schema")
        return _validate_plan(args)
    try:
        if args.command == "bench":
            result = benchmark()
            rendered = _json(result)
            if args.write:
                atomic_write(args.write, rendered)
            sys.stdout.write(rendered)
            return 0 if result["idempotent"] and result["after"]["defects"] < result["before"]["defects"] else 1
        if args.command in ("preview", "apply"):
            return _saved_plan(args)
        config = load_config(args.config)
        for name in ("min_chars", "max_chars", "fail_on"):
            if getattr(args, name) is not None:
                setattr(config, name, getattr(args, name))
        config.disable = sorted(set(config.disable + [x for group in args.disable for x in group.split(",")]))
        config.validate()
        if args.command == "plan":
            return _saved_plan(args, config)
        chunks = load_chunks(args.input, args.adapter)
        source = read_text(args.source) if args.source is not None else None
        if args.command == "check":
            report = lint(chunks, source, config)
            value = report.record()
            rendered = text_report(report)
        else:
            if args.output is None and not args.dry_run:
                raise InputError("fix requires --output (or --dry-run)")
            result = fix(chunks, source, config)
            report, value = result.after, result.record()
            rendered = f"{len(result.changes)} changes; {len(result.skipped)} skipped\n" + text_report(report)
            rendered += "Conservation: input non-whitespace preserved except logged exact duplicates; "
            rendered += "source coverage preserved\n" if source is not None else "source coverage not checked\n"
            if not args.dry_run:
                log = args.change_log or Path(str(args.output) + ".changes.json")
                protected = {args.input.resolve()}
                if args.source:
                    protected.add(args.source.resolve())
                protected.add((args.config or Path("chunklint.toml")).resolve())
                _check_outputs([args.output, log], protected)
                encoded = "".join(json.dumps(c.record(), ensure_ascii=False, allow_nan=False) + "\n" for c in result.chunks)
                # Validate everything before either explicit output is written.
                log_text = _json(value)
                atomic_write(log, log_text)
                atomic_write(args.output, encoded)
        if args.format == "json":
            rendered = _json(value)
        elif args.format == "sarif":
            rendered = _json(sarif(report, args.input, args.source))
        sys.stdout.write(rendered)
        return int(failed(report, config.fail_on))
    except (InputError, OSError, UnicodeError, RecursionError) as exc:
        print(f"ragwrench lint: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
