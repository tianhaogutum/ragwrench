"""Small argparse CLI; 0 = success, 1 = parity mismatch, 2 = invalid input/backend failure."""
from __future__ import annotations

import argparse
import json
import sys

from . import SUBCOMMANDS, __version__, engine
from .model import ParityError, StoreURI


def positive(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an integer") from exc
    if not 1 <= result <= 4096:
        raise argparse.ArgumentTypeError("expected an integer in [1, 4096]")
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ragwrench migrate", description="Offline Chroma <-> Qdrant migration and parity checks")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("copy", "verify", "inspect"):
        cmd = sub.add_parser(name, help=SUBCOMMANDS[name])
        cmd.add_argument("source", help="chroma:PATH#collection or qdrant:PATH#collection")
        if name != "inspect":
            cmd.add_argument("target")
        cmd.add_argument("--batch-size", type=positive, default=256)
        cmd.add_argument("--vector-name", help="select the sole named Qdrant dense vector")
        cmd.add_argument("--json", action="store_true", help="emit a machine-readable JSON report")
        if name == "copy":
            cmd.add_argument("--checkpoint", help="save resumable progress to a new JSON file outside both stores")
            cmd.add_argument("--resume", action="store_true", help="resume the supplied checkpoint after validating source and acknowledged target records")
            cmd.add_argument("--top-k", type=positive, default=10, help="top-k for automatic checkpoint parity verification")
            cmd.add_argument("--dry-run", action="store_true", help="validate snapshots without creating or changing target")
        if name == "verify":
            cmd.add_argument("--tolerance", type=float, default=1e-5, help="maximum absolute vector difference")
            cmd.add_argument("--sample-queries", type=int, default=10, help="deterministic sample size; 0 explicitly skips retrieval")
            cmd.add_argument("--top-k", type=positive, default=10)
            cmd.add_argument("--min-overlap", type=float, default=1.0, help="minimum mean top-k set overlap in [0,1]")
            cmd.add_argument("--seed", type=int, default=0)
    return p


def text_report(report: dict) -> str:
    if report["operation"] == "copy":
        prefix = f"dry-run: would copy {report['would_copy']} points" if report["dry_run"] else f"copied {report['copied']} points"
        result = f"{prefix}; dimension={report['dimension']} distance={report['distance']}\nmapping: {json.dumps(report['mapping'], sort_keys=True)}"
        if "checkpoint" in report:
            result += f"\nskipped {report['skipped']} acknowledged points; checkpoint={report['checkpoint']}\n" + text_report(report["parity"])
        return result
    if report["operation"] == "inspect":
        return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    retrieval = report["top_k"]
    top = f"top-{retrieval['k']} overlap mean={retrieval['overlap_mean']} ({retrieval['samples']} queries)" if retrieval["checked"] else "retrieval NOT CHECKED"
    return (f"count {'OK' if report['count_ok'] else 'MISMATCH'} ({report['source_count']}/{report['target_count']}), "
            f"ids {'OK' if report['ids_ok'] else 'MISMATCH'}, schema {'OK' if report['schema_ok'] else 'MISMATCH'}\n"
            f"max_abs_diff={report['max_abs_diff']:.8g} tolerance={report['tolerance']}\n"
            f"mismatches: {json.dumps(report['mismatch_counts'], sort_keys=True)}\n{top}\n"
            + ("PARITY OK" if report["ok"] else "PARITY MISMATCH"))


def main(argv: list[str] | None = None) -> int:
    args = vars(parser().parse_args(argv))
    command = args.pop("command")
    json_output = args.pop("json")
    try:
        args["source"] = StoreURI.parse(args["source"])
        if "target" in args:
            args["target"] = StoreURI.parse(args["target"])
        if command == "inspect":
            args["uri"] = args.pop("source")
        report = getattr(engine, command)(**args)
    except (ParityError, OSError) as exc:
        message = str(exc)
        print(json.dumps({"error": message}) if json_output else f"error: {message}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Do not echo backend exception strings that may contain arbitrary payload/document data.
        message = f"Backend operation failed ({type(exc).__name__}); check store format, free disk space and backend compatibility"
        print(json.dumps({"error": message}) if json_output else f"error: {message}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, allow_nan=False) if json_output else text_report(report))
    parity = report if command == "verify" else report.get("parity")
    return 1 if parity is not None and not parity["ok"] else 0
