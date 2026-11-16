"""CLI exit codes: 0 gate passes, 1 gate findings, 2 input/usage/output error."""
import argparse
import json
import sys

from . import __version__
from .adapters import FORMATS, adapt
from .benchmark import run_benchmark
from .checker import check, fails, markdown_report
from .io import read_payloads, write_output
from .model import Config, InputError


def add_thresholds(parser):
    parser.add_argument("--support-threshold", type=float, default=0.5)
    parser.add_argument("--misattribution-margin", type=float, default=0.3)
    parser.add_argument("--best-source-threshold", type=float, default=0.65)
    parser.add_argument("--min-claim-tokens", type=int, default=2)
    parser.add_argument("--strict-quotes", action="store_true",
                        help="report normalization-only quote matches (CC008) as errors; unsupported quotes remain CC006")


def make_parser():
    parser = argparse.ArgumentParser(prog="ragwrench cite-check", description="Offline structural and lexical RAG citation diagnostics")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    lint = sub.add_parser("check", help="check a complete exported JSON/JSONL response")
    lint.add_argument("input", help="UTF-8 JSON file, or '-' for stdin")
    lint.add_argument("--format", choices=FORMATS, default="generic")
    lint.add_argument("--jsonl", action="store_true", help="one complete response per nonblank line")
    lint.add_argument("--report", choices=("json", "markdown"), default="markdown")
    lint.add_argument("--fail-on", choices=("error", "warning", "hint", "info", "none"), default="error")
    lint.add_argument("-o", "--output", help="report path below cwd (default stdout)")
    lint.add_argument("--force", action="store_true", help="replace an existing output report")
    add_thresholds(lint)
    bench = sub.add_parser("bench", help="replay labelled synthetic cases and save measured comparison")
    bench.add_argument("--cases", help="custom labelled JSONL; default bundled corpus")
    bench.add_argument("-o", "--output", default="citecheck-bench.json", help="report below cwd; default citecheck-bench.json")
    bench.add_argument("--force", action="store_true", help="replace an existing report")
    add_thresholds(bench)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        config = Config(args.support_threshold, args.misattribution_margin, args.best_source_threshold,
                        args.min_claim_tokens, strict_quotes=args.strict_quotes)
        if args.command == "check":
            payloads = read_payloads(args.input, args.jsonl)
            reports = []
            for n, payload in enumerate(payloads, 1):
                try:
                    reports.append(check(adapt(payload, args.format), config))
                except InputError as exc:
                    raise InputError(f"record {n}: {exc}") from None
            result = {"schema_version": 1, "kind": "citecheck-report", "format": args.format,
                      "record_count": len(reports), "fail_on": args.fail_on,
                      "gate_failed": fails(reports, args.fail_on), "records": reports}
            content = json.dumps(result, ensure_ascii=False, indent=2) if args.report == "json" else markdown_report(reports)
            if args.output:
                write_output(args.output, content + "\n", force=args.force, inputs=(args.input,))
            else:
                print(content)
            return 1 if result["gate_failed"] else 0
        result = run_benchmark(args.cases, config)
        write_output(args.output, json.dumps(result, ensure_ascii=False, indent=2) + "\n", force=args.force,
                     inputs=(args.cases,) if args.cases else ())
        micro, base = result["metrics"]["micro"], result["baseline_metrics"]["micro"]
        print(f"{result['case_count']} cases; checker precision={micro['precision']} recall={micro['recall']} F1={micro['f1']}; "
              f"marker-exists baseline F1={base['f1']}; report: {args.output}")
        return 0
    except BrokenPipeError:
        return 2
    except (InputError, OSError, RecursionError) as exc:
        # OSError strings can contain untrusted filenames. Do not echo them (or
        # arbitrary JSON values) into terminal diagnostics.
        message = str(exc) if isinstance(exc, InputError) else "could not read/write local file or input exceeds nesting limits"
        print(f"ragwrench cite-check: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
